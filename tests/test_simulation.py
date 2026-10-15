from collections.abc import Sequence

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recommender.algorithms.base import Policy, ScoredOffer
from recommender.algorithms.random_policy import RandomPolicy
from recommender.algorithms.thompson import ThompsonSamplingPolicy
from recommender.api.main import create_app
from recommender.config import load_settings
from recommender.customers import CustomerRepository
from recommender.db import make_engine
from recommender.domain import Context, Offer
from recommender.evaluation.compare import comparison_table, run_comparison, warm_up_history
from recommender.evaluation.metrics import cumulative, summarize
from recommender.simulator.engine import World, simulate, simulate_via_api
from recommender.store.options_store import InMemoryOptionsStore


@pytest.fixture(scope="module")
def world() -> World:
    return World.load()


class OraclePolicy(Policy):
    """Test-only: ranks by the true probabilities, so it should have no regret."""

    name = "oracle"

    def __init__(self, world: World) -> None:
        self.world = world

    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        p = self.world.truth.probabilities(context, candidates)
        return [ScoredOffer(o, p[o]) for o in sorted(p, key=p.get, reverse=True)[:n]]


def test_regret_is_never_negative(world: World) -> None:
    log = simulate(RandomPolicy(rng=np.random.default_rng(0)), world, rounds=500, seed=1)
    assert ((log["oracle_take_ups"] - log["expected_take_ups"]) >= -1e-12).all()
    assert ((log["oracle_revenue"] - log["expected_revenue"]) >= -1e-9).all()


def test_the_oracle_has_no_regret(world: World) -> None:
    log = simulate(OraclePolicy(world), world, rounds=500, seed=1)
    assert summarize(log)["regret"] == pytest.approx(0, abs=1e-9)
    assert summarize(log)["share_of_oracle"] == pytest.approx(1.0)


def test_every_policy_faces_the_same_app_opens(world: World) -> None:
    random_log = simulate(RandomPolicy(rng=np.random.default_rng(0)), world, rounds=300, seed=4)
    oracle_log = simulate(OraclePolicy(world), world, rounds=300, seed=4)
    columns = ["segment", "day", "is_payday_week"]
    assert random_log[columns].equals(oracle_log[columns])


def test_learning_closes_the_gap_to_the_oracle(world: World) -> None:
    policy = ThompsonSamplingPolicy(
        "ts", InMemoryOptionsStore(), rng=np.random.default_rng(0), context_mode="segment"
    )
    log = simulate(policy, world, rounds=6000, seed=2)
    first, last = log.iloc[:1000], log.iloc[-1000:]
    gap = lambda part: (part["oracle_take_ups"] - part["expected_take_ups"]).mean()  # noqa: E731
    assert gap(last) < 0.5 * gap(first)


def test_cumulative_curves(world: World) -> None:
    log = simulate(RandomPolicy(rng=np.random.default_rng(0)), world, rounds=100, seed=1)
    curves = cumulative(log, every=10)
    assert list(curves["round"]) == list(range(10, 101, 10))
    assert curves["regret"].is_monotonic_increasing


def test_warm_up_history_feeds_the_static_baseline(world: World) -> None:
    history = warm_up_history(world, rounds=200, seed=0)
    assert len(history) > 200  # roughly 2 noticed offers per app open
    assert {reward for _, reward in history} <= {0.0, 1.0}


def test_comparison_runs_every_policy_on_every_seed(world: World) -> None:
    settings = load_settings()
    comparison = run_comparison(
        settings,
        world,
        rounds=400,
        shift_round=200,
        seeds=[0, 1],
        warm_up_rounds=200,
        snapshot_every=200,
        progress=lambda _: None,
    )
    assert len(comparison.summary) == 2 * len(settings.policies)
    assert set(comparison.rounds["policy"]) == set(settings.policies)
    assert len(comparison.rounds) == 400 * len(settings.policies)  # first seed only
    assert set(comparison.snapshots["round"]) == {200, 400}
    table = comparison_table(comparison.summary)
    assert table["regret"].is_monotonic_increasing
    assert table.loc["random", "regret_vs_random"] == pytest.approx(0)


def test_simulation_through_the_api(world: World, tmp_path) -> None:
    """API mode: real invocations and responses, logged and learned by the API."""
    url = f"sqlite:///{(tmp_path / 'sim.db').as_posix()}"
    engine = make_engine(url)
    CustomerRepository(engine).load(world.customers)
    engine.dispose()
    with TestClient(create_app(database_url=url)) as client:
        log = simulate_via_api(
            client, world, rounds=30, shift_round=15, seed=0, policy="thompson_segment"
        )
        metrics = client.get("/metrics").json()["policies"]["thompson_segment"]
        state = client.get("/policies/thompson_segment/state").json()
    assert len(log) == 30
    assert metrics["invocations"] == 30
    assert sum(metrics["responses"].values()) == 90  # 3 offers per app open
    assert sum(arm["pulls"] for arm in state["arms"]) > 0
