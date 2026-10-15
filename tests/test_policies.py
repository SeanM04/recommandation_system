from collections import Counter

import numpy as np
import pytest

from recommender.algorithms.base import BanditPolicy, context_key
from recommender.algorithms.epsilon_greedy import EpsilonGreedyPolicy
from recommender.algorithms.random_policy import RandomPolicy
from recommender.algorithms.rewards import binary_reward
from recommender.algorithms.static import StaticPolicy
from recommender.algorithms.thompson import ThompsonSamplingPolicy
from recommender.config import load_offers, load_settings
from recommender.domain import Context, Customer, Offer
from recommender.simulator.ground_truth import GroundTruth, take_up_table
from recommender.simulator.population import generate_customers, sample_context
from recommender.simulator.settings import load_simulation_settings
from recommender.store.options_store import ArmStats, InMemoryOptionsStore
from recommender.store.sql import create_store


def make_offers(*offer_ids: str) -> list[Offer]:
    return [
        Offer(
            offer_id=offer_id,
            name=offer_id,
            category="data",
            price_zar=10,
            margin_zar=5,
            eligible_segments=("student", "family"),
        )
        for offer_id in offer_ids
    ]


def make_context(segment: str = "family", payday: bool = False) -> Context:
    return Context(
        customer_id="C00001",
        segment=segment,
        device_tier="mid",
        prepaid_vs_contract="prepaid",
        avg_monthly_spend=300.0,
        province="Gauteng",
        channel="app",
        time_of_day="evening",
        day_of_week="tuesday",
        is_payday_week=payday,
    )


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def play_bandit(policy, offers, true_rates, rounds, seed=1, context=None) -> list[str]:
    """Show one offer per round on a Bernoulli bandit; return what was shown."""
    world = np.random.default_rng(seed)
    context = context or make_context()
    shown = []
    for _ in range(rounds):
        offer_id = policy.rank(context, offers, 1)[0].offer_id
        policy.update(context, offer_id, float(world.random() < true_rates[offer_id]))
        shown.append(offer_id)
    return shown


# --- context keys ---------------------------------------------------------------------


def test_context_keys() -> None:
    assert context_key(make_context("family"), "global") == "all"
    assert context_key(make_context("family"), "segment") == "family"
    assert context_key(make_context("family", payday=True), "segment_payday") == "family|payday"
    assert context_key(make_context("student"), "segment_payday") == "student|rest"
    with pytest.raises(ValueError, match="unknown context mode"):
        context_key(make_context(), "province")


# --- baselines ------------------------------------------------------------------------


def test_random_policy_returns_distinct_offers_uniformly() -> None:
    offers = make_offers("a", "b", "c", "d")
    policy = RandomPolicy(rng=rng())
    ranked = policy.rank(make_context(), offers, 3)
    assert len({r.offer_id for r in ranked}) == 3
    firsts = Counter(policy.rank(make_context(), offers, 1)[0].offer_id for _ in range(4000))
    assert all(count / 4000 == pytest.approx(0.25, abs=0.03) for count in firsts.values())


def test_static_policy_ranks_by_frozen_popularity() -> None:
    offers = make_offers("a", "b", "c", "d")
    policy = StaticPolicy.fit([("a", 0.0), ("a", 1.0), ("b", 0.0), ("c", 1.0), ("c", 1.0)])
    assert policy.popularity == {"a": 0.5, "b": 0.0, "c": 1.0}
    assert [r.offer_id for r in policy.rank(make_context(), offers, 4)] == ["c", "a", "b", "d"]
    policy.update(make_context(), "d", 1.0)  # frozen: responses change nothing
    assert [r.offer_id for r in policy.rank(make_context(), offers, 2)] == ["c", "a"]


def test_static_policy_without_history_uses_catalogue_order() -> None:
    offers = make_offers("x", "y", "z")
    assert [r.offer_id for r in StaticPolicy().rank(make_context(), offers, 2)] == ["x", "y"]


# --- epsilon-greedy -------------------------------------------------------------------


def seeded_store(stats: dict[str, ArmStats], policy="eg", key="all") -> InMemoryOptionsStore:
    store = InMemoryOptionsStore()
    store.put(policy, key, stats)
    return store


def test_epsilon_zero_is_greedy() -> None:
    offers = make_offers("a", "b", "c")
    store = seeded_store({"a": ArmStats(10, 5), "b": ArmStats(10, 2), "c": ArmStats(10, 8)})
    policy = EpsilonGreedyPolicy("eg", store, rng=rng(), epsilon=0.0)
    for _ in range(50):
        assert [r.offer_id for r in policy.rank(make_context(), offers, 3)] == ["c", "a", "b"]


def test_epsilon_greedy_tries_untried_offers_first() -> None:
    offers = make_offers("a", "b", "c")
    store = seeded_store({"a": ArmStats(10, 9)})
    policy = EpsilonGreedyPolicy("eg", store, rng=rng(), epsilon=0.0)
    first_two = {r.offer_id for r in policy.rank(make_context(), offers, 2)}
    assert first_two == {"b", "c"}


def test_epsilon_one_explores_uniformly() -> None:
    offers = make_offers("a", "b", "c", "d")
    store = seeded_store({o.offer_id: ArmStats(10, i) for i, o in enumerate(offers)})
    policy = EpsilonGreedyPolicy("eg", store, rng=rng(), epsilon=1.0)
    firsts = Counter(policy.rank(make_context(), offers, 1)[0].offer_id for _ in range(4000))
    assert all(count / 4000 == pytest.approx(0.25, abs=0.03) for count in firsts.values())


def test_epsilon_decays_with_evidence_down_to_the_floor() -> None:
    policy = EpsilonGreedyPolicy(
        "eg", InMemoryOptionsStore(), rng=rng(), epsilon=0.2, epsilon_decay=0.99, min_epsilon=0.05
    )
    assert policy.current_epsilon({"a": ArmStats()}) == 0.2
    assert policy.current_epsilon({"a": ArmStats(pulls=50)}) == pytest.approx(0.2 * 0.99**50)
    assert policy.current_epsilon({"a": ArmStats(pulls=10_000)}) == 0.05


def test_epsilon_greedy_learns_the_best_arm() -> None:
    offers = make_offers("a", "b", "c")
    rates = {"a": 0.05, "b": 0.25, "c": 0.10}
    policy = EpsilonGreedyPolicy("eg", InMemoryOptionsStore(), rng=rng(), epsilon=0.1)
    shown = play_bandit(policy, offers, rates, rounds=3000)
    assert Counter(shown[-1000:])["b"] / 1000 > 0.85


# --- Thompson sampling ----------------------------------------------------------------


def test_thompson_converges_to_the_best_arm() -> None:
    offers = make_offers("a", "b", "c", "d")
    rates = {"a": 0.05, "b": 0.10, "c": 0.20, "d": 0.12}
    store = InMemoryOptionsStore()
    policy = ThompsonSamplingPolicy("ts", store, rng=rng())
    shown = play_bandit(policy, offers, rates, rounds=3000)
    assert Counter(shown[-1000:])["c"] / 1000 > 0.85
    best = store.get_context("ts", "all")["c"]
    assert best.mean == pytest.approx(0.20, abs=0.03)


def test_contextual_policy_learns_separately_per_segment() -> None:
    offers = make_offers("a", "b")
    student, family = make_context("student"), make_context("family")
    policy = ThompsonSamplingPolicy("ts", InMemoryOptionsStore(), rng=rng(), context="segment")
    play_bandit(policy, offers, {"a": 0.4, "b": 0.05}, rounds=600, context=student)
    play_bandit(policy, offers, {"a": 0.05, "b": 0.4}, rounds=600, context=family)
    student_means = {o: s.mean for o, s in policy.store.get_context("ts", "student").items()}
    family_means = {o: s.mean for o, s in policy.store.get_context("ts", "family").items()}
    assert max(student_means, key=student_means.get) == "a"
    assert max(family_means, key=family_means.get) == "b"


# --- forgetting -----------------------------------------------------------------------


def test_decay_shrinks_every_offer_in_the_context() -> None:
    store = seeded_store({"a": ArmStats(100, 20), "b": ArmStats(50, 5)}, policy="ts")
    policy = ThompsonSamplingPolicy("ts", store, rng=rng(), decay=0.9)
    policy.update(make_context(), "a", 1.0)
    arms = store.get_context("ts", "all")
    assert arms["a"] == ArmStats(100 * 0.9 + 1, 20 * 0.9 + 1)
    assert arms["b"] == ArmStats(50 * 0.9, 5 * 0.9)


def test_forgetting_adapts_when_the_best_offer_changes() -> None:
    """The core claim behind 'sophisticated forgetfulness', on a two-armed bandit."""
    offers = make_offers("a", "b")
    before, after = {"a": 0.30, "b": 0.10}, {"a": 0.10, "b": 0.30}
    share_of_new_best = {}
    for decay in (1.0, 0.99):
        policy = ThompsonSamplingPolicy("ts", InMemoryOptionsStore(), rng=rng(), decay=decay)
        play_bandit(policy, offers, before, rounds=2000, seed=1)
        shown = play_bandit(policy, offers, after, rounds=1500, seed=2)
        share_of_new_best[decay] = Counter(shown[-1000:])["b"] / 1000
    assert share_of_new_best[0.99] > 0.8
    assert share_of_new_best[1.0] < share_of_new_best[0.99] - 0.3


# --- validation and persistence -------------------------------------------------------


def test_invalid_arguments_are_rejected() -> None:
    store = InMemoryOptionsStore()
    with pytest.raises(ValueError, match="decay"):
        ThompsonSamplingPolicy("ts", store, rng=rng(), decay=0.0)
    with pytest.raises(ValueError, match="context mode"):
        ThompsonSamplingPolicy("ts", store, rng=rng(), context="province")
    policy = ThompsonSamplingPolicy("ts", store, rng=rng())
    with pytest.raises(ValueError, match="reward"):
        policy.update(make_context(), "a", 1.5)
    with pytest.raises(ValueError, match="at least 1"):
        policy.rank(make_context(), make_offers("a"), 0)


def test_learning_survives_a_restart_with_the_sql_store(tmp_path) -> None:
    url = f"sqlite:///{(tmp_path / 'model.db').as_posix()}"
    first = ThompsonSamplingPolicy("ts", create_store(url), rng=rng(), context="segment")
    first.update(make_context("family"), "a", 1.0)
    restarted = ThompsonSamplingPolicy("ts", create_store(url), rng=rng(), context="segment")
    stats = restarted.stats(make_context("family"), make_offers("a", "b"))
    assert stats == {"a": ArmStats(1, 1), "b": ArmStats()}


# --- end to end in the simulated world ------------------------------------------------


def test_contextual_thompson_learns_each_segment_favourite_in_the_simulated_world() -> None:
    """Rank, let the simulated customer respond, learn; then check what was learned."""
    settings, simulation = load_settings(), load_simulation_settings()
    offers = load_offers()
    truth = GroundTruth.from_yaml(simulation.ground_truth, offers)
    frame = generate_customers(5000, simulation.population.segment_mix, seed=settings.seed)
    customers = [Customer(**record) for record in frame.to_dict("records")]
    days = simulation.calendar.days()
    world = np.random.default_rng(7)

    policy = ThompsonSamplingPolicy("ts", InMemoryOptionsStore(), rng=rng(), context="segment")
    for _ in range(12_000):
        customer = customers[world.integers(len(customers))]
        context = sample_context(customer, world, days[world.integers(len(days))])
        eligible = [o for o in offers if o.is_eligible(context.segment)]
        shown = [offers_by_id(offers)[r.offer_id] for r in policy.rank(context, eligible, 3)]
        for offer_id, outcome in truth.respond(context, shown, world).items():
            reward = binary_reward(offers_by_id(offers)[offer_id], outcome)
            if reward is not None:
                policy.update(context, offer_id, reward)

    favourites = take_up_table(truth, customers, offers, days, samples_per_segment=2000).idxmax(
        axis=1
    )
    for segment, favourite in favourites.items():
        learned = {o: s.mean for o, s in policy.store.get_context("ts", segment).items()}
        assert max(learned, key=learned.get) == favourite, segment


def offers_by_id(offers: list[Offer]) -> dict[str, Offer]:
    return {offer.offer_id: offer for offer in offers}


def test_every_bandit_policy_shares_the_update_rule() -> None:
    assert issubclass(ThompsonSamplingPolicy, BanditPolicy)
    assert issubclass(EpsilonGreedyPolicy, BanditPolicy)
