import pytest

from recommender.store.options_store import ArmStats, InMemoryOptionsStore, OptionsStore
from recommender.store.sql import SqlOptionsStore, create_store


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path) -> OptionsStore:
    if request.param == "memory":
        return InMemoryOptionsStore()
    return create_store(f"sqlite:///{(tmp_path / 'nested' / 'test.db').as_posix()}")


# --- arm statistics -------------------------------------------------------------------


def test_new_arm_is_the_uniform_prior() -> None:
    arm = ArmStats()
    assert (arm.alpha, arm.beta, arm.mean) == (1.0, 1.0, 0.5)


def test_update_adds_reward_to_alpha_and_the_rest_to_beta() -> None:
    arm = ArmStats().updated(1.0).updated(0.0).updated(0.25)
    assert arm.pulls == 3
    assert arm.alpha == pytest.approx(1 + 1.25)
    assert arm.beta == pytest.approx(1 + 1.75)


def test_decay_shrinks_evidence_towards_the_prior() -> None:
    arm = ArmStats(pulls=100, total_reward=20).decayed(0.5)
    assert (arm.pulls, arm.total_reward) == (50, 10)
    assert arm.mean == pytest.approx(11 / 52)  # same rate, less certainty
    assert ArmStats().decayed(0.5) == ArmStats()  # the prior itself is never forgotten


# --- store behaviour (both implementations) -------------------------------------------


def test_unknown_offers_get_the_prior(store: OptionsStore) -> None:
    assert store.get("thompson", "family", ["a", "b"]) == {"a": ArmStats(), "b": ArmStats()}


def test_put_then_get_round_trip(store: OptionsStore) -> None:
    store.put("thompson", "family", {"a": ArmStats(10, 3), "b": ArmStats(4, 1)})
    store.put("thompson", "family", {"a": ArmStats(11, 4)})  # replace one row
    assert store.get("thompson", "family", ["a", "b", "c"]) == {
        "a": ArmStats(11, 4),
        "b": ArmStats(4, 1),
        "c": ArmStats(),
    }


def test_policies_and_context_keys_are_isolated(store: OptionsStore) -> None:
    store.put("thompson", "family", {"a": ArmStats(10, 3)})
    assert store.get_context("thompson", "student") == {}
    assert store.get_context("epsilon_greedy", "family") == {}


def test_rows_and_reset(store: OptionsStore) -> None:
    store.put("thompson", "family", {"b": ArmStats(1, 1), "a": ArmStats(2, 0)})
    store.put("random", "all", {"a": ArmStats(1, 0)})
    rows = store.rows("thompson")
    assert [(r.context_key, r.offer_id) for r in rows] == [("family", "a"), ("family", "b")]
    assert rows[0].stats.beta == 3.0
    store.reset("thompson")
    assert store.rows("thompson") == []
    assert len(store.rows()) == 1
    store.reset()
    assert store.rows() == []


def test_sqlite_store_persists_across_connections(tmp_path) -> None:
    url = f"sqlite:///{(tmp_path / 'model.db').as_posix()}"
    create_store(url).put("thompson", "family", {"a": ArmStats(7, 2)})
    reopened = SqlOptionsStore(url)
    assert reopened.get_context("thompson", "family") == {"a": ArmStats(7, 2)}
    assert reopened.rows()[0].stats.alpha == 3.0
