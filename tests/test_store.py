import threading

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from recommender.config import load_settings
from recommender.store.options_store import ArmStats, InMemoryOptionsStore, OptionsStore
from recommender.store.sql import SqlOptionsStore, copy_options, create_store

PG_TEST_SCHEMA = "pytest_recommender"  # tests never touch the real tables


def _postgres_store():
    """A store in a throwaway schema of the configured Postgres database, if there is one."""
    url = load_settings().database_url
    if not url.startswith("postgresql"):
        pytest.skip("no Postgres configured: set RECOMMENDER_DATABASE_URL in .env")
    admin = create_engine(url)
    try:
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {PG_TEST_SCHEMA} CASCADE"))
            connection.execute(text(f"CREATE SCHEMA {PG_TEST_SCHEMA}"))
    except OperationalError as error:
        pytest.skip(f"Postgres not reachable: {error.orig}")
    engine = create_engine(url, connect_args={"options": f"-csearch_path={PG_TEST_SCHEMA}"})
    yield SqlOptionsStore(engine)
    engine.dispose()
    with admin.begin() as connection:
        connection.execute(text(f"DROP SCHEMA IF EXISTS {PG_TEST_SCHEMA} CASCADE"))
    admin.dispose()


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryOptionsStore()
    elif request.param == "sqlite":
        sqlite_store = create_store(f"sqlite:///{(tmp_path / 'nested' / 'test.db').as_posix()}")
        yield sqlite_store
        sqlite_store.engine.dispose()
    else:
        yield from _postgres_store()


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


# --- store behaviour (every implementation) -------------------------------------------


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


def test_learn_adds_one_reward(store: OptionsStore) -> None:
    store.learn("thompson", "family", "a", 1.0)
    store.learn("thompson", "family", "a", 0.0)
    store.learn("thompson", "family", "b", 0.5)
    assert store.get_context("thompson", "family") == {"a": ArmStats(2, 1), "b": ArmStats(1, 0.5)}


def test_learn_with_decay_forgets_every_offer_in_the_context_first(store: OptionsStore) -> None:
    store.put("thompson", "family", {"a": ArmStats(100, 20), "b": ArmStats(50, 5)})
    store.put("thompson", "student", {"a": ArmStats(10, 1)})
    store.learn("thompson", "family", "a", 1.0, decay=0.9)
    family = store.get_context("thompson", "family")
    assert family["a"].pulls == pytest.approx(100 * 0.9 + 1)
    assert family["a"].total_reward == pytest.approx(20 * 0.9 + 1)
    assert family["b"].pulls == pytest.approx(50 * 0.9)
    assert store.get_context("thompson", "student") == {"a": ArmStats(10, 1)}  # untouched


def test_concurrent_learning_loses_nothing(store: OptionsStore) -> None:
    """Four request threads updating the same offer at once: every update must count."""

    def worker() -> None:
        for _ in range(100):
            store.learn("thompson", "family", "a", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert store.get_context("thompson", "family")["a"] == ArmStats(400, 400)


def test_concurrent_learning_with_decay_matches_the_sequential_result(store: OptionsStore) -> None:
    gamma, updates = 0.99, 4 * 50

    def worker() -> None:
        for _ in range(50):
            store.learn("thompson", "family", "a", 1.0, decay=gamma)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    expected = (1 - gamma**updates) / (1 - gamma)  # 1 + γ + γ² + ... whatever the order
    assert store.get_context("thompson", "family")["a"].pulls == pytest.approx(expected)


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


# --- SQL specifics --------------------------------------------------------------------


def test_sqlite_store_persists_across_connections(tmp_path) -> None:
    url = f"sqlite:///{(tmp_path / 'model.db').as_posix()}"
    create_store(url).put("thompson", "family", {"a": ArmStats(7, 2)})
    reopened = SqlOptionsStore(url)
    assert reopened.get_context("thompson", "family") == {"a": ArmStats(7, 2)}
    assert reopened.rows()[0].stats.alpha == 3.0


def test_alpha_and_beta_are_computed_by_the_database(tmp_path) -> None:
    store = create_store(f"sqlite:///{(tmp_path / 'model.db').as_posix()}")
    store.learn("thompson", "family", "a", 1.0)
    store.learn("thompson", "family", "a", 0.0, decay=0.5)
    with store.engine.connect() as connection:
        row = connection.execute(text("SELECT pulls, total_reward, alpha, beta FROM options")).one()
    assert row.alpha == pytest.approx(1 + row.total_reward)
    assert row.beta == pytest.approx(1 + row.pulls - row.total_reward)


def test_copy_options_migrates_every_row_and_can_be_rerun(tmp_path) -> None:
    source = InMemoryOptionsStore()
    source.put("thompson", "family", {"a": ArmStats(10, 3), "b": ArmStats(4, 1)})
    source.put("static", "all", {"a": ArmStats(1, 1)})
    target = create_store(f"sqlite:///{(tmp_path / 'target.db').as_posix()}")
    assert copy_options(source, target) == 3
    assert copy_options(source, target) == 3  # idempotent: rows are replaced, not duplicated
    assert [(r.policy, r.context_key, r.offer_id, r.stats) for r in target.rows()] == [
        (r.policy, r.context_key, r.offer_id, r.stats) for r in source.rows()
    ]
