import threading

import pytest

from conftest import postgres_schema
from recommender.algorithms.base import ScoredOffer
from recommender.db import create_tables, make_engine
from recommender.domain import Context
from recommender.events.log import DuplicateResponse, EventLog

CONTEXT = Context(
    "C00001", "family", "mid", "prepaid", 300.0, "Gauteng", "app", "evening", "tuesday", False
)


@pytest.fixture(params=["sqlite", "postgres"])
def log(request, tmp_path):
    """An event log on a throwaway SQLite file, and on Postgres when configured."""
    if request.param == "sqlite":
        engine = make_engine(f"sqlite:///{(tmp_path / 'events.db').as_posix()}")
        yield EventLog(engine)
        engine.dispose()
    else:
        with postgres_schema("pytest_events") as url:
            engine = make_engine(url)
            yield EventLog(engine)
            engine.dispose()


def shown(*offer_ids: str) -> list[ScoredOffer]:
    """Offers as a policy would rank them, scores falling with rank."""
    return [ScoredOffer(offer_id, 1.0 / rank) for rank, offer_id in enumerate(offer_ids, 1)]


def test_invocation_round_trip(log: EventLog) -> None:
    invocation_id = log.record_invocation(
        customer_id="C00001", policy="ts", context=CONTEXT, ranked=shown("a", "b")
    )
    logged = log.get_invocation(invocation_id)
    assert logged.policy == "ts"
    assert logged.context == CONTEXT
    assert logged.shown == {"a": 1, "b": 2}
    assert log.get_invocation("no-such-id") is None


def test_second_response_for_the_same_offer_is_a_duplicate(log: EventLog) -> None:
    invocation_id = log.record_invocation(
        customer_id="C00001", policy="ts", context=CONTEXT, ranked=shown("a", "b")
    )
    log.record_response(invocation_id=invocation_id, offer_id="a", outcome="viewed", reward=0.0)
    with pytest.raises(DuplicateResponse):
        log.record_response(
            invocation_id=invocation_id, offer_id="a", outcome="taken_up", reward=1.0
        )
    log.record_response(invocation_id=invocation_id, offer_id="b", outcome="ignored", reward=None)


def test_simultaneous_duplicates_are_recorded_exactly_once(log: EventLog) -> None:
    """Eight copies of one response arriving at the same moment: one wins, seven are refused."""
    invocation_id = log.record_invocation(
        customer_id="C00001", policy="ts", context=CONTEXT, ranked=shown("a")
    )
    results: list[str] = []
    start = threading.Barrier(8)

    def send() -> None:
        """Send the same response once all threads are ready, noting the result."""
        start.wait()
        try:
            log.record_response(
                invocation_id=invocation_id, offer_id="a", outcome="taken_up", reward=1.0
            )
            results.append("recorded")
        except DuplicateResponse:
            results.append("duplicate")

    threads = [threading.Thread(target=send) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["duplicate"] * 7 + ["recorded"]


def test_average_rewards_per_offer(log: EventLog) -> None:
    for outcome, reward in [("taken_up", 1.0), ("viewed", 0.0), ("ignored", None)]:
        invocation_id = log.record_invocation(
            customer_id="C00001", policy="ts", context=CONTEXT, ranked=shown("a", "b")
        )
        log.record_response(
            invocation_id=invocation_id, offer_id="a", outcome=outcome, reward=reward
        )
        log.record_response(invocation_id=invocation_id, offer_id="b", outcome="viewed", reward=0.0)
    assert log.average_rewards() == {"a": 0.5, "b": 0.0}  # ignored rows are left out


def test_table_creation_is_safe_when_workers_start_together() -> None:
    """Several API processes starting at once on an empty database must not collide."""
    with postgres_schema("pytest_startup") as url:
        engines = [make_engine(url) for _ in range(6)]
        errors: list[Exception] = []
        start = threading.Barrier(len(engines))

        def boot(engine) -> None:
            """Create the tables as a starting worker would, keeping any error."""
            start.wait()
            try:
                create_tables(engine)
            except Exception as error:  # noqa: BLE001 - any failure fails the test
                errors.append(error)

        threads = [threading.Thread(target=boot, args=(engine,)) for engine in engines]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for engine in engines:
            engine.dispose()
        assert errors == []
