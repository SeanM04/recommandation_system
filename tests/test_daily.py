from datetime import UTC, date, datetime

import pytest
from sqlalchemy import func, select, update

from conftest import postgres_schema
from recommender.algorithms.base import ScoredOffer
from recommender.db import make_engine
from recommender.domain import Context
from recommender.events.daily import aggregate_day, daily_offer_stats, day_bounds, logged_days
from recommender.events.log import EventLog, invocations

TIMEZONE = "Africa/Johannesburg"  # UTC+2 all year
MARGINS = {"a": 10.0, "b": 4.0}
CONTEXT = Context(
    "C00001", "family", "mid", "prepaid", 300.0, "Gauteng", "app", "evening", "tuesday", False
)


@pytest.fixture(params=["sqlite", "postgres"])
def log(request, tmp_path):
    """An event log on a throwaway SQLite file, and on Postgres when configured."""
    if request.param == "sqlite":
        engine = make_engine(f"sqlite:///{(tmp_path / 'daily.db').as_posix()}")
        yield EventLog(engine)
        engine.dispose()
    else:
        with postgres_schema("pytest_daily") as url:
            engine = make_engine(url)
            yield EventLog(engine)
            engine.dispose()


def app_open(log: EventLog, at: datetime, outcomes: dict[str, str | None]) -> str:
    """Log an app open at ``at`` (UTC) showing ``outcomes``' offers, then their responses."""
    ranked = [ScoredOffer(offer_id, 1.0 / rank) for rank, offer_id in enumerate(outcomes, 1)]
    invocation_id = log.record_invocation(
        customer_id="C00001", policy="ts", context=CONTEXT, ranked=ranked
    )
    with log.engine.begin() as connection:
        connection.execute(
            update(invocations)
            .where(invocations.c.invocation_id == invocation_id)
            .values(created_at=at)
        )
    for offer_id, outcome in outcomes.items():
        if outcome:
            log.record_response(
                invocation_id=invocation_id, offer_id=offer_id, outcome=outcome, reward=None
            )
    return invocation_id


def test_day_bounds_follow_the_local_calendar() -> None:
    start, end = day_bounds(date(2026, 10, 14), TIMEZONE)
    assert start == datetime(2026, 10, 13, 22, tzinfo=UTC)
    assert end == datetime(2026, 10, 14, 22, tzinfo=UTC)


def test_one_row_per_offer_with_every_outcome_counted(log: EventLog) -> None:
    noon = datetime(2026, 10, 14, 10, tzinfo=UTC)
    app_open(log, noon, {"a": "taken_up", "b": "viewed"})
    app_open(log, noon, {"b": "taken_up", "a": "ignored"})
    app_open(log, noon, {"a": "taken_up", "b": None})  # no response for b yet
    rows = {
        row["offer_id"]: row
        for row in aggregate_day(log.engine, date(2026, 10, 14), TIMEZONE, MARGINS)
    }
    assert rows["a"] | {"aggregated_at": None} == {
        "day": date(2026, 10, 14), "policy": "ts", "segment": "family", "offer_id": "a",
        "impressions": 3, "shown_first": 2, "taken_up": 2, "viewed": 0, "ignored": 1,
        "no_response": 0, "revenue_zar": 20.0, "aggregated_at": None,
    }  # fmt: skip
    assert (rows["b"]["impressions"], rows["b"]["taken_up"], rows["b"]["no_response"]) == (3, 1, 1)
    assert rows["b"]["revenue_zar"] == 4.0
    assert sum(row["shown_first"] for row in rows.values()) == 3  # the day's app opens


def test_app_opens_count_on_their_local_day(log: EventLog) -> None:
    app_open(log, datetime(2026, 10, 14, 21, 30, tzinfo=UTC), {"a": "taken_up"})  # 23:30 local
    app_open(log, datetime(2026, 10, 14, 22, 30, tzinfo=UTC), {"a": "viewed"})  # 00:30 next day
    first = aggregate_day(log.engine, date(2026, 10, 14), TIMEZONE, MARGINS)
    second = aggregate_day(log.engine, date(2026, 10, 15), TIMEZONE, MARGINS)
    assert [(row["taken_up"], row["viewed"]) for row in first] == [(1, 0)]
    assert [(row["taken_up"], row["viewed"]) for row in second] == [(0, 1)]
    assert logged_days(log.engine, TIMEZONE) == [date(2026, 10, 14), date(2026, 10, 15)]


def test_rerunning_a_day_replaces_it_and_picks_up_late_responses(log: EventLog) -> None:
    day = date(2026, 10, 14)
    invocation_id = app_open(log, datetime(2026, 10, 14, 10, tzinfo=UTC), {"a": None})
    aggregate_day(log.engine, day, TIMEZONE, MARGINS)
    log.record_response(invocation_id=invocation_id, offer_id="a", outcome="taken_up", reward=1.0)
    aggregate_day(log.engine, day, TIMEZONE, MARGINS)
    with log.engine.connect() as connection:
        stored = connection.execute(select(daily_offer_stats)).mappings().all()
        assert connection.execute(select(func.count()).select_from(daily_offer_stats)).scalar() == 1
    assert (stored[0]["taken_up"], stored[0]["no_response"]) == (1, 0)


def test_an_empty_day_and_an_empty_log(log: EventLog) -> None:
    assert aggregate_day(log.engine, date(2026, 10, 14), TIMEZONE, MARGINS) == []
    assert logged_days(log.engine, TIMEZONE) == []
