"""Nightly aggregation: the event log rolled up into one row per day, policy, segment and offer.

Reports and drift alerts read ``daily_offer_stats`` instead of scanning the raw log. A day
is a calendar day in the configured timezone, and every outcome counts on the day of the
app open it answers. Rebuilding a day replaces its rows in one transaction, so the job is
safe to rerun (and late responses are picked up by rerunning yesterday).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Table,
    and_,
    case,
    delete,
    func,
    insert,
    select,
)

from recommender.db import metadata
from recommender.domain import IGNORED, TAKEN_UP, VIEWED
from recommender.events.log import impressions, invocations, responses

daily_offer_stats = Table(
    "daily_offer_stats",
    metadata,
    Column("day", Date, primary_key=True),
    Column("policy", String(64), primary_key=True),
    Column("segment", String(32), primary_key=True),
    Column("offer_id", String(64), primary_key=True),
    Column("impressions", Integer, nullable=False),
    Column("shown_first", Integer, nullable=False),  # summed over offers: the app opens
    Column("taken_up", Integer, nullable=False),
    Column("viewed", Integer, nullable=False),
    Column("ignored", Integer, nullable=False),
    Column("no_response", Integer, nullable=False),  # shown, but no outcome reported (yet)
    Column("revenue_zar", Float, nullable=False),  # take-ups x the offer's margin
    Column("aggregated_at", DateTime(timezone=True), nullable=False),
)


def day_bounds(day: date, timezone: str) -> tuple[datetime, datetime]:
    """The UTC instants where ``day`` starts and ends in ``timezone``."""
    zone = ZoneInfo(timezone)
    start = datetime.combine(day, time.min, tzinfo=zone)
    return start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC)


def _count(outcome: str):
    return func.sum(case((responses.c.outcome == outcome, 1), else_=0))


def aggregate_day(
    engine: Engine, day: date, timezone: str, margins: Mapping[str, float]
) -> list[dict]:
    """Rebuild ``day`` in ``daily_offer_stats`` from the event log. Returns the rows."""
    daily_offer_stats.create(engine, checkfirst=True)
    start, end = day_bounds(day, timezone)
    query = (
        select(
            invocations.c.policy,
            invocations.c.segment,
            impressions.c.offer_id,
            func.count().label("impressions"),
            func.sum(case((impressions.c.position == 1, 1), else_=0)).label("shown_first"),
            _count(TAKEN_UP).label("taken_up"),
            _count(VIEWED).label("viewed"),
            _count(IGNORED).label("ignored"),
            func.sum(case((responses.c.outcome.is_(None), 1), else_=0)).label("no_response"),
        )
        .select_from(
            impressions.join(invocations).outerjoin(
                responses,
                and_(
                    responses.c.invocation_id == impressions.c.invocation_id,
                    responses.c.offer_id == impressions.c.offer_id,
                ),
            )
        )
        .where(invocations.c.created_at >= start, invocations.c.created_at < end)
        .group_by(invocations.c.policy, invocations.c.segment, impressions.c.offer_id)
    )
    now = datetime.now(UTC)
    with engine.begin() as connection:
        rows = [dict(row._mapping, day=day, aggregated_at=now) for row in connection.execute(query)]
        for row in rows:
            row["revenue_zar"] = row["taken_up"] * margins.get(row["offer_id"], 0.0)
        connection.execute(delete(daily_offer_stats).where(daily_offer_stats.c.day == day))
        if rows:
            connection.execute(insert(daily_offer_stats), rows)
    return rows


def logged_days(engine: Engine, timezone: str) -> list[date]:
    """Every local day from the first app open in the log to the last."""
    with engine.connect() as connection:
        first, last = connection.execute(
            select(func.min(invocations.c.created_at), func.max(invocations.c.created_at))
        ).one()
    if first is None:
        return []
    zone = ZoneInfo(timezone)
    first_day, last_day = (_as_utc(t).astimezone(zone).date() for t in (first, last))
    return [first_day + timedelta(days=n) for n in range((last_day - first_day).days + 1)]


def _as_utc(moment: datetime) -> datetime:
    """SQLite returns stored timestamps without their timezone; they were written in UTC."""
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
