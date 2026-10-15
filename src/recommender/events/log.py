"""Event log: every invocation, every offer shown, every response.

Three normalised tables, so the log can be queried and aggregated with plain SQL:

    invocations   one row per app open: who, when, which policy, the full context
    impressions   one row per offer shown: its position and score
    responses     one row per reported outcome. Its primary key (invocation_id, offer_id)
                  makes POST /response idempotent, enforced by the database itself.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Table,
    func,
    select,
)

from recommender.algorithms.base import ScoredOffer
from recommender.db import metadata, upsert_insert
from recommender.domain import OUTCOMES, TAKEN_UP, Context

invocations = Table(
    "invocations",
    metadata,
    Column("invocation_id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("customer_id", String(16), nullable=False, index=True),
    Column("policy", String(64), nullable=False, index=True),
    Column("channel", String(8), nullable=False),
    Column("segment", String(32), nullable=False),
    Column("context", JSON, nullable=False),  # the full Context, to learn with later
)

impressions = Table(
    "impressions",
    metadata,
    Column(
        "invocation_id",
        String(36),
        ForeignKey("invocations.invocation_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("offer_id", String(64), primary_key=True),
    Column("position", Integer, nullable=False),  # 1 = top of the list
    Column("score", Float, nullable=False),
)

responses = Table(
    "responses",
    metadata,
    Column("invocation_id", String(36), primary_key=True),
    Column("offer_id", String(64), primary_key=True),
    Column("outcome", String(16), nullable=False),
    Column("reward", Float),  # null: no model update (an ignored offer, by default)
    Column("value", Float),  # optional value the app reported, e.g. rand spent
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    ForeignKeyConstraint(
        ["invocation_id", "offer_id"],
        ["impressions.invocation_id", "impressions.offer_id"],
        ondelete="CASCADE",
    ),
)

EVENT_TABLES = (invocations, impressions, responses)


class DuplicateResponse(Exception):
    """A response for this invocation and offer has already been recorded."""


@dataclass(frozen=True)
class LoggedInvocation:
    """An invocation as the response endpoint needs it."""

    invocation_id: str
    policy: str
    context: Context
    shown: dict[str, int]  # offer_id -> position


class EventLog:
    """Writes and reads the event tables."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        for table in EVENT_TABLES:
            table.create(engine, checkfirst=True)

    def record_invocation(
        self, *, customer_id: str, policy: str, context: Context, ranked: Sequence[ScoredOffer]
    ) -> str:
        """Log one app open and the offers shown, in one transaction. Returns its id."""
        invocation_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                invocations.insert().values(
                    invocation_id=invocation_id,
                    created_at=datetime.now(UTC),
                    customer_id=customer_id,
                    policy=policy,
                    channel=context.channel,
                    segment=context.segment,
                    context=asdict(context),
                )
            )
            connection.execute(
                impressions.insert(),
                [
                    {
                        "invocation_id": invocation_id,
                        "offer_id": offer.offer_id,
                        "position": position,
                        "score": offer.score,
                    }
                    for position, offer in enumerate(ranked, start=1)
                ],
            )
        return invocation_id

    def get_invocation(self, invocation_id: str) -> LoggedInvocation | None:
        """The invocation and the offers it showed, in one query."""
        query = (
            select(
                invocations.c.policy,
                invocations.c.context,
                impressions.c.offer_id,
                impressions.c.position,
            )
            .select_from(invocations.outerjoin(impressions))
            .where(invocations.c.invocation_id == invocation_id)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(query).all()
        if not rows:
            return None
        return LoggedInvocation(
            invocation_id=invocation_id,
            policy=rows[0].policy,
            context=Context(**rows[0].context),
            shown={row.offer_id: row.position for row in rows if row.offer_id is not None},
        )

    def record_response(
        self,
        *,
        invocation_id: str,
        offer_id: str,
        outcome: str,
        reward: float | None,
        value: float | None = None,
    ) -> None:
        """Log one outcome. Raises ``DuplicateResponse`` if this offer already has one."""
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        statement = (
            upsert_insert(self.engine, responses)
            .values(
                invocation_id=invocation_id,
                offer_id=offer_id,
                outcome=outcome,
                reward=reward,
                value=value,
                created_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["invocation_id", "offer_id"])
            .returning(responses.c.offer_id)
        )
        # RETURNING gives back a row only if this insert wrote one. (rowcount is not
        # reliable here: the Postgres driver reports -1 for ON CONFLICT DO NOTHING.)
        with self.engine.begin() as connection:
            if connection.execute(statement).first() is None:
                raise DuplicateResponse(
                    f"a response for offer {offer_id!r} in invocation {invocation_id!r} "
                    "was already recorded"
                )

    def average_rewards(self) -> dict[str, float]:
        """Average reward per offer over every response that updated a model: the static
        baseline's batch training, done by the database in one aggregate query."""
        query = (
            select(responses.c.offer_id, func.avg(responses.c.reward))
            .where(responses.c.reward.is_not(None))
            .group_by(responses.c.offer_id)
        )
        with self.engine.connect() as connection:
            return {offer_id: float(average) for offer_id, average in connection.execute(query)}

    def summary(self, margins: Mapping[str, float]) -> dict[str, dict]:
        """Per-policy counts, take-up rate and revenue, straight from the log."""
        by_invocation = responses.c.invocation_id == invocations.c.invocation_id
        with self.engine.connect() as connection:
            opens = connection.execute(
                select(invocations.c.policy, func.count()).group_by(invocations.c.policy)
            ).all()
            shown = connection.execute(
                select(invocations.c.policy, func.count())
                .select_from(impressions.join(invocations))
                .group_by(invocations.c.policy)
            ).all()
            outcomes = connection.execute(
                select(
                    invocations.c.policy,
                    responses.c.outcome,
                    responses.c.offer_id,
                    func.count(),
                )
                .select_from(responses.join(invocations, by_invocation))
                .group_by(invocations.c.policy, responses.c.outcome, responses.c.offer_id)
            ).all()

        summary: dict[str, dict] = defaultdict(
            lambda: (
                {"invocations": 0, "impressions": 0, "revenue_zar": 0.0}
                | {"responses": dict.fromkeys(OUTCOMES, 0)}
            )
        )
        for policy, count in opens:
            summary[policy]["invocations"] = count
        for policy, count in shown:
            summary[policy]["impressions"] = count
        for policy, outcome, offer_id, count in outcomes:
            summary[policy]["responses"][outcome] += count
            if outcome == TAKEN_UP:
                summary[policy]["revenue_zar"] += count * margins.get(offer_id, 0.0)
        for stats in summary.values():
            stats["take_up_rate"] = (
                stats["responses"][TAKEN_UP] / stats["invocations"] if stats["invocations"] else 0.0
            )
        return dict(sorted(summary.items()))
