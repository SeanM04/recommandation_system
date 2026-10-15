"""Options store in a SQL database via SQLAlchemy Core.

SQLite locally; any database SQLAlchemy supports (Postgres in production) by changing
the URL. Only portable SQL is used: SELECT, DELETE and INSERT inside one transaction.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
)

from recommender.store.options_store import (
    ArmStats,
    InMemoryOptionsStore,
    OptionRow,
    OptionsStore,
)

metadata = MetaData()

options = Table(
    "options",
    metadata,
    Column("policy", String(64), primary_key=True),
    Column("context_key", String(64), primary_key=True),
    Column("offer_id", String(64), primary_key=True),
    Column("alpha", Float, nullable=False),  # written for readers; derived from the two below
    Column("beta", Float, nullable=False),
    Column("pulls", Float, nullable=False),  # effective count once forgetting is on
    Column("total_reward", Float, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class SqlOptionsStore(OptionsStore):
    """Options store backed by the ``options`` table."""

    def __init__(self, engine: Engine | str) -> None:
        self.engine = create_engine(engine) if isinstance(engine, str) else engine
        metadata.create_all(self.engine)

    def get_context(self, policy: str, context_key: str) -> dict[str, ArmStats]:
        query = select(options.c.offer_id, options.c.pulls, options.c.total_reward).where(
            options.c.policy == policy, options.c.context_key == context_key
        )
        with self.engine.connect() as connection:
            return {
                row.offer_id: ArmStats(row.pulls, row.total_reward)
                for row in connection.execute(query)
            }

    def put(self, policy: str, context_key: str, arms: Mapping[str, ArmStats]) -> None:
        if not arms:
            return
        now = datetime.now(UTC)
        rows = [
            {
                "policy": policy,
                "context_key": context_key,
                "offer_id": offer_id,
                "alpha": stats.alpha,
                "beta": stats.beta,
                "pulls": stats.pulls,
                "total_reward": stats.total_reward,
                "updated_at": now,
            }
            for offer_id, stats in arms.items()
        ]
        with self.engine.begin() as connection:  # one transaction: replace these rows
            connection.execute(
                delete(options).where(
                    options.c.policy == policy,
                    options.c.context_key == context_key,
                    options.c.offer_id.in_(list(arms)),
                )
            )
            connection.execute(insert(options), rows)

    def rows(self, policy: str | None = None) -> list[OptionRow]:
        query = select(options).order_by(
            options.c.policy, options.c.context_key, options.c.offer_id
        )
        if policy is not None:
            query = query.where(options.c.policy == policy)
        with self.engine.connect() as connection:
            return [
                OptionRow(
                    row.policy,
                    row.context_key,
                    row.offer_id,
                    ArmStats(row.pulls, row.total_reward),
                    row.updated_at,
                )
                for row in connection.execute(query)
            ]

    def reset(self, policy: str | None = None) -> None:
        statement = delete(options)
        if policy is not None:
            statement = statement.where(options.c.policy == policy)
        with self.engine.begin() as connection:
            connection.execute(statement)


def create_store(url: str) -> OptionsStore:
    """``"memory"`` for an in-memory store, otherwise a SQLAlchemy database URL."""
    if url == "memory":
        return InMemoryOptionsStore()
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return SqlOptionsStore(url)
