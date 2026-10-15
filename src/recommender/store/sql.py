"""Options store in a SQL database via SQLAlchemy Core.

Postgres (the ``Telco_Dynamic_Recomender`` database) or a local SQLite file, whichever
``Settings.database_url`` points at. See ``recommender.db`` for the shared plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import (
    Column,
    Computed,
    DateTime,
    Engine,
    Float,
    Insert,
    String,
    Table,
    delete,
    func,
    select,
    update,
)

from recommender.db import make_engine, metadata, upsert_insert
from recommender.store.options_store import (
    ArmStats,
    InMemoryOptionsStore,
    OptionRow,
    OptionsStore,
)

options = Table(
    "options",
    metadata,
    Column("policy", String(64), primary_key=True),
    Column("context_key", String(64), primary_key=True),
    Column("offer_id", String(64), primary_key=True),
    Column("pulls", Float, nullable=False),  # effective count once forgetting is on
    Column("total_reward", Float, nullable=False),
    # The Beta belief, computed by the database itself so it can never disagree with
    # the counts: a Beta(1, 1) prior plus the evidence.
    Column("alpha", Float, Computed("1.0 + total_reward", persisted=True)),
    Column("beta", Float, Computed("1.0 + pulls - total_reward", persisted=True)),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

KEY_COLUMNS = ["policy", "context_key", "offer_id"]


class SqlOptionsStore(OptionsStore):
    """Options store backed by the ``options`` table."""

    def __init__(self, engine: Engine | str) -> None:
        self.engine = make_engine(engine) if isinstance(engine, str) else engine
        options.create(self.engine, checkfirst=True)

    def _insert(self) -> Insert:
        return upsert_insert(self.engine, options)

    def get_context(self, policy: str, context_key: str) -> dict[str, ArmStats]:
        query = select(options.c.offer_id, options.c.pulls, options.c.total_reward).where(
            options.c.policy == policy, options.c.context_key == context_key
        )
        with self.engine.connect() as connection:
            return {
                row.offer_id: ArmStats(row.pulls, row.total_reward)
                for row in connection.execute(query)
            }

    def learn(
        self, policy: str, context_key: str, offer_id: str, reward: float, decay: float = 1.0
    ) -> None:
        now = datetime.now(UTC)
        forget = (
            update(options)
            .where(options.c.policy == policy, options.c.context_key == context_key)
            .values(
                pulls=options.c.pulls * decay,
                total_reward=options.c.total_reward * decay,
                updated_at=now,
            )
        )
        add = (
            self._insert()
            .values(
                policy=policy,
                context_key=context_key,
                offer_id=offer_id,
                pulls=1.0,
                total_reward=reward,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=KEY_COLUMNS,
                set_={
                    "pulls": options.c.pulls + 1.0,
                    "total_reward": options.c.total_reward + reward,
                    "updated_at": now,
                },
            )
        )
        # One transaction. Each statement computes from the row's current value inside
        # the database, so two requests running at once cannot overwrite each other.
        with self.engine.begin() as connection:
            if decay < 1:
                if connection.dialect.name == "postgresql":
                    # Forgetting locks every row in the context. Two requests locking them in
                    # different orders deadlock (seen with 4 workers), so updates to one
                    # context queue on a lock of their own first. SQLite locks the whole file.
                    connection.execute(
                        select(
                            func.pg_advisory_xact_lock(
                                func.hashtext(policy), func.hashtext(context_key)
                            )
                        )
                    )
                connection.execute(forget)
            connection.execute(add)

    def put(self, policy: str, context_key: str, arms: Mapping[str, ArmStats]) -> None:
        if not arms:
            return
        now = datetime.now(UTC)
        rows = [
            {
                "policy": policy,
                "context_key": context_key,
                "offer_id": offer_id,
                "pulls": stats.pulls,
                "total_reward": stats.total_reward,
                "updated_at": now,
            }
            for offer_id, stats in arms.items()
        ]
        statement = self._insert()
        statement = statement.on_conflict_do_update(
            index_elements=KEY_COLUMNS,
            set_={
                "pulls": statement.excluded.pulls,
                "total_reward": statement.excluded.total_reward,
                "updated_at": statement.excluded.updated_at,
            },
        )
        with self.engine.begin() as connection:
            connection.execute(statement, rows)

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
    return SqlOptionsStore(url)


def copy_options(source: OptionsStore, target: OptionsStore) -> int:
    """Copy every options row from one store to another (e.g. SQLite to Postgres).

    Rows that already exist in ``target`` are replaced, so running it twice is safe.
    Returns the number of rows copied.
    """
    rows = source.rows()
    grouped: dict[tuple[str, str], dict[str, ArmStats]] = {}
    for row in rows:
        grouped.setdefault((row.policy, row.context_key), {})[row.offer_id] = row.stats
    for (policy, context_key), arms in grouped.items():
        target.put(policy, context_key, arms)
    return len(rows)
