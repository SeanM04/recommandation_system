"""Shared database plumbing: one MetaData for every table, engines, and the upsert insert.

Postgres (``Telco_Dynamic_Recomender``) or SQLite, whichever ``Settings.database_url``
points at. Only portable SQL is used, plus ``INSERT ... ON CONFLICT``, which both support.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, Insert, MetaData, Table, create_engine, text
from sqlalchemy.dialects import postgresql, sqlite

metadata = MetaData()


SQLITE_BUSY_TIMEOUT_S = 30  # SQLite has one writer at a time; the others wait this long


def make_engine(url: str) -> Engine:
    """An engine for ``url``, creating the folder of a SQLite file if needed."""
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        # The driver's default wait is 5 seconds: too short when the API, the dashboard and
        # a simulation write at once on a busy machine ("database is locked").
        return create_engine(
            url, pool_pre_ping=True, connect_args={"timeout": SQLITE_BUSY_TIMEOUT_S}
        )
    return create_engine(url, pool_pre_ping=True)


def upsert_insert(engine: Engine, table: Table) -> Insert:
    """An INSERT for ``table`` that supports ``on_conflict_do_update`` / ``do_nothing``."""
    dialects = {"postgresql": postgresql.insert, "sqlite": sqlite.insert}
    name = engine.dialect.name
    if name not in dialects:
        raise NotImplementedError(f"no upsert for {name}; use Postgres or SQLite")
    return dialects[name](table)


SCHEMA_LOCK_KEY = 820_551  # any constant; identifies "creating this project's tables"


def create_tables(engine: Engine) -> list[str]:
    """Create every table the project defines, if missing. Returns their names.

    Safe when several API worker processes start at once on an empty database: on
    Postgres, a transaction-level advisory lock lets one process create the tables while
    the others wait, then find them already there.
    """
    # Importing the modules registers their tables on ``metadata``.
    import recommender.accounts  # noqa: F401
    import recommender.customers  # noqa: F401
    import recommender.events.daily  # noqa: F401
    import recommender.events.log  # noqa: F401
    import recommender.store.sql  # noqa: F401

    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": SCHEMA_LOCK_KEY})
        metadata.create_all(connection)
    return sorted(metadata.tables)
