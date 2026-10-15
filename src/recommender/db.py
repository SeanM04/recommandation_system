"""Shared database plumbing: one MetaData for every table, engines, and the upsert insert.

Postgres (``Telco_Dynamic_Recomender``) or SQLite, whichever ``Settings.database_url``
points at. Only portable SQL is used, plus ``INSERT ... ON CONFLICT``, which both support.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, Insert, MetaData, Table, create_engine
from sqlalchemy.dialects import postgresql, sqlite

metadata = MetaData()


def make_engine(url: str) -> Engine:
    """An engine for ``url``, creating the folder of a SQLite file if needed."""
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True)


def upsert_insert(engine: Engine, table: Table) -> Insert:
    """An INSERT for ``table`` that supports ``on_conflict_do_update`` / ``do_nothing``."""
    dialects = {"postgresql": postgresql.insert, "sqlite": sqlite.insert}
    name = engine.dialect.name
    if name not in dialects:
        raise NotImplementedError(f"no upsert for {name}; use Postgres or SQLite")
    return dialects[name](table)


def create_tables(engine: Engine) -> list[str]:
    """Create every table the project defines, if missing. Returns their names."""
    # Importing the modules registers their tables on ``metadata``.
    import recommender.customers  # noqa: F401
    import recommender.events.log  # noqa: F401
    import recommender.store.sql  # noqa: F401

    metadata.create_all(engine)
    return sorted(metadata.tables)
