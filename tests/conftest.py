"""Shared fixtures."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from recommender.config import load_settings


@contextmanager
def postgres_schema(schema: str) -> Iterator[str]:
    """A database URL scoped to a throwaway schema of the configured Postgres database.

    Tests never touch the real tables: the schema is created empty and dropped afterwards.
    Skips the test when no Postgres is configured or reachable.
    """
    url = load_settings().database_url
    if not url.startswith("postgresql"):
        pytest.skip("no Postgres configured: set the DB_* values in .env")
    admin = create_engine(url)
    try:
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            connection.execute(text(f"CREATE SCHEMA {schema}"))
    except OperationalError as error:
        admin.dispose()
        pytest.skip(f"Postgres not reachable: {error.orig}")
    try:
        scoped = make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
        yield scoped.render_as_string(hide_password=False)
    finally:
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        admin.dispose()
