"""Create the database tables and migrate a local SQLite model if there is one.

Usage:  python scripts/init_db.py      (make db)

Targets RECOMMENDER_DATABASE_URL from .env (the Postgres database Telco_Dynamic_Recomender),
or the local SQLite file when it is not set. Safe to run repeatedly.
"""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from recommender.config import load_settings
from recommender.store.sql import copy_options, create_store


def main() -> None:
    settings = load_settings()
    url = settings.database_url
    parsed = make_url(url)
    shown = parsed.database if url.startswith("sqlite") else parsed.render_as_string(True)
    print(f"Database: {parsed.get_backend_name()} {shown}")

    store = create_store(url)  # creates any missing tables
    print(f"Tables:   {', '.join(sorted(inspect(store.engine).get_table_names()))}")

    sqlite_file = settings.paths.database
    if not url.startswith("sqlite") and sqlite_file.exists():
        local = create_store(f"sqlite:///{sqlite_file.as_posix()}")
        copied = copy_options(local, store)
        print(f"Migrated {copied} options rows from {sqlite_file}")
    print(f"Options rows in the database: {len(store.rows())}")


if __name__ == "__main__":
    main()
