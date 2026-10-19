"""Create the database tables and migrate a local SQLite model if there is one.

Usage:  python scripts/init_db.py      (make db)

Targets the database configured in .env (Postgres: Telco_Dynamic_Recomender), or the local
SQLite file when none is. Safe to run repeatedly.
"""

from __future__ import annotations

from sqlalchemy.engine import make_url

from recommender.config import load_settings
from recommender.customers import CustomerRepository
from recommender.db import create_tables, make_engine
from recommender.store.sql import SqlOptionsStore, copy_options, create_store


def main() -> None:
    """Create the tables and copy a local SQLite model into Postgres if there is one."""
    settings = load_settings()
    url = settings.database_url
    parsed = make_url(url)
    shown = parsed.database if url.startswith("sqlite") else parsed.render_as_string(True)
    print(f"Database:  {parsed.get_backend_name()} {shown}")

    engine = make_engine(url)
    print(f"Tables:    {', '.join(create_tables(engine))}")
    store = SqlOptionsStore(engine)

    sqlite_file = settings.paths.database
    if not url.startswith("sqlite") and sqlite_file.exists():
        copied = copy_options(create_store(f"sqlite:///{sqlite_file.as_posix()}"), store)
        print(f"Migrated {copied} options rows from {sqlite_file}")
    print(f"Customers: {CustomerRepository(engine).count():,}  (run `make data` to load them)")
    print(f"Options:   {len(store.rows()):,} rows")


if __name__ == "__main__":
    main()
