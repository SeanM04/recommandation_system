"""Customer profiles: the CSV the generator writes, and the database table the API reads."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, Engine, Float, String, Table, func, select

from recommender.db import metadata, upsert_insert
from recommender.domain import CONTRACT_TYPES, DEVICE_TIERS, SEGMENTS, TIMES_OF_DAY, Customer

CUSTOMER_COLUMNS = [field.name for field in fields(Customer)]

_ALLOWED_VALUES = {
    "segment": SEGMENTS,
    "device_tier": DEVICE_TIERS,
    "prepaid_vs_contract": CONTRACT_TYPES,
    "preferred_time_of_day": TIMES_OF_DAY,
}

customers_table = Table(
    "customers",
    metadata,
    Column("customer_id", String(16), primary_key=True),
    Column("segment", String(32), nullable=False, index=True),
    Column("device_tier", String(8), nullable=False),
    Column("prepaid_vs_contract", String(16), nullable=False),
    Column("avg_monthly_spend", Float, nullable=False),
    Column("province", String(32), nullable=False),
    Column("preferred_time_of_day", String(16), nullable=False),
)


def read_customers(path: Path) -> list[Customer]:
    """Load and validate the customer CSV written by ``scripts/generate_customers.py``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` to generate customers.")
    frame = pd.read_csv(path, dtype={"customer_id": str})

    missing = set(CUSTOMER_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"customer file is missing columns: {sorted(missing)}")
    for column, allowed in _ALLOWED_VALUES.items():
        invalid = set(frame[column]) - set(allowed)
        if invalid:
            raise ValueError(f"invalid {column} values: {sorted(invalid)}")
    if frame["customer_id"].duplicated().any():
        raise ValueError("customer_id must be unique")

    records = frame[CUSTOMER_COLUMNS].to_dict("records")
    return [Customer(**record) for record in records]


class CustomerRepository:
    """Customer profiles in the ``customers`` table: one primary-key lookup per request."""

    def __init__(self, engine: Engine) -> None:
        """Customers in ``engine``'s database, creating the table if missing."""
        self.engine = engine
        customers_table.create(engine, checkfirst=True)

    def get(self, customer_id: str) -> Customer | None:
        """One customer by id, or None."""
        query = select(customers_table).where(customers_table.c.customer_id == customer_id)
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
        return Customer(**row) if row else None

    def count(self) -> int:
        """How many customers are loaded."""
        with self.engine.connect() as connection:
            return connection.execute(
                select(func.count()).select_from(customers_table)
            ).scalar_one()

    def list_customers(self, segment: str | None = None, limit: int = 20) -> list[tuple[str, str]]:
        """A few (customer_id, segment) pairs, optionally from one segment, in one query."""
        query = select(customers_table.c.customer_id, customers_table.c.segment).order_by(
            customers_table.c.customer_id
        )
        if segment is not None:
            query = query.where(customers_table.c.segment == segment)
        with self.engine.connect() as connection:
            return [
                (row.customer_id, row.segment) for row in connection.execute(query.limit(limit))
            ]

    def load(self, customers: Iterable[Customer]) -> int:
        """Insert or update profiles. Returns how many rows were written."""
        rows = [asdict(customer) for customer in customers]
        if not rows:
            return 0
        statement = upsert_insert(self.engine, customers_table)
        statement = statement.on_conflict_do_update(
            index_elements=["customer_id"],
            set_={name: statement.excluded[name] for name in CUSTOMER_COLUMNS[1:]},
        )
        with self.engine.begin() as connection:
            connection.execute(statement, rows)
        return len(rows)
