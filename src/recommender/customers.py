"""Read side of the customer table: the profiles the business already holds."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd

from recommender.domain import CONTRACT_TYPES, DEVICE_TIERS, SEGMENTS, TIMES_OF_DAY, Customer

CUSTOMER_COLUMNS = [field.name for field in fields(Customer)]

_ALLOWED_VALUES = {
    "segment": SEGMENTS,
    "device_tier": DEVICE_TIERS,
    "prepaid_vs_contract": CONTRACT_TYPES,
    "preferred_time_of_day": TIMES_OF_DAY,
}


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
