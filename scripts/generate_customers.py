"""Generate the synthetic customer population: ``data/customers.csv`` and the database.

Usage:  python scripts/generate_customers.py [--n 5000] [--seed 42] [--out data/customers.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recommender.config import load_settings
from recommender.customers import CustomerRepository
from recommender.db import make_engine
from recommender.domain import Customer
from recommender.simulator.population import generate_customers, write_customers_csv
from recommender.simulator.settings import load_simulation_settings


def main() -> None:
    """Generate the customers, write the CSV, load the database and print a summary."""
    settings = load_settings()
    population = load_simulation_settings().population
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=population.n_customers)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--out", type=Path, default=settings.paths.customers)
    args = parser.parse_args()

    customers = generate_customers(args.n, population.segment_mix, args.seed)
    write_customers_csv(customers, args.out)

    print(f"Wrote {len(customers):,} customers to {args.out}")

    engine = make_engine(settings.database_url)
    loaded = CustomerRepository(engine).load(
        Customer(**record) for record in customers.to_dict("records")
    )
    print(f"Loaded {loaded:,} customers into the {engine.dialect.name} customers table")
    summary = customers.groupby("segment").agg(
        customers=("customer_id", "count"),
        median_spend_zar=("avg_monthly_spend", "median"),
        contract_share=("prepaid_vs_contract", lambda s: round((s == "contract").mean(), 2)),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
