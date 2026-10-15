"""Generate the synthetic customer population and write it to ``data/customers.csv``.

Usage:  python scripts/generate_customers.py [--n 5000] [--seed 42] [--out data/customers.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from recommender.config import load_settings
from recommender.simulator.population import generate_customers, write_customers_csv


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=settings.population.n_customers)
    parser.add_argument("--seed", type=int, default=settings.seed)
    parser.add_argument("--out", type=Path, default=settings.paths.customers)
    args = parser.parse_args()

    customers = generate_customers(args.n, settings.population.segment_mix, args.seed)
    write_customers_csv(customers, args.out)

    print(f"Wrote {len(customers):,} customers to {args.out}")
    summary = customers.groupby("segment").agg(
        customers=("customer_id", "count"),
        median_spend_zar=("avg_monthly_spend", "median"),
        contract_share=("prepaid_vs_contract", lambda s: round((s == "contract").mean(), 2)),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
