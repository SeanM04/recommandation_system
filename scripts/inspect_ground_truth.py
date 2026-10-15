"""Print the hidden take-up table per segment, before and after the behaviour shift.

This is for humans reviewing the simulation. The recommender never reads it.

Usage:  python scripts/inspect_ground_truth.py
"""

from __future__ import annotations

import pandas as pd

from recommender.config import load_offers, load_settings
from recommender.customers import read_customers
from recommender.simulator.ground_truth import GroundTruth, segment_take_up_table


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}\n{'=' * len(title)}")
    print((table * 100).round(1).T.fillna("-").to_string())
    print("\nBest offer per segment:")
    for segment, row in table.iterrows():
        ranked = row.dropna().sort_values(ascending=False)
        gap = ranked.iloc[0] / ranked.iloc[1]
        print(
            f"  {segment:<20} {ranked.index[0]:<22} {ranked.iloc[0]:6.1%}"
            f"   (runner-up {ranked.index[1]} {ranked.iloc[1]:.1%}, x{gap:.2f})"
        )


def main() -> None:
    settings = load_settings()
    offers = load_offers()
    customers = read_customers(settings.paths.customers)
    truth = GroundTruth.from_yaml(settings.simulation.ground_truth, offers)

    before = segment_take_up_table(truth, customers, offers, seed=settings.seed)
    after = segment_take_up_table(truth.after_shift(), customers, offers, seed=settings.seed)
    _print_table("True take-up rate % (before shift)", before)
    _print_table("True take-up rate % (after shift)", after)
    print(f"\nShift: {truth.shift_description}")


if __name__ == "__main__":
    main()
