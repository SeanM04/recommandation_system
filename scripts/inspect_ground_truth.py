"""Print the hidden take-up rates and each segment's favourite offer.

This is for humans reviewing the simulation. The recommender never reads it.

Usage:  python scripts/inspect_ground_truth.py
"""

from __future__ import annotations

import pandas as pd

from recommender.config import load_offers, load_settings
from recommender.customers import read_customers
from recommender.simulator.ground_truth import GroundTruth, take_up_table, winners
from recommender.simulator.settings import load_simulation_settings


def _heading(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _label(key: str | tuple) -> str:
    """'segment' or ('segment', is_payday_week) as a readable row label."""
    if not isinstance(key, tuple):
        return key
    segment, payday = key
    return f"{segment} / {'payday week' if payday else 'rest of month'}"


def _favourites(table: pd.DataFrame, value_format: str) -> str:
    best = winners(table)
    lines = [
        f"  {_label(key):<36} {row.best:<22} {format(row.value, value_format):>8}"
        f"   lead x{row.lead:.2f} over {row.runner_up}"
        for key, row in best.iterrows()
    ]
    return "\n".join(lines)


def main() -> None:
    settings = load_settings()
    simulation = load_simulation_settings()
    offers = load_offers()
    customers = read_customers(settings.paths.customers)
    days = simulation.calendar.days()
    truth = GroundTruth.from_yaml(simulation.ground_truth, offers)
    margins = pd.Series({offer.offer_id: offer.margin_zar for offer in offers})

    def table(world: GroundTruth, by: tuple[str, ...] = ("segment",)) -> pd.DataFrame:
        return take_up_table(world, customers, offers, days, by=by, seed=settings.seed)

    before = table(truth)
    _heading("True take-up rate, % per offer shown in first position (before shift)")
    print((before * 100).round(1).T.fillna("-").to_string())

    _heading("Favourite by take-up")
    print(_favourites(before, ".1%"))

    _heading("Favourite by revenue (margin x take-up, R per offer shown)")
    print(_favourites(before.mul(margins, axis=1), ".2f"))

    _heading("Favourite by take-up, per segment / payday week")
    print(_favourites(table(truth, ("segment", "is_payday_week")), ".1%"))

    _heading("Favourite by take-up after the behaviour shift")
    print(_favourites(table(truth.after_shift()), ".1%"))
    print(f"\n  Shift: {truth.shift_description}")

    _heading("Response model")
    attention = ", ".join(f"#{k} {a:.0%}" for k, a in enumerate(truth.response.attention, 1))
    print(f"  Chance of noticing each position: {attention}")
    print("  At most one take-up per app open; noticed but not taken = viewed.")


if __name__ == "__main__":
    main()
