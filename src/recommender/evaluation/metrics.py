"""Metrics over a simulation log (one row per app open, from ``simulator.engine``)."""

from __future__ import annotations

import math

import pandas as pd


def summarize(log: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for one policy run.

    Regret is expected take-ups lost against the oracle, summed over app opens. It uses
    the response model's expected values, so it measures choices rather than luck.
    """
    before, after = log[~log["shifted"]], log[log["shifted"]]
    regret = log["oracle_take_ups"] - log["expected_take_ups"]

    def mean(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].mean()) if len(frame) else math.nan

    return {
        "take_up_rate": mean(log, "take_ups"),
        "take_up_before_shift": mean(before, "take_ups"),
        "take_up_after_shift": mean(after, "take_ups"),
        "expected_take_up_rate": mean(log, "expected_take_ups"),
        "revenue_per_open": mean(log, "revenue"),
        "regret": float(regret.sum()),
        "regret_after_shift": float(regret[log["shifted"]].sum()),
        "share_of_oracle": float(log["expected_take_ups"].sum() / log["oracle_take_ups"].sum()),
        "revenue_regret": float((log["oracle_revenue"] - log["expected_revenue"]).sum()),
    }


def cumulative(log: pd.DataFrame, every: int = 1) -> pd.DataFrame:
    """Running totals per app open: take-ups, revenue and regret (for curves)."""
    curves = pd.DataFrame(
        {
            "round": log["round"] + 1,
            "take_ups": log["take_ups"].cumsum(),
            "revenue": log["revenue"].cumsum(),
            "regret": (log["oracle_take_ups"] - log["expected_take_ups"]).cumsum(),
        }
    )
    return curves.iloc[every - 1 :: every] if every > 1 else curves
