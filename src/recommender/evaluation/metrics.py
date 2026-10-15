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


def convergence(
    log: pd.DataFrame,
    favourites: dict[tuple[str, bool], str],
    *,
    start_round: int = 0,
    end_round: int | None = None,
    window: int = 100,
    threshold: float = 0.8,
) -> dict[str, float]:
    """App opens until each segment's best offer is ranked first most of the time.

    ``favourites`` maps (segment, is_payday_week) to the right first offer for that moment,
    so a policy that can tell payday apart is judged against payday's answer. Counting
    starts at ``start_round`` (0, or the shift). The policy has converged for a segment
    once the right offer led in ``threshold`` of that segment's last ``window`` app
    opens; the result is the app opens elapsed until then (NaN if it never did).
    """
    phase = log[(log["round"] >= start_round) & (log["round"] < (end_round or len(log)))]
    result = {}
    for segment, opens in phase.groupby("segment", sort=True):
        right = [
            favourites[(segment, payday)] == top
            for payday, top in zip(opens["is_payday_week"], opens["top_offer"], strict=True)
        ]
        share = pd.Series(right, dtype=float).rolling(window).mean()
        reached = share[share >= threshold]
        result[segment] = (
            float(opens["round"].iloc[reached.index[0]] - start_round + 1)
            if len(reached)
            else math.nan
        )
    return result


def selection_share(log: pd.DataFrame, segment: str, every: int = 1000) -> pd.DataFrame:
    """Share of app opens in which each offer was ranked first, per block of ``every``
    app opens, for one segment (rows: blocks, columns: offers)."""
    opens = log[log["segment"] == segment]
    blocks = (opens["round"] // every) * every + every
    return pd.crosstab(blocks, opens["top_offer"], normalize="index").rename_axis("round")
