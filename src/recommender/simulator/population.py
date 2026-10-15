"""Synthetic customer population and app-open sessions.

The per-segment profiles below are illustrative assumptions, not real customer data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.domain import SEGMENTS, TIMES_OF_DAY, Context, Customer


@dataclass(frozen=True)
class SegmentProfile:
    """How attributes are distributed within one customer segment."""

    device_tier_mix: Mapping[str, float]
    contract_share: float  # share of customers on a contract rather than prepaid
    median_spend_zar: float  # median in-app spend per month
    time_of_day_mix: Mapping[str, float]  # when they usually open the app


SEGMENT_PROFILES: dict[str, SegmentProfile] = {
    "student": SegmentProfile(
        device_tier_mix={"low": 0.45, "mid": 0.45, "high": 0.10},
        contract_share=0.08,
        median_spend_zar=150,
        time_of_day_mix={"morning": 0.10, "afternoon": 0.20, "evening": 0.30, "night": 0.40},
    ),
    "young_professional": SegmentProfile(
        device_tier_mix={"low": 0.10, "mid": 0.50, "high": 0.40},
        contract_share=0.50,
        median_spend_zar=450,
        time_of_day_mix={"morning": 0.30, "afternoon": 0.15, "evening": 0.40, "night": 0.15},
    ),
    "family": SegmentProfile(
        device_tier_mix={"low": 0.35, "mid": 0.50, "high": 0.15},
        contract_share=0.30,
        median_spend_zar=350,
        time_of_day_mix={"morning": 0.25, "afternoon": 0.25, "evening": 0.40, "night": 0.10},
    ),
    "small_business": SegmentProfile(
        device_tier_mix={"low": 0.15, "mid": 0.50, "high": 0.35},
        contract_share=0.55,
        median_spend_zar=900,
        time_of_day_mix={"morning": 0.40, "afternoon": 0.35, "evening": 0.20, "night": 0.05},
    ),
    "pensioner": SegmentProfile(
        device_tier_mix={"low": 0.70, "mid": 0.25, "high": 0.05},
        contract_share=0.10,
        median_spend_zar=120,
        time_of_day_mix={"morning": 0.50, "afternoon": 0.35, "evening": 0.15, "night": 0.0},
    ),
}

# Approximate share of the South African population per province.
PROVINCE_MIX = {
    "Gauteng": 0.26,
    "KwaZulu-Natal": 0.20,
    "Western Cape": 0.12,
    "Eastern Cape": 0.12,
    "Limpopo": 0.10,
    "Mpumalanga": 0.08,
    "North West": 0.06,
    "Free State": 0.05,
    "Northern Cape": 0.01,
}

SPEND_SIGMA = 0.5  # log-normal spread of monthly spend around the segment median
HABIT_STRENGTH = 0.7  # chance an app open happens at the customer's usual time of day


def _draw(rng: np.random.Generator, mix: Mapping[str, float], size: int) -> np.ndarray:
    """Draw ``size`` categories with the probabilities in ``mix``."""
    return rng.choice(list(mix), size=size, p=list(mix.values()))


def generate_customers(n: int, segment_mix: Mapping[str, float], seed: int) -> pd.DataFrame:
    """Generate ``n`` synthetic app users. The same seed always gives the same population."""
    rng = np.random.default_rng(seed)
    segments = _draw(rng, {segment: segment_mix[segment] for segment in SEGMENTS}, n)

    device_tier = np.empty(n, dtype=object)
    prepaid_vs_contract = np.empty(n, dtype=object)
    avg_monthly_spend = np.empty(n, dtype=float)
    preferred_time_of_day = np.empty(n, dtype=object)

    for segment, profile in SEGMENT_PROFILES.items():
        rows = np.flatnonzero(segments == segment)
        k = len(rows)
        device_tier[rows] = _draw(rng, profile.device_tier_mix, k)
        on_contract = rng.random(k) < profile.contract_share
        prepaid_vs_contract[rows] = np.where(on_contract, "contract", "prepaid")
        spend = rng.lognormal(np.log(profile.median_spend_zar), SPEND_SIGMA, k)
        avg_monthly_spend[rows] = np.round(spend)
        preferred_time_of_day[rows] = _draw(rng, profile.time_of_day_mix, k)

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "segment": segments,
            "device_tier": device_tier,
            "prepaid_vs_contract": prepaid_vs_contract,
            "avg_monthly_spend": avg_monthly_spend,
            "province": _draw(rng, PROVINCE_MIX, n),
            "preferred_time_of_day": preferred_time_of_day,
        }
    )


def write_customers_csv(customers: pd.DataFrame, path: Path) -> None:
    """Write the population to CSV, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(path, index=False)


def sample_context(
    customer: Customer, rng: np.random.Generator, day: date, channel: str = "app"
) -> Context:
    """Simulate one app open: usually at the customer's habitual time of day."""
    if rng.random() < HABIT_STRENGTH:
        time_of_day = customer.preferred_time_of_day
    else:
        time_of_day = TIMES_OF_DAY[rng.integers(len(TIMES_OF_DAY))]
    return Context.build(customer, day=day, time_of_day=time_of_day, channel=channel)
