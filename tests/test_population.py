from datetime import date

import numpy as np
import pandas as pd
import pytest

from recommender.customers import read_customers
from recommender.domain import DEVICE_TIERS, PROVINCES, SEGMENTS, TIMES_OF_DAY
from recommender.simulator.population import (
    generate_customers,
    sample_context,
    write_customers_csv,
)
from recommender.simulator.settings import load_simulation_settings

SEGMENT_MIX = load_simulation_settings().population.segment_mix


@pytest.fixture(scope="module")
def customers() -> pd.DataFrame:
    """The generated population, once for the module."""
    return generate_customers(5000, SEGMENT_MIX, seed=42)


def test_same_seed_gives_same_population(customers: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(customers, generate_customers(5000, SEGMENT_MIX, seed=42))
    assert not customers.equals(generate_customers(5000, SEGMENT_MIX, seed=7))


def test_population_has_valid_attributes(customers: pd.DataFrame) -> None:
    assert len(customers) == 5000
    assert customers["customer_id"].is_unique
    assert set(customers["segment"]) == set(SEGMENTS)
    assert set(customers["device_tier"]) <= set(DEVICE_TIERS)
    assert set(customers["province"]) <= set(PROVINCES)
    assert set(customers["preferred_time_of_day"]) <= set(TIMES_OF_DAY)
    assert (customers["avg_monthly_spend"] > 0).all()


def test_segment_shares_follow_config(customers: pd.DataFrame) -> None:
    shares = customers["segment"].value_counts(normalize=True)
    for segment, expected in SEGMENT_MIX.items():
        assert shares[segment] == pytest.approx(expected, abs=0.02)


def test_segment_profiles_are_realistic(customers: pd.DataFrame) -> None:
    by_segment = customers.groupby("segment")
    contract_share = by_segment["prepaid_vs_contract"].apply(lambda s: (s == "contract").mean())
    median_spend = by_segment["avg_monthly_spend"].median()
    assert contract_share["student"] < 0.15  # students are overwhelmingly prepaid
    assert median_spend["small_business"] > median_spend["young_professional"]
    assert median_spend["pensioner"] < median_spend["family"]


def test_csv_round_trip(customers: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "nested" / "customers.csv"
    write_customers_csv(customers, path)
    loaded = read_customers(path)
    assert len(loaded) == 5000
    assert loaded[0].customer_id == "C00001"
    assert loaded[0].segment == customers.loc[0, "segment"]


def test_read_customers_rejects_bad_values(customers: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "customers.csv"
    write_customers_csv(customers.assign(segment="martian"), path)
    with pytest.raises(ValueError, match="invalid segment"):
        read_customers(path)


def test_sample_context_mostly_follows_habit(customers: pd.DataFrame, tmp_path) -> None:
    path = tmp_path / "customers.csv"
    write_customers_csv(customers, path)
    customer = read_customers(path)[0]
    rng = np.random.default_rng(0)
    contexts = [sample_context(customer, rng, date(2026, 9, 1)) for _ in range(2000)]
    habitual = np.mean([c.time_of_day == customer.preferred_time_of_day for c in contexts])
    # 70% habit + 1/4 of the remaining 30% that land on the same bucket by chance
    assert habitual == pytest.approx(0.7 + 0.3 / 4, abs=0.03)
    assert all(c.customer_id == customer.customer_id for c in contexts)
