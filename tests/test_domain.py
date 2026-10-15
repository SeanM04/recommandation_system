from datetime import date

import pytest
from pydantic import ValidationError

from recommender.config import load_offers, load_settings
from recommender.domain import (
    SEGMENTS,
    Context,
    Customer,
    Offer,
    is_payday_week,
    time_of_day_for_hour,
)
from recommender.simulator.settings import PopulationSettings, load_simulation_settings


def make_customer(**overrides) -> Customer:
    fields = {
        "customer_id": "C00001",
        "segment": "student",
        "device_tier": "mid",
        "prepaid_vs_contract": "prepaid",
        "avg_monthly_spend": 150.0,
        "province": "Gauteng",
        "preferred_time_of_day": "night",
    }
    return Customer(**(fields | overrides))


# --- configuration -------------------------------------------------------------------


def test_settings_load_and_resolve_paths() -> None:
    settings = load_settings()
    assert settings.seed == 42
    assert settings.paths.offers.is_absolute()
    assert settings.paths.offers.exists()


def test_offer_catalogue_is_valid() -> None:
    offers = load_offers()
    assert len(offers) == 12
    assert len({offer.offer_id for offer in offers}) == 12
    for segment in SEGMENTS:
        eligible = [offer for offer in offers if offer.is_eligible(segment)]
        assert len(eligible) >= 5, f"{segment} needs enough offers to learn from"


def test_offer_rejects_unknown_segment() -> None:
    with pytest.raises(ValidationError, match="unknown segments"):
        Offer(
            offer_id="x",
            name="X",
            category="data",
            price_zar=10,
            margin_zar=5,
            eligible_segments=("aliens",),
        )


def test_offer_rejects_margin_above_price() -> None:
    with pytest.raises(ValidationError, match="margin_zar cannot exceed"):
        Offer(
            offer_id="x",
            name="X",
            category="data",
            price_zar=10,
            margin_zar=50,
            eligible_segments=("student",),
        )


def test_simulation_settings_are_separate_from_recommender_settings() -> None:
    simulation = load_simulation_settings()
    assert simulation.population.n_customers == 5000
    assert simulation.ground_truth.is_absolute()
    days = simulation.calendar.days()
    assert (days[0], len(days)) == (date(2026, 9, 1), 60)
    assert "ground_truth" not in load_settings().model_dump_json()


def test_segment_mix_must_sum_to_one() -> None:
    mix = dict.fromkeys(SEGMENTS, 0.1)
    with pytest.raises(ValidationError, match="must sum to 1"):
        PopulationSettings(n_customers=10, segment_mix=mix)


# --- context helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        (5, "morning"),
        (11, "morning"),
        (12, "afternoon"),
        (17, "evening"),
        (22, "night"),
        (0, "night"),
    ],
)
def test_time_of_day_for_hour(hour: int, expected: str) -> None:
    assert time_of_day_for_hour(hour) == expected


def test_payday_week_salaried_vs_pensioner() -> None:
    assert is_payday_week(date(2026, 9, 25), "young_professional")
    assert not is_payday_week(date(2026, 9, 24), "young_professional")
    assert is_payday_week(date(2026, 9, 3), "pensioner")
    assert not is_payday_week(date(2026, 9, 25), "pensioner")


def test_context_build_derives_calendar_fields() -> None:
    saturday_payday = date(2026, 9, 26)
    context = Context.build(make_customer(), day=saturday_payday, time_of_day="night")
    assert context.day_of_week == "saturday"
    assert context.is_weekend
    assert context.is_payday_week
    assert context.spend_band == "low"
    assert context.channel == "app"
