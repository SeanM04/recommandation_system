"""Thompson sampling on customer features."""

import dataclasses

import numpy as np
import pytest

from recommender.algorithms.linear_thompson import FEATURES, LinearThompsonPolicy, features
from recommender.config import load_offers
from recommender.domain import Context

STUDENT = Context(
    "C00001", "student", "low", "prepaid", 150.0, "Limpopo", "app", "night", "saturday", True
)
FAMILY = dataclasses.replace(
    STUDENT,
    segment="family",
    device_tier="high",
    prepaid_vs_contract="contract",
    avg_monthly_spend=700.0,
    province="Gauteng",
    time_of_day="morning",
    day_of_week="tuesday",
    is_payday_week=False,
)
OFFERS = load_offers()[:2]
A, B = OFFERS[0].offer_id, OFFERS[1].offer_id


def test_features_describe_the_customer_and_the_moment() -> None:
    x = dict(zip(FEATURES, features(STUDENT), strict=True))
    on = {name for name, value in x.items() if value}
    assert on == {
        "bias",
        "segment=student",
        "device=low",
        "spend=low",
        "time=night",
        "weekend",
        "payday_week",
        "border_province",
    }
    family = {name for name, value in zip(FEATURES, features(FAMILY), strict=True) if value}
    assert {"segment=family", "device=high", "contract", "spend=high"} <= family


def test_it_learns_a_different_favourite_per_segment() -> None:
    """Students take up A, families take up B: the model should rank accordingly."""
    policy = LinearThompsonPolicy("features", rng=np.random.default_rng(0))
    for _ in range(200):
        for context, good, poor in ((STUDENT, A, B), (FAMILY, B, A)):
            policy.update(context, good, 1.0)
            policy.update(context, poor, 0.0)
    firsts = {
        context.segment: [policy.rank(context, OFFERS, 1)[0].offer_id for _ in range(50)]
        for context in (STUDENT, FAMILY)
    }
    assert firsts["student"].count(A) >= 48
    assert firsts["family"].count(B) >= 48
    assert policy.weights(A)["segment=student"] > policy.weights(A)["segment=family"]


def test_forgetting_shrinks_old_evidence() -> None:
    policy = LinearThompsonPolicy("features", rng=np.random.default_rng(0), decay=0.5)
    policy.update(STUDENT, A, 1.0)
    before = policy.weights(A)["bias"]
    policy.update(FAMILY, B, 0.0)  # every offer's evidence halves first
    assert 0 < policy.weights(A)["bias"] < before


def test_it_runs_in_the_simulator_only() -> None:
    policy = LinearThompsonPolicy("features", rng=np.random.default_rng(0))
    assert policy.learns and not policy.servable
    with pytest.raises(ValueError, match="reward"):
        policy.update(STUDENT, A, 2.0)
