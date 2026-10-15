import ast
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from recommender.config import PROJECT_ROOT, load_offers, load_settings
from recommender.domain import SEGMENTS, Context, Customer
from recommender.simulator.ground_truth import Effect, GroundTruth, segment_take_up_table
from recommender.simulator.population import generate_customers

SETTINGS = load_settings()
OFFERS = load_offers()
OFFERS_BY_ID = {offer.offer_id: offer for offer in OFFERS}


@pytest.fixture(scope="module")
def truth() -> GroundTruth:
    return GroundTruth.from_yaml(SETTINGS.simulation.ground_truth, OFFERS)


@pytest.fixture(scope="module")
def customers() -> list[Customer]:
    frame = generate_customers(5000, SETTINGS.population.segment_mix, seed=SETTINGS.seed)
    return [Customer(**record) for record in frame.to_dict("records")]


@pytest.fixture(scope="module")
def tables(truth, customers) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = segment_take_up_table(truth, customers, OFFERS, samples_per_segment=2000)
    after = segment_take_up_table(truth.after_shift(), customers, OFFERS, samples_per_segment=2000)
    return before, after


def make_context(segment: str = "family", **overrides) -> Context:
    fields = {
        "customer_id": "C00001",
        "segment": segment,
        "device_tier": "mid",
        "prepaid_vs_contract": "prepaid",
        "avg_monthly_spend": 400.0,
        "province": "Gauteng",
        "channel": "app",
        "time_of_day": "afternoon",
        "day_of_week": "tuesday",
        "is_payday_week": False,
    }
    return Context(**(fields | overrides))


def best_offer(table: pd.DataFrame, segment: str) -> str:
    return table.loc[segment].idxmax()


# --- probabilities -------------------------------------------------------------------


def test_probabilities_are_valid_and_respect_eligibility(truth: GroundTruth) -> None:
    for segment in SEGMENTS:
        probabilities = truth.probabilities(make_context(segment), OFFERS)
        for offer_id, p in probabilities.items():
            assert 0 <= p <= truth.max_probability
            if not OFFERS_BY_ID[offer_id].is_eligible(segment):
                assert p == 0


def test_payday_boosts_insurance_and_reduces_credit(truth: GroundTruth) -> None:
    normal, payday = make_context(), make_context(is_payday_week=True)
    funeral, paylater = OFFERS_BY_ID["funeral_cover"], OFFERS_BY_ID["paylater_voucher"]
    assert truth.take_up_probability(payday, funeral) > truth.take_up_probability(normal, funeral)
    assert truth.take_up_probability(payday, paylater) < truth.take_up_probability(normal, paylater)


def test_probability_is_capped() -> None:
    effect = Effect(name="hype", when={"segment": "student"}, multiply={"data": 100.0})
    truth = GroundTruth(
        base_take_up={"student": {"data_1gb": 0.5}}, effects=(effect,), max_probability=0.9
    )
    assert truth.take_up_probability(make_context("student"), OFFERS_BY_ID["data_1gb"]) == 0.9


def test_effect_with_list_condition() -> None:
    effect = Effect(name="border", when={"province": ["Limpopo"]}, multiply={"travel": 2.0})
    assert effect.applies_to(make_context(province="Limpopo"))
    assert not effect.applies_to(make_context(province="Gauteng"))
    assert effect.multiplier_for(OFFERS_BY_ID["roaming_pack"]) == 2.0
    assert effect.multiplier_for(OFFERS_BY_ID["data_1gb"]) == 1.0


# --- segment preferences (the "story" the recommender should discover) -------------


def test_each_segment_has_the_intended_favourite(tables) -> None:
    before, _ = tables
    assert best_offer(before, "student").startswith("data_")
    assert best_offer(before, "family") == "funeral_cover"
    assert best_offer(before, "small_business") == "data_20gb"
    assert best_offer(before, "pensioner") == "airtime_r50_bonus"


def test_best_offer_is_clearly_better_than_runner_up(tables) -> None:
    before, _ = tables
    for segment in SEGMENTS:
        ranked = before.loc[segment].dropna().sort_values(ascending=False)
        assert ranked.iloc[0] > 1.2 * ranked.iloc[1], f"{segment} has no clear winner"


def test_behaviour_shift_makes_streaming_the_young_professional_favourite(truth, tables) -> None:
    before, after = tables
    assert best_offer(before, "young_professional") != "streaming_bundle"
    assert best_offer(after, "young_professional") == "streaming_bundle"
    for segment in ("family", "small_business", "pensioner"):
        assert best_offer(after, segment) == best_offer(before, segment)
    assert not truth.is_shifted  # after_shift() returns a new object
    assert truth.after_shift().is_shifted


# --- validation ----------------------------------------------------------------------


def test_validation_rejects_ineligible_offer(truth: GroundTruth) -> None:
    bad = {**truth.base_take_up, "pensioner": {**truth.base_take_up["pensioner"], "data_20gb": 0.1}}
    with pytest.raises(ValueError, match="not eligible"):
        replace(truth, base_take_up=bad).validate(OFFERS)


def test_validation_rejects_unknown_effect_key(truth: GroundTruth) -> None:
    typo = Effect(name="typo", when={"is_payday": True}, multiply={"data": 1.1})
    with pytest.raises(ValueError, match="unknown"):
        replace(truth, effects=(typo,)).validate(OFFERS)


# --- architecture: the recommender must never see the ground truth ------------------


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_recommender_code_never_imports_the_simulator() -> None:
    package = PROJECT_ROOT / "src" / "recommender"
    allowed = {package / "simulator", package / "evaluation"}  # offline-only code
    for path in package.rglob("*.py"):
        if any(parent in allowed for parent in path.parents):
            continue
        leaks = {m for m in _imported_modules(path) if m.startswith("recommender.simulator")}
        assert not leaks, f"{path.relative_to(PROJECT_ROOT)} imports {leaks}"


def test_payday_context_flows_from_calendar(truth: GroundTruth, customers) -> None:
    student = next(c for c in customers if c.segment == "student")
    payday = Context.build(student, day=date(2026, 9, 26), time_of_day="night")
    normal = Context.build(student, day=date(2026, 9, 10), time_of_day="night")
    data = OFFERS_BY_ID["data_1gb"]
    assert truth.take_up_probability(payday, data) > truth.take_up_probability(normal, data)
