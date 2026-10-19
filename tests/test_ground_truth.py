import ast
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recommender.config import PROJECT_ROOT, load_offers, load_settings
from recommender.domain import IGNORED, OUTCOMES, SEGMENTS, TAKEN_UP, VIEWED, Context, Customer
from recommender.simulator.ground_truth import (
    Effect,
    GroundTruth,
    ResponseModel,
    take_up_table,
    winners,
)
from recommender.simulator.population import generate_customers
from recommender.simulator.settings import load_simulation_settings

SETTINGS = load_settings()
SIMULATION = load_simulation_settings()
DAYS = SIMULATION.calendar.days()
OFFERS = load_offers()
OFFERS_BY_ID = {offer.offer_id: offer for offer in OFFERS}


@pytest.fixture(scope="module")
def truth() -> GroundTruth:
    """The configured hidden ground truth."""
    return GroundTruth.from_yaml(SIMULATION.ground_truth, OFFERS)


@pytest.fixture(scope="module")
def customers() -> list[Customer]:
    """The generated customer population."""
    frame = generate_customers(5000, SIMULATION.population.segment_mix, seed=SETTINGS.seed)
    return [Customer(**record) for record in frame.to_dict("records")]


@pytest.fixture(scope="module")
def tables(truth, customers) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take-up rates per segment before and after the shift, and by payday week."""
    before = take_up_table(truth, customers, OFFERS, DAYS, samples_per_segment=2000)
    after = take_up_table(truth.after_shift(), customers, OFFERS, DAYS, samples_per_segment=2000)
    return before, after


def make_context(segment: str = "family", **overrides) -> Context:
    """A context for a typical customer of ``segment``, with any field overridden."""
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


def best_offer(table: pd.DataFrame, key: str | tuple) -> str:
    """The offer with the highest take-up rate in one row of a table."""
    return table.loc[key].idxmax()


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
    assert best_offer(before, "small_business") == "paylater_voucher"
    assert best_offer(before, "pensioner") == "airtime_r50_bonus"


def assert_clear_winners(table: pd.DataFrame, min_lead: float = 1.2) -> None:
    """Fail unless every row's favourite leads the runner-up by at least ``min_lead``."""
    lead = winners(table)["lead"]
    assert (lead > min_lead).all(), f"no clear winner: {lead[lead <= min_lead].to_dict()}"


def test_best_offer_is_clearly_better_than_runner_up(tables) -> None:
    before, after = tables
    assert_clear_winners(before)
    assert_clear_winners(after)


def test_payday_changes_the_favourite_for_pensioners_and_small_businesses(truth, customers) -> None:
    by_payday = take_up_table(
        truth, customers, OFFERS, DAYS, by=("segment", "is_payday_week"), samples_per_segment=2000
    )
    assert_clear_winners(by_payday)
    assert best_offer(by_payday, ("pensioner", False)) == "airtime_r50_bonus"
    assert best_offer(by_payday, ("pensioner", True)) == "electricity_cashback"
    assert best_offer(by_payday, ("small_business", False)) == "paylater_voucher"
    assert best_offer(by_payday, ("small_business", True)) == "data_20gb"
    for segment in ("student", "young_professional", "family"):
        assert best_offer(by_payday, (segment, False)) == best_offer(by_payday, (segment, True))


def test_revenue_favourites_are_clear_and_differ_from_take_up(tables) -> None:
    before, _ = tables
    margins = pd.Series({offer.offer_id: offer.margin_zar for offer in OFFERS})
    revenue = before.mul(margins, axis=1)
    assert_clear_winners(revenue)
    assert best_offer(revenue, "student") == "data_5gb"  # take-up favourite is data_1gb
    assert best_offer(revenue, "pensioner") == "funeral_cover"  # take-up: airtime


def test_behaviour_shift_changes_two_favourites(truth, tables) -> None:
    before, after = tables
    assert best_offer(before, "young_professional") != "streaming_bundle"
    assert best_offer(after, "young_professional") == "streaming_bundle"
    assert best_offer(before, "family") == "funeral_cover"
    assert best_offer(after, "family") == "electricity_cashback"  # a well-known runner-up
    for segment in ("student", "small_business", "pensioner"):
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
    """Every module a source file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _recommender_modules() -> list[Path]:
    """Every module that serves recommendations (simulator and offline evaluation excluded)."""
    package = PROJECT_ROOT / "src" / "recommender"
    offline = {package / "simulator", package / "evaluation"}
    return [p for p in package.rglob("*.py") if not any(d in offline for d in p.parents)]


def test_recommender_code_never_imports_the_simulator() -> None:
    for path in _recommender_modules():
        leaks = {m for m in _imported_modules(path) if m.startswith("recommender.simulator")}
        assert not leaks, f"{path.relative_to(PROJECT_ROOT)} imports {leaks}"


def test_recommender_code_never_mentions_the_hidden_config() -> None:
    for path in _recommender_modules():
        source = path.read_text(encoding="utf-8")
        for secret in ("ground_truth", "simulation.yaml"):
            assert secret not in source, f"{path.relative_to(PROJECT_ROOT)} mentions {secret}"


def test_payday_context_flows_from_calendar(truth: GroundTruth, customers) -> None:
    student = next(c for c in customers if c.segment == "student")
    payday = Context.build(student, day=date(2026, 9, 26), time_of_day="night")
    normal = Context.build(student, day=date(2026, 9, 10), time_of_day="night")
    data = OFFERS_BY_ID["data_1gb"]
    assert truth.take_up_probability(payday, data) > truth.take_up_probability(normal, data)


# --- response model for a ranked list -------------------------------------------------


def test_response_model_is_loaded_from_yaml(truth: GroundTruth) -> None:
    assert truth.response.attention == (1.0, 0.6, 0.4, 0.25, 0.15)
    assert truth.response.max_offers == 5


def test_top_offer_taken_ends_the_visit() -> None:
    model = ResponseModel(attention=(1.0, 1.0, 1.0))
    outcomes = model.draw([1.0, 1.0, 1.0], np.random.default_rng(0))
    assert outcomes == [TAKEN_UP, IGNORED, IGNORED]


def test_noticed_but_not_taken_is_viewed_and_unnoticed_is_ignored() -> None:
    model = ResponseModel(attention=(1.0, 0.0))
    assert model.draw([0.0, 1.0], np.random.default_rng(0)) == [VIEWED, IGNORED]


def test_at_most_one_take_up_per_visit(truth: GroundTruth) -> None:
    rng = np.random.default_rng(1)
    for _ in range(500):
        outcomes = truth.response.draw([0.9, 0.9, 0.9], rng)
        assert outcomes.count(TAKEN_UP) <= 1
        assert set(outcomes) <= set(OUTCOMES)


def test_simulated_take_ups_match_expected_value(truth: GroundTruth) -> None:
    probabilities = [0.3, 0.2, 0.1]
    rng = np.random.default_rng(2)
    draws = [truth.response.draw(probabilities, rng).count(TAKEN_UP) for _ in range(20_000)]
    assert np.mean(draws) == pytest.approx(truth.response.expected_value(probabilities), abs=0.01)


def test_expected_value_weights_by_value_and_position() -> None:
    model = ResponseModel(attention=(1.0, 0.5))
    assert model.expected_value([0.5], values=[10.0]) == pytest.approx(5.0)
    # second slot: reached with prob 0.5 (no take-up above), noticed with prob 0.5
    assert model.expected_value([0.5, 0.4]) == pytest.approx(0.5 + 0.5 * 0.5 * 0.4)


def test_response_model_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        ResponseModel(attention=(1.0, 1.5))
    with pytest.raises(ValueError, match="at most 1 offers"):
        ResponseModel(attention=(1.0,)).draw([0.1, 0.2], np.random.default_rng(0))


def test_respond_maps_outcomes_to_offer_ids(truth: GroundTruth) -> None:
    shown = [OFFERS_BY_ID["funeral_cover"], OFFERS_BY_ID["electricity_cashback"]]
    outcomes = truth.respond(make_context("family"), shown, np.random.default_rng(3))
    assert list(outcomes) == ["funeral_cover", "electricity_cashback"]
