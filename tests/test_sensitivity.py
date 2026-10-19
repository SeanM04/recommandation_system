"""The sensitivity sweep's variants of the world, and a tiny sweep end to end."""

import pytest

from recommender.config import load_settings
from recommender.evaluation.sensitivity import (
    SCENARIOS,
    Scenario,
    flatten_preferences,
    run_sweep,
    scale_effects,
)
from recommender.simulator.engine import World


@pytest.fixture(scope="module")
def world() -> World:
    """The published world, loaded once for the module."""
    return World.load()


def test_scaling_effects_moves_every_multiplier_towards_or_away_from_one(world) -> None:
    truth = world.truth
    none, half, double = (scale_effects(truth, k) for k in (0.0, 0.5, 2.0))
    for original, gone, weaker, stronger in zip(
        truth.effects, none.effects, half.effects, double.effects, strict=True
    ):
        for key, m in original.multiply.items():
            assert gone.multiply[key] == 1.0
            assert weaker.multiply[key] == pytest.approx(1 + (m - 1) / 2)
            assert stronger.multiply[key] == pytest.approx(max(0.0, 1 + 2 * (m - 1)))
            assert stronger.multiply[key] >= 0
    assert scale_effects(truth, 1.0).effects == truth.effects


def test_flattening_keeps_each_segments_average_and_narrows_the_gaps(world) -> None:
    truth = world.truth
    flat = flatten_preferences(truth, 0.5)
    for segment, rates in truth.base_take_up.items():
        new = flat.base_take_up[segment]
        mean = sum(rates.values()) / len(rates)
        assert sum(new.values()) / len(new) == pytest.approx(mean)
        assert max(new.values()) - min(new.values()) == pytest.approx(
            (max(rates.values()) - min(rates.values())) / 2
        )
    flat.validate(world.offers)  # still a valid world


def test_a_tiny_sweep_runs_every_scenario(world) -> None:
    scenarios = [SCENARIOS[0], Scenario("No shift", "never", lambda t: t, shift_at=None)]
    rows = run_sweep(
        load_settings(),
        world,
        ["random", "thompson_segment"],
        rounds=300,
        seeds=[0, 1],
        scenarios=scenarios,
        progress=lambda _: None,
    )
    assert len(rows) == 2 * 2 * 2  # scenarios x policies x seeds
    assert set(rows["scenario"]) == {"As published", "No shift"}
    assert rows["share_of_oracle"].between(0, 1).all()
