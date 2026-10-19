"""The numbers behind the comparison story, on a tiny hand-made run."""

import pandas as pd
import pytest

from recommender.evaluation.story import build_story, gap_curve, label

POLICIES = ["random", "static", "thompson_segment", "thompson_segment_decay"]


def make_run(best_rounds_behind: int = 3, rounds: int = 10) -> tuple[pd.DataFrame, ...]:
    """A 3-seed run where the decay policy trails the static one for its first few app opens."""
    table = pd.DataFrame(
        {
            "policy": POLICIES,
            "take_up_rate": [0.15, 0.17, 0.26, 0.263],
            "take_up_before_shift": [0.14, 0.168, 0.234, 0.231],
            "take_up_after_shift": [0.157, 0.175, 0.289, 0.295],
            "revenue_per_open": [5.8, 4.4, 10.0, 9.7],
            "regret": [6700, 5500, 1078, 1041],
            "share_of_oracle": [0.53, 0.61, 0.924, 0.927],
        }
    )
    regret = {  # per seed: decay beats static every time, ties segment on one of three
        "random": [6700, 6710, 6690],
        "static": [5500, 5600, 5400],
        "thompson_segment": [1000, 1100, 1135],
        "thompson_segment_decay": [1040, 1050, 1033],
    }
    after = {"thompson_segment": [600, 560, 570], "thompson_segment_decay": [450, 440, 448]}
    by_seed = pd.DataFrame(
        [
            {
                "policy": policy,
                "seed": seed,
                "regret": regret[policy][seed],
                "regret_after_shift": after.get(policy, [0, 0, 0])[seed],
                "expected_take_up_rate": 0.2,
                "share_of_oracle": 0.8,
            }
            for policy in POLICIES
            for seed in range(3)
        ]
    )
    expected = {
        "static": [0.2] * rounds,
        "thompson_segment_decay": [0.1] * best_rounds_behind
        + [0.4] * (rounds - best_rounds_behind),
    }
    log = pd.DataFrame(
        [
            {"policy": p, "round": r, "shifted": r >= rounds // 2, "expected_take_ups": v}
            for p, values in expected.items()
            for r, v in enumerate(values)
        ]
    )
    convergence = pd.DataFrame(
        {
            "policy": ["thompson_segment_decay", "thompson_segment"],
            "phase": ["after the shift"] * 2,
            "segment": ["young_professional"] * 2,
            "app_opens": [1460.0, float("nan")],
        }
    )
    return table, by_seed, log, convergence


def test_the_story_of_a_run() -> None:
    story = build_story(*make_run())
    assert story.best == "thompson_segment_decay"
    assert (story.seeds, story.app_opens, story.shift_round) == (3, 10, 5)
    assert story.oracle_per_1000 == pytest.approx(250)  # 0.2 / 0.8 per app open
    assert story.per_1000["static"] == pytest.approx(170)
    assert story.uplift == pytest.approx(0.263 / 0.17 - 1)
    assert (story.best_beats_baseline, story.best_beats_random) == (3, 3)
    assert story.tied_with_best == ["thompson_segment"]  # beaten in only 2 of 3 seeds
    assert story.forgetting_after_shift == (3, pytest.approx(130.6667, abs=1e-3))
    assert story.adaptation == (1460.0, None)  # without forgetting: not within the run
    assert story.revenue_trade is None  # no revenue policy in this run


def test_the_story_recommends_only_what_the_api_can_serve() -> None:
    table, by_seed, log, convergence = make_run()
    # A simulation-only policy with the lowest regret of all, in the table and every seed.
    table = pd.concat(
        [table, pd.DataFrame([{"policy": "thompson_features", "regret": 900.0}])],
        ignore_index=True,
    )
    extra = by_seed[by_seed["policy"] == "thompson_segment_decay"].assign(
        policy="thompson_features", regret=900.0
    )
    by_seed = pd.concat([by_seed, extra], ignore_index=True)

    story = build_story(table, by_seed, log, convergence, servable=POLICIES)
    assert story.best == "thompson_segment_decay"
    assert story.tied_with_best == ["thompson_segment"]  # never the simulation-only one


def test_when_the_learner_overtakes_the_baseline() -> None:
    # Behind by 0.1 for 3 app opens (-0.3), then 0.2 ahead each: -0.1 after the 4th app
    # open, +0.1 after the 5th, and ahead from then on.
    story = build_story(*make_run(best_rounds_behind=3))
    assert story.overtakes_baseline_at == 4
    assert story.extra_at_end == pytest.approx(-0.3 + 7 * 0.2)
    never = build_story(*make_run(best_rounds_behind=10))
    assert never.overtakes_baseline_at is None


def test_gap_curve_samples_the_cumulative_lead() -> None:
    _, _, log, _ = make_run()
    curve = gap_curve(log, "thompson_segment_decay", every=5)
    assert curve["round"].tolist() == [5, 10]
    assert curve["extra"].tolist() == pytest.approx([-0.3 + 2 * 0.2, -0.3 + 7 * 0.2])


def test_plain_names_for_a_business_audience() -> None:
    assert label("static") == "Fixed popularity ranking"
    assert label("something_new") == "something_new"
