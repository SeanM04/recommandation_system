"""The behaviour-shift measures, on a hand-made run where the answer is known."""

import pandas as pd
import pytest

from recommender.evaluation.adaptation import build_shift_story, first_share, take_up_per_1000

SHIFT = 4
POLICIES = ["thompson_segment_decay", "thompson_segment", "static"]


def make_run() -> tuple[pd.DataFrame, ...]:
    """Eight app opens by students; the favourite changes from a to b after the fourth."""
    firsts = {
        "thompson_segment_decay": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "thompson_segment": ["a", "a", "a", "a", "a", "a", "b", "b"],
        "static": ["c"] * 8,
    }
    expected = {
        "thompson_segment_decay": [0.2] * 4 + [0.4] * 4,
        "thompson_segment": [0.2] * 4 + [0.3] * 4,
        "static": [0.1] * 8,
    }
    rounds = pd.DataFrame(
        [
            {
                "policy": p,
                "round": r,
                "segment": "student",
                "shifted": r >= SHIFT,
                "top_offer": firsts[p][r],
                "expected_take_ups": expected[p][r],
            }
            for p in POLICIES
            for r in range(8)
        ]
    )
    table = pd.DataFrame({"policy": POLICIES, "regret_after_shift": [450.0, 580.0, 3400.0]})
    by_seed = pd.DataFrame(
        [
            {"policy": p, "seed": seed, "regret": total, "regret_after_shift": after}
            for seed, rows in enumerate(
                [
                    {
                        "thompson_segment_decay": (1000, 440),
                        "thompson_segment": (1000, 570),
                        "static": (5500, 3300),
                    },
                    {
                        "thompson_segment_decay": (1100, 460),
                        "thompson_segment": (1050, 590),
                        "static": (5600, 3500),
                    },
                ]
            )
            for p, (total, after) in rows.items()
        ]
    )
    convergence = pd.DataFrame(
        {
            "policy": POLICIES,
            "phase": "after the shift",
            "segment": "student",
            "app_opens": [1.0, 3.0, float("nan")],
        }
    )
    return table, by_seed, rounds, convergence


def test_first_share_tracks_the_new_favourite() -> None:
    _, _, rounds, _ = make_run()
    share = first_share(rounds, "thompson_segment", "student", "b", window=2)
    assert share.loc[[5, 6, 7]].tolist() == [0.0, 0.5, 1.0]


def test_take_ups_before_and_after() -> None:
    _, _, rounds, _ = make_run()
    before, after = take_up_per_1000(rounds, "thompson_segment_decay", "student", SHIFT)
    assert (before, after) == (pytest.approx(200), pytest.approx(400))


def test_the_story_of_the_shift() -> None:
    story = build_shift_story(*make_run(), ({"student": "a"}, {"student": "b"}), "student")
    assert (story.old_favourite, story.new_favourite, story.shift_round) == ("a", "b", SHIFT)
    assert story.fewer_than_baseline == pytest.approx(1 - 450 / 3400)
    assert (story.beats_baseline_after, story.seeds) == (2, 2)
    # Before: decay lost 560 and 640 against 430 and 460 without forgetting.
    assert story.forgetting_cost_before == pytest.approx(155)
    assert story.forgetting_saving_after == pytest.approx(130)
    assert (story.seeds_saved_after, story.seeds_cost_before) == (2, 2)
    assert story.adapted_after == {
        "thompson_segment_decay": 1.0,
        "thompson_segment": 3.0,
        "static": None,
    }
    assert story.baseline_new_first == 0.0
    assert story.take_ups["static"] == (pytest.approx(100), pytest.approx(100))
