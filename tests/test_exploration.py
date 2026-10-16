"""The explore-vs-exploit measures, on hand-made runs where the answer is known."""

import pandas as pd
import pytest

from recommender.evaluation.exploration import (
    build_explore_story,
    chance_of_best,
    evidence,
    exploration_share,
)


def log(
    policy: str, firsts: list[str], segment: str = "student", shift_at: int = 6
) -> pd.DataFrame:
    """A per-app-open log: one segment, the given first offers in order."""
    return pd.DataFrame(
        {
            "policy": policy,
            "round": range(len(firsts)),
            "segment": segment,
            "top_offer": firsts,
            "shifted": [r >= shift_at for r in range(len(firsts))],
        }
    )


def test_exploration_share_counts_app_opens_away_from_the_favourite() -> None:
    firsts = ["a", "b", "a", "c"] + ["a", "a", "a", "b"]  # two periods of four
    shares = exploration_share(log("ts", firsts), "ts", every=4)
    assert shares["round"].tolist() == [4, 8]
    assert shares["share"].tolist() == [pytest.approx(0.5), pytest.approx(0.25)]


def test_each_segment_has_its_own_favourite() -> None:
    both = pd.concat([log("ts", ["a", "a"], "student"), log("ts", ["b", "b"], "family")])
    assert exploration_share(both, "ts", every=4)["share"].tolist() == [0.0]


def snapshots() -> pd.DataFrame:
    """Beliefs about two offers: early, uncertain and close; later, one clearly ahead."""
    return pd.DataFrame(
        {
            "policy": "ts",
            "context_key": "student",
            "round": [10, 10, 20, 20],
            "offer_id": ["a", "b", "a", "b"],
            "alpha": [3.0, 2.0, 301.0, 101.0],
            "beta": [7.0, 8.0, 701.0, 901.0],
        }
    )


def test_chance_of_best_grows_with_evidence() -> None:
    early = chance_of_best(snapshots(), "ts", "student", 10)
    late = chance_of_best(snapshots(), "ts", "student", 20)
    assert early.index[0] == "a" and 0.5 < early["a"] < 0.9  # ahead, but b might be best
    assert late["a"] > 0.999
    assert early.sum() == pytest.approx(1)
    assert chance_of_best(snapshots(), "ts", "pensioner", 10).empty


def test_evidence_adds_up_the_responses_behind_a_context() -> None:
    assert evidence(snapshots(), "ts", "student").to_dict() == {10: 16.0, 20: 2000.0}


def test_the_story_compares_the_learner_with_epsilon_greedy() -> None:
    learner = ["a", "b", "c", "d"] + ["a"] * 8 + ["e", "e", "a", "e"]  # explores, settles, shifts
    greedy = ["a", "b", "a", "a"] * 4  # strays a quarter of the time, always
    no_forgetting = ["a"] * 16
    rounds = pd.concat(
        [
            log("thompson_segment_decay", learner, shift_at=12),
            log("epsilon_greedy_segment", greedy, shift_at=12),
            log("thompson_segment", no_forgetting, shift_at=12),
        ]
    )
    table = pd.DataFrame(
        {"policy": ["thompson_segment", "epsilon_greedy_segment"], "regret": [1078.0, 1690.0]}
    )
    by_seed = pd.DataFrame(
        {
            "policy": ["thompson_segment", "epsilon_greedy_segment"] * 2,
            "seed": [0, 0, 1, 1],
            "regret": [1000.0, 1500.0, 1200.0, 1100.0],
        }
    )
    story = build_explore_story(rounds, table, by_seed, every=4)
    assert story.early == pytest.approx(0.75)
    assert story.settled == pytest.approx(0.0)  # the third period, before the shift
    assert story.fixed_rate_settled == pytest.approx(0.25)
    assert story.after_shift == pytest.approx(0.25)
    assert story.after_shift_no_forgetting == pytest.approx(0.0)
    assert story.regret == (1078.0, 1690.0)
    assert story.seeds_won == (1, 2)
