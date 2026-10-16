"""The numbers behind the dashboard's explore-vs-exploit story, computed from a saved run.

Three measures, each from files ``make compare`` saves:

- exploration share: of a segment's app opens in a period, the share whose first offer
  was not the one the policy showed first most often in that period (its favourite then);
- chance of being best: for each offer, the probability under the policy's Beta beliefs
  that it has the highest take-up rate. Thompson sampling shows an offer first with
  exactly this probability, so it is how much the policy explores, offer by offer;
- evidence: how many responses the beliefs in a context rest on (alpha + beta - 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEARNER = "thompson_segment_decay"
NO_FORGETTING = "thompson_segment"
FIXED_RATE = "epsilon_greedy_segment"  # explores a fixed share of the time
BLOCK = 2500  # app opens per period


def exploration_share(rounds: pd.DataFrame, policy: str, every: int = BLOCK) -> pd.DataFrame:
    """Per period of ``every`` app opens: the share of app opens that strayed from each
    segment's most-shown first offer in that period (columns: round, share)."""
    log = rounds[rounds["policy"] == policy]
    rows = []
    for block, chunk in log.groupby(log["round"] // every, sort=True):
        firsts = chunk.groupby("segment")["top_offer"]
        favourite_count = firsts.agg(lambda offers: offers.value_counts().iloc[0]).sum()
        rows.append({"round": int(block + 1) * every, "share": 1 - favourite_count / len(chunk)})
    return pd.DataFrame(rows, columns=["round", "share"])


def chance_of_best(
    snapshots: pd.DataFrame,
    policy: str,
    context: str,
    at: int,
    draws: int = 20_000,
    seed: int = 0,
) -> pd.Series:
    """Each offer's probability of having the highest take-up rate under the beliefs saved
    after ``at`` app opens, by Monte Carlo from the Beta beliefs (largest first)."""
    beliefs = snapshots[
        (snapshots["policy"] == policy)
        & (snapshots["context_key"] == context)
        & (snapshots["round"] == at)
    ]
    if beliefs.empty:
        return pd.Series(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.beta(
        beliefs["alpha"].to_numpy(), beliefs["beta"].to_numpy(), (draws, len(beliefs))
    )
    wins = np.bincount(samples.argmax(axis=1), minlength=len(beliefs)) / draws
    return pd.Series(wins, index=beliefs["offer_id"].to_numpy()).sort_values(ascending=False)


def evidence(snapshots: pd.DataFrame, policy: str, context: str) -> pd.Series:
    """Responses the policy's beliefs in ``context`` rest on, after each saved app open."""
    beliefs = snapshots[(snapshots["policy"] == policy) & (snapshots["context_key"] == context)]
    return (beliefs["alpha"] + beliefs["beta"] - 2).groupby(beliefs["round"]).sum()


@dataclass(frozen=True)
class ExploreStory:
    """What the explore-vs-exploit story states, for one saved run."""

    early: float  # the learner's exploration share in the first period
    settled: float  # ... its typical share once confident (median before the shift)
    fixed_rate_settled: float  # epsilon-greedy's typical share over the same periods
    fixed_rate_floor: float  # ... and its lowest after the first period
    after_shift: float | None  # the learner's share in the first period after the shift
    after_shift_no_forgetting: float | None  # ... without forgetting
    regret: tuple[float, float] | None  # Thompson vs epsilon-greedy per segment (mean regret)
    seeds_won: tuple[int, int] | None  # seeds where Thompson's regret was lower, of all


def _settled(shares: pd.DataFrame, shift_round: int | None) -> pd.Series:
    """The periods once the policy had settled: after the first two, before the shift."""
    settled = shares.iloc[2:]
    if shift_round:
        settled = settled[settled["round"] <= shift_round]
    return settled["share"]


def _first_after(shares: pd.DataFrame, shift_round: int | None) -> float | None:
    """The share in the first period that starts at or after the shift."""
    if not shift_round:
        return None
    after = shares[shares["round"] > shift_round]
    return float(after["share"].iloc[0]) if len(after) else None


def build_explore_story(
    rounds: pd.DataFrame, table: pd.DataFrame, by_seed: pd.DataFrame, every: int = BLOCK
) -> ExploreStory:
    """Compute the explore-vs-exploit story from the files ``make compare`` saves, in
    periods of ``every`` app opens."""
    shifted = rounds.loc[rounds["shifted"], "round"]
    shift_round = int(shifted.min()) if len(shifted) else None
    learner = exploration_share(rounds, LEARNER, every)
    fixed_rate = exploration_share(rounds, FIXED_RATE, every)
    no_forgetting = exploration_share(rounds, NO_FORGETTING, every)

    regret = seeds_won = None
    means = table.set_index("policy")["regret"]
    if {NO_FORGETTING, FIXED_RATE} <= set(means.index):
        regret = (float(means[NO_FORGETTING]), float(means[FIXED_RATE]))
        wide = by_seed.pivot_table(index="seed", columns="policy", values="regret")
        lower = (wide[NO_FORGETTING] < wide[FIXED_RATE]).sum()
        seeds_won = (int(lower), int(len(wide)))

    return ExploreStory(
        early=float(learner["share"].iloc[0]),
        settled=float(_settled(learner, shift_round).median()),
        fixed_rate_settled=float(_settled(fixed_rate, shift_round).median()),
        fixed_rate_floor=float(fixed_rate["share"].iloc[1:].min()),
        after_shift=_first_after(learner, shift_round),
        after_shift_no_forgetting=_first_after(no_forgetting, shift_round),
        regret=regret,
        seeds_won=seeds_won,
    )
