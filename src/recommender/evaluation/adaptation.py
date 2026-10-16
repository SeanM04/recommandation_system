"""The numbers behind the dashboard's behaviour-shift story, computed from a saved run.

What changed comes from the simulator's ground truth (each segment's favourite before and
after the shift); how each policy responded comes from the files ``make compare`` saves.
Overall claims are paired by seed; per-segment curves and counts use the first seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

LEARNER = "thompson_segment_decay"
NO_FORGETTING = "thompson_segment"
BASELINE = "static"


def first_share(
    rounds: pd.DataFrame, policy: str, segment: str, offer: str, window: int = 400
) -> pd.Series:
    """Share of the segment's last ``window`` app opens with ``offer`` ranked first,
    indexed by app open (first seed)."""
    opens = rounds[(rounds["policy"] == policy) & (rounds["segment"] == segment)]
    return (opens["top_offer"] == offer).rolling(window).mean().set_axis(opens["round"])


def take_up_per_1000(
    rounds: pd.DataFrame, policy: str, segment: str, shift_round: int
) -> tuple[float, float]:
    """Expected take-ups per 1,000 of the segment's app opens, before and after the shift."""
    opens = rounds[(rounds["policy"] == policy) & (rounds["segment"] == segment)]
    before = opens.loc[opens["round"] < shift_round, "expected_take_ups"].mean()
    after = opens.loc[opens["round"] >= shift_round, "expected_take_ups"].mean()
    return float(before * 1000), float(after * 1000)


@dataclass(frozen=True)
class ShiftStory:
    """What the behaviour-shift story states, for one saved run and one segment."""

    segment: str
    old_favourite: str
    new_favourite: str
    shift_round: int
    seeds: int
    # Overall, after the shift (mean over seeds): take-ups lost to the oracle.
    regret_after: dict[str, float]
    beats_baseline_after: int  # seeds where the learner lost fewer than the fixed ranking
    forgetting_cost_before: float  # extra take-ups lost before the shift, with forgetting
    forgetting_saving_after: float  # take-ups saved after the shift, with forgetting
    seeds_saved_after: int
    seeds_cost_before: int  # seeds where forgetting lost more before the shift
    # For this segment (first seed).
    adapted_after: dict[str, float | None]  # app opens until the new favourite led; None: never
    baseline_new_first: float  # share of the fixed ranking's app opens with the new favourite first
    take_ups: dict[str, tuple[float, float]]  # per 1,000 app opens, before / after

    @property
    def fewer_than_baseline(self) -> float:
        """How much less the learner lost after the shift than the fixed ranking (0-1)."""
        return 1 - self.regret_after[LEARNER] / self.regret_after[BASELINE]


def build_shift_story(
    table: pd.DataFrame,
    by_seed: pd.DataFrame,
    rounds: pd.DataFrame,
    convergence: pd.DataFrame,
    favourites: tuple[dict[str, str], dict[str, str]],
    segment: str,
) -> ShiftStory:
    """Compute the story of the shift for ``segment``. ``favourites`` maps each segment to
    its favourite offer before and after the shift (from the simulator's ground truth)."""
    shift_round = int(rounds.loc[rounds["shifted"], "round"].min())
    after = by_seed.pivot_table(index="seed", columns="policy", values="regret_after_shift")
    before = by_seed.assign(before=by_seed["regret"] - by_seed["regret_after_shift"]).pivot_table(
        index="seed", columns="policy", values="before"
    )
    saving = after[NO_FORGETTING] - after[LEARNER]
    cost = before[LEARNER] - before[NO_FORGETTING]

    adapted = convergence[
        (convergence["phase"] == "after the shift") & (convergence["segment"] == segment)
    ].set_index("policy")["app_opens"]
    old, new = favourites[0][segment], favourites[1][segment]
    baseline_opens = rounds[
        (rounds["policy"] == BASELINE)
        & (rounds["segment"] == segment)
        & (rounds["round"] >= shift_round)
    ]
    return ShiftStory(
        segment=segment,
        old_favourite=old,
        new_favourite=new,
        shift_round=shift_round,
        seeds=int(len(after)),
        regret_after={
            p: float(v) for p, v in table.set_index("policy")["regret_after_shift"].items()
        },
        beats_baseline_after=int((after[LEARNER] < after[BASELINE]).sum()),
        forgetting_cost_before=float(cost.mean()),
        forgetting_saving_after=float(saving.mean()),
        seeds_saved_after=int((saving > 0).sum()),
        seeds_cost_before=int((cost > 0).sum()),
        adapted_after={
            p: (None if p not in adapted or pd.isna(adapted[p]) else float(adapted[p]))
            for p in (LEARNER, NO_FORGETTING, BASELINE)
        },
        baseline_new_first=float((baseline_opens["top_offer"] == new).mean()),
        take_ups={
            p: take_up_per_1000(rounds, p, segment, shift_round)
            for p in (LEARNER, NO_FORGETTING, BASELINE)
        },
    )
