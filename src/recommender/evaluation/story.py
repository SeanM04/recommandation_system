"""The numbers behind the dashboard's comparison story, computed from a saved run.

The comparison page follows Brent Dykes' data storytelling arc: the setting and a hook
(what the usual approach leaves on the table), rising insights (one chart and one message
each), the aha moment, then the recommendation and next steps. Everything it says comes
from here, computed from whichever run is on disk, so a 3-seed run never borrows the
10-seed numbers and every "in N of N runs" is counted, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Plain names for a business audience; the policy ids stay visible in hover text.
LABELS = {
    "random": "Random offers",
    "static": "Fixed popularity ranking",
    "epsilon_greedy": "One learning model for everyone (epsilon-greedy)",
    "thompson": "One learning model for everyone",
    "epsilon_greedy_segment": "Learning per segment (epsilon-greedy)",
    "thompson_segment": "Learning per segment",
    "thompson_segment_payday": "Learning per segment and payday week",
    "thompson_segment_decay": "Learning per segment, with forgetting",
    "thompson_segment_revenue": "Learning per segment, from revenue",
}
# Each step adds one idea, from no intelligence to the recommended policy.
LADDER = ["random", "static", "thompson", "thompson_segment", "thompson_segment_decay"]
BASELINE = "static"


def label(policy: str) -> str:
    """The plain name of a policy (its id when there is none)."""
    return LABELS.get(policy, policy)


@dataclass(frozen=True)
class Story:
    """Everything the comparison story states, for one saved run."""

    best: str  # lowest mean regret
    seeds: int
    app_opens: int
    shift_round: int | None
    oracle_per_1000: float  # take-ups an all-knowing ranker would get per 1,000 app opens
    per_1000: dict[str, float]  # take-ups per 1,000 app opens, by policy
    share_of_oracle: dict[str, float]
    best_beats_baseline: int  # seeds where the best policy's regret was below the baseline's
    best_beats_random: int
    overtakes_baseline_at: int | None  # app opens after which the best leads for good
    extra_at_shift: float  # expected take-ups won over the baseline by the shift (first seed)
    extra_at_end: float  # ... and by the end of the run
    before_after: dict[str, tuple[float, float]]  # take-up per open before / after the shift
    tied_with_best: list[str]  # beat by the best in fewer than 80% of seeds
    forgetting_after_shift: tuple[int, float] | None  # seeds won, take-ups saved on average
    revenue_trade: tuple[float, float] | None  # revenue uplift, take-up drop of the revenue twin
    adaptation: tuple[float | None, float | None] | None  # app opens to adapt, with / without

    @property
    def uplift(self) -> float:
        """Relative gain in take-ups per app open of the best policy over the baseline."""
        return self.per_1000[self.best] / self.per_1000[BASELINE] - 1


def _paired(by_seed: pd.DataFrame, a: str, b: str, metric: str) -> pd.Series:
    """Per seed, ``metric`` of policy ``a`` minus policy ``b`` (both saw the same app opens)."""
    wide = by_seed.pivot_table(index="seed", columns="policy", values=metric)
    if a not in wide or b not in wide:
        return pd.Series(dtype=float)
    return (wide[a] - wide[b]).dropna()


def _gap(rounds: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Cumulative expected take-ups of policy ``a`` minus ``b``, by app open (first seed)."""
    expected = rounds.pivot_table(index="round", columns="policy", values="expected_take_ups")
    return (expected[a] - expected[b]).cumsum()


def build_story(
    table: pd.DataFrame, by_seed: pd.DataFrame, rounds: pd.DataFrame, convergence: pd.DataFrame
) -> Story:
    """Compute the story of one run from the files ``make compare`` saves."""
    table = table.set_index("policy")
    best = str(table["regret"].idxmin())
    seeds = by_seed["seed"].nunique()
    shifted = rounds.loc[rounds["shifted"], "round"]
    shift_round = int(shifted.min()) if len(shifted) else None
    oracle = (by_seed["expected_take_up_rate"] / by_seed["share_of_oracle"]).mean()

    gap = _gap(rounds, best, BASELINE)
    behind = gap.index[gap <= 0]  # app opens after which the best had not yet pulled ahead
    if len(behind) == 0:
        overtakes = 0
    elif behind.max() == gap.index.max():
        overtakes = None  # still behind at the end of the run
    else:
        overtakes = int(behind.max()) + 1

    tied = [
        policy
        for policy in table.index
        if policy not in (best, "random", BASELINE)
        and (_paired(by_seed, best, policy, "regret") < 0).sum() < 0.8 * seeds
    ]

    forgetting = None
    after = _paired(by_seed, "thompson_segment_decay", "thompson_segment", "regret_after_shift")
    if len(after):
        forgetting = (int((after < 0).sum()), float(-after.mean()))

    revenue_trade = None
    if {"thompson_segment_revenue", "thompson_segment_decay"} <= set(table.index):
        twin, base = table.loc["thompson_segment_revenue"], table.loc["thompson_segment_decay"]
        revenue_trade = (
            float(twin["revenue_per_open"] / base["revenue_per_open"] - 1),
            float(1 - twin["take_up_rate"] / base["take_up_rate"]),
        )

    adaptation = None
    after_shift = convergence[
        (convergence["phase"] == "after the shift")
        & (convergence["segment"] == "young_professional")
    ].set_index("policy")["app_opens"]
    if {"thompson_segment_decay", "thompson_segment"} <= set(after_shift.index):
        with_forgetting, without = (
            None if pd.isna(after_shift[p]) else float(after_shift[p])
            for p in ("thompson_segment_decay", "thompson_segment")
        )
        adaptation = (with_forgetting, without)

    return Story(
        best=best,
        seeds=seeds,
        app_opens=int(rounds["round"].max()) + 1,
        shift_round=shift_round,
        oracle_per_1000=float(oracle * 1000),
        per_1000={p: float(v * 1000) for p, v in table["take_up_rate"].items()},
        share_of_oracle={p: float(v) for p, v in table["share_of_oracle"].items()},
        best_beats_baseline=int((_paired(by_seed, best, BASELINE, "regret") < 0).sum()),
        best_beats_random=int((_paired(by_seed, best, "random", "regret") < 0).sum()),
        overtakes_baseline_at=overtakes,
        extra_at_shift=float(gap.loc[shift_round - 1]) if shift_round else float(gap.iloc[-1]),
        extra_at_end=float(gap.iloc[-1]),
        before_after={
            p: (float(row["take_up_before_shift"]), float(row["take_up_after_shift"]))
            for p, row in table.iterrows()
        },
        tied_with_best=tied,
        forgetting_after_shift=forgetting,
        revenue_trade=revenue_trade,
        adaptation=adaptation,
    )


def gap_curve(rounds: pd.DataFrame, policy: str, every: int = 250) -> pd.DataFrame:
    """Expected take-ups ``policy`` won over the baseline so far, sampled every ``every``."""
    gap = _gap(rounds, policy, BASELINE)
    sampled = gap.iloc[every - 1 :: every]
    return pd.DataFrame({"round": sampled.index + 1, "extra": sampled.to_numpy()})
