"""Run every configured policy through the same simulated world and compare them.

Fair by construction: every policy faces the identical sequence of app opens (common
random numbers) and the same behaviour shift. The static baseline is "batch-trained" on
a warm-up month of responses gathered by a random policy before the experiment starts;
the learning policies start with no data at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.algorithms.factory import build_policy
from recommender.algorithms.random_policy import RandomPolicy
from recommender.algorithms.rewards import binary_reward
from recommender.algorithms.static import StaticPolicy
from recommender.config import Settings
from recommender.evaluation.metrics import convergence, summarize
from recommender.simulator.engine import World, simulate
from recommender.simulator.ground_truth import take_up_table
from recommender.store.options_store import InMemoryOptionsStore


@dataclass
class Comparison:
    summary: pd.DataFrame  # one row per policy and seed
    rounds: pd.DataFrame  # every app open of the first seed, all policies
    snapshots: pd.DataFrame  # beliefs (alpha, beta) over time, first seed, learning policies


def warm_up_history(
    world: World, rounds: int, seed: int, n_offers: int = 3
) -> list[tuple[str, float]]:
    """Last month's data for the batch model: (offer_id, reward) pairs from a random
    policy in the world before the shift."""
    rng = np.random.default_rng([seed, 3])
    policy = RandomPolicy(rng=rng)
    history = []
    for round_ in range(rounds):
        context = world.app_open(rng, round_, rounds)
        eligible = world.eligible(context.segment)
        shown = [scored.offer_id for scored in policy.rank(context, eligible, n_offers)]
        probabilities = world.truth.probabilities(context, eligible)
        outcomes = world.truth.response.draw([probabilities[o] for o in shown], rng)
        for offer_id, outcome in zip(shown, outcomes, strict=True):
            reward = binary_reward(world.offers_by_id[offer_id], outcome)
            if reward is not None:
                history.append((offer_id, reward))
    return history


def run_comparison(
    settings: Settings,
    world: World,
    *,
    rounds: int,
    shift_round: int | None,
    seeds: Sequence[int],
    warm_up_rounds: int = 5000,
    snapshot_every: int = 2500,
    progress=print,
) -> Comparison:
    """Every policy in ``settings.policies``, on every seed, through the same world."""
    n_offers = settings.recommendation.number_of_offers
    summaries, round_logs, snapshots = [], [], []
    for seed in seeds:
        popularity = StaticPolicy.fit(warm_up_history(world, warm_up_rounds, seed)).popularity
        for name, spec in settings.policies.items():
            store = InMemoryOptionsStore()
            policy = build_policy(name, spec, store, seed=seed, popularity=popularity)
            record_beliefs = seed == seeds[0] and policy.learns

            def snapshot(round_: int, name: str = name, store=store) -> None:
                snapshots.extend(
                    {
                        "policy": name,
                        "round": round_,
                        "context_key": row.context_key,
                        "offer_id": row.offer_id,
                        "alpha": row.stats.alpha,
                        "beta": row.stats.beta,
                    }
                    for row in store.rows(name)
                )

            log = simulate(
                policy,
                world,
                rounds=rounds,
                shift_round=shift_round,
                n_offers=n_offers,
                reward=settings.reward_function(name, world.offers),
                seed=seed,
                snapshot=snapshot if record_beliefs else None,
                snapshot_every=snapshot_every,
            )
            summaries.append({"policy": name, "seed": seed} | summarize(log))
            if seed == seeds[0]:
                round_logs.append(log.assign(policy=name))
            progress(f"  seed {seed}  {name:<26} regret {summaries[-1]['regret']:8.1f}")
    return Comparison(
        summary=pd.DataFrame(summaries),
        rounds=pd.concat(round_logs, ignore_index=True),
        snapshots=pd.DataFrame(snapshots),
    )


HEADLINE = {
    "take_up_rate": "Take-up per app open",
    "take_up_before_shift": "Before shift",
    "take_up_after_shift": "After shift",
    "revenue_per_open": "Revenue per open (R)",
    "regret": "Regret (take-ups lost)",
    "regret_after_shift": "Regret after shift",
    "share_of_oracle": "Share of oracle",
}


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per policy, best first, with regret reduction against the baselines."""
    table = summary.groupby("policy")[list(HEADLINE)].mean().sort_values("regret")
    for baseline in ("random", "static"):
        if baseline in table.index:
            table[f"regret_vs_{baseline}"] = 1 - table["regret"] / table.loc[baseline, "regret"]
    spread = summary.groupby("policy")["regret"].std()
    table.insert(table.columns.get_loc("regret") + 1, "regret_sd", spread)
    return table


def moment_favourites(truth, world: World) -> dict[tuple[str, bool], str]:
    """The right first offer for each (segment, payday week) under ``truth``."""
    table = take_up_table(
        truth, world.customers, world.offers, world.days, by=("segment", "is_payday_week")
    )
    return {key: row.idxmax() for key, row in table.iterrows()}


def convergence_table(rounds: pd.DataFrame, world: World, shift_round: int | None) -> pd.DataFrame:
    """App opens each policy needed to rank each segment's best offer first, from the start
    and again after the behaviour shift (NaN: not within the run)."""
    before = moment_favourites(world.truth, world)
    after = moment_favourites(world.truth.after_shift(), world) if shift_round else None
    rows = []
    for policy, log in rounds.groupby("policy", sort=False):
        log = log.reset_index(drop=True)
        phases = [("from the start", before, 0, shift_round)]
        if after is not None:
            phases.append(("after the shift", after, shift_round, None))
        for phase, favourites, start, end in phases:
            reached = convergence(log, favourites, start_round=start, end_round=end)
            rows.extend(
                {"policy": policy, "phase": phase, "segment": segment, "app_opens": opens}
                for segment, opens in reached.items()
            )
    return pd.DataFrame(rows)


def save_results(
    comparison: Comparison, world: World, shift_round: int | None, out: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the files the dashboard reads; returns the headline and convergence tables."""
    out.mkdir(parents=True, exist_ok=True)
    table = comparison_table(comparison.summary)
    convergence = convergence_table(comparison.rounds, world, shift_round)
    comparison.summary.to_csv(out / "comparison_by_seed.csv", index=False)
    table.to_csv(out / "comparison.csv")
    comparison.rounds.to_parquet(out / "comparison_rounds.parquet", index=False)
    comparison.snapshots.to_parquet(out / "comparison_snapshots.parquet", index=False)
    convergence.to_csv(out / "convergence.csv", index=False)
    return table, convergence
