"""Does forgetting help after the behaviour shift? Reproduces decision P9.

Runs contextual Thompson sampling with several decay rates through the same simulated
world (3 offers per app open, behaviour shift halfway), over several seeds, and reports
the mean and standard deviation of each measure.

Usage:  python scripts/experiment_forgetting.py [--rounds 50000] [--seeds 5]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from recommender.algorithms.thompson import ThompsonSamplingPolicy
from recommender.config import PROJECT_ROOT
from recommender.simulator.engine import World, simulate
from recommender.store.options_store import InMemoryOptionsStore

DECAYS = (1.0, 0.9998, 0.9995, 0.999)
# After the shift, the right answer for these segments changes.
NEW_FAVOURITES = {"family": "electricity_cashback", "young_professional": "streaming_bundle"}


def measure(log: pd.DataFrame) -> dict[str, float]:
    """Take-up before and after the shift, and how often the new favourites led."""
    before, after = log[~log["shifted"]], log[log["shifted"]]
    metrics = {
        "take_up_before": before["take_ups"].mean(),
        "take_up_after": after["take_ups"].mean(),
    }
    for segment, favourite in NEW_FAVOURITES.items():
        opens = after[after["segment"] == segment]
        metrics[f"{segment}_new_favourite_first"] = (opens["top_offer"] == favourite).mean()
    return metrics


def main() -> None:
    """Run each decay rate over several seeds and print the mean and spread."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    world = World.load()
    rows = []
    for decay in DECAYS:
        for seed in range(args.seeds):
            policy = ThompsonSamplingPolicy(
                "thompson_segment",
                InMemoryOptionsStore(),
                rng=np.random.default_rng(seed),
                context_mode="segment",
                decay=decay,
            )
            log = simulate(
                policy, world, rounds=args.rounds, shift_round=args.rounds // 2, seed=seed
            )
            rows.append({"decay": decay, "seed": seed, **measure(log)})
            print(f"  decay {decay:<7g} seed {seed} done", flush=True)

    results = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "experiment_forgetting.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)

    summary = results.drop(columns="seed").groupby("decay", sort=False).agg(["mean", "std"])
    table = pd.DataFrame(
        {
            metric: [f"{m:.3f} (sd {s:.3f})" for m, s in summary[metric].itertuples(index=False)]
            for metric in summary.columns.get_level_values(0).unique()
        },
        index=[f"{d:g}" for d in summary.index],
    )
    print(f"\n{args.rounds:,} app opens, shift at {args.rounds // 2:,}, {args.seeds} seeds")
    print(table.rename_axis("decay").to_string())
    print(f"\nPer-seed results: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
