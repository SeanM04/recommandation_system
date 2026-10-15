"""Compare every configured policy on the same simulated world and save the results.

Usage:  python scripts/compare_algorithms.py [--rounds 50000] [--seeds 10]   (make compare)

Writes to results/:
    comparison.csv / comparison.md   one row per policy, mean over seeds (the headline table)
    comparison_by_seed.csv           one row per policy and seed
    comparison_rounds.parquet        every app open of the first seed (for the dashboard)
    comparison_snapshots.parquet     beliefs over time of the learning policies
    convergence.csv                  app opens to rank each segment's best offer first
and docs/data/comparison.json, which the docs and README tables are built from.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from recommender.config import PROJECT_ROOT, load_settings
from recommender.evaluation.compare import (
    HEADLINE,
    run_comparison,
    save_results,
)
from recommender.evaluation.metrics import cumulative
from recommender.simulator.engine import World


def to_markdown(table: pd.DataFrame) -> str:
    """The headline table as GitHub Markdown."""
    columns = ["Policy", *HEADLINE.values(), "Regret cut vs random", "Regret cut vs static"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for policy, row in table.iterrows():
        cells = [
            f"`{policy}`",
            f"{row.take_up_rate:.3f}",
            f"{row.take_up_before_shift:.3f}",
            f"{row.take_up_after_shift:.3f}",
            f"R{row.revenue_per_open:.2f}",
            f"{row.regret:,.0f} ± {row.regret_sd:,.0f}",
            f"{row.regret_after_shift:,.0f}",
            f"{row.share_of_oracle:.1%}",
            f"{row.regret_vs_random:.0%}" if "regret_vs_random" in row else "",
            f"{row.regret_vs_static:.0%}" if "regret_vs_static" in row else "",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--warm-up", type=int, default=5000, help="static baseline's history")
    parser.add_argument("--no-shift", action="store_true", help="keep behaviour stable")
    parser.add_argument("--no-docs", action="store_true", help="leave docs/data untouched")
    args = parser.parse_args()

    settings = load_settings()
    world = World.load()
    shift_round = None if args.no_shift else args.rounds // 2
    print(f"{len(settings.policies)} policies x {args.seeds} seeds x {args.rounds:,} app opens")
    start = time.perf_counter()
    comparison = run_comparison(
        settings,
        world,
        rounds=args.rounds,
        shift_round=shift_round,
        seeds=range(args.seeds),
        warm_up_rounds=args.warm_up,
    )
    print(f"done in {time.perf_counter() - start:.0f}s\n")

    out = PROJECT_ROOT / "results"
    table, convergence = save_results(comparison, world, shift_round, out)
    markdown = to_markdown(table)
    (out / "comparison.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    if args.no_docs:
        print("\nSaved results/ (docs/data left untouched)")
        return

    curves = {
        policy: cumulative(log, every=500)[["round", "regret"]].round(2).to_dict("list")
        for policy, log in comparison.rounds.groupby("policy")
    }
    docs_data = PROJECT_ROOT / "docs" / "data"
    docs_data.mkdir(parents=True, exist_ok=True)
    payload = {
        "rounds": args.rounds,
        "seeds": args.seeds,
        "shift_round": shift_round,
        "warm_up_rounds": args.warm_up,
        "table": table.reset_index().round(6).to_dict("records"),
        "by_seed": comparison.summary.round(6).to_dict("records"),  # for paired comparisons
        "regret_curves": curves,
        "convergence": convergence.to_dict("records"),
    }
    (docs_data / "comparison.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\nSaved results/ and docs/data/comparison.json")


if __name__ == "__main__":
    main()
