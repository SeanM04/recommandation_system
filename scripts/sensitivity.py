"""Rerun the key policies in six variants of the simulated world (a sensitivity sweep).

Each scenario changes one assumption: no behaviour shift, weaker or stronger context
effects, flatter preferences, or an earlier shift. Writes results/sensitivity.csv and
docs/data/sensitivity.json, which the Results page is built from.

Usage:  python scripts/sensitivity.py [--rounds 30000] [--seeds 3] [--no-docs]
                                                                        (make sensitivity)
"""

from __future__ import annotations

import argparse
import json
import time

from recommender.config import PROJECT_ROOT, load_settings
from recommender.evaluation.sensitivity import SCENARIOS, run_sweep
from recommender.simulator.engine import World

POLICIES = (
    "random",
    "static",
    "epsilon_greedy_segment",
    "thompson",
    "thompson_segment",
    "thompson_segment_decay",
    "thompson_features",
)


def main() -> None:
    """Run the sweep, print share of oracle per scenario and policy, and save both files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=30_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-docs", action="store_true", help="leave docs/data untouched")
    args = parser.parse_args()

    settings = load_settings()
    policies = [p for p in POLICIES if p in settings.policies]
    started = time.monotonic()
    rows = run_sweep(settings, World.load(), policies, rounds=args.rounds, seeds=range(args.seeds))
    print(f"done in {time.monotonic() - started:.0f}s\n")
    table = rows.pivot_table(index="policy", columns="scenario", values="share_of_oracle")
    table = table.reindex(columns=[s.name for s in SCENARIOS]).sort_values("As published")
    print(table.map("{:.1%}".format).to_string())

    out = PROJECT_ROOT / "results"
    out.mkdir(exist_ok=True)
    rows.to_csv(out / "sensitivity.csv", index=False)
    if args.no_docs:
        print("\nSaved results/sensitivity.csv (docs/data left untouched)")
        return
    payload = {
        "rounds": args.rounds,
        "seeds": args.seeds,
        "scenarios": [{"name": s.name, "change": s.change} for s in SCENARIOS],
        "by_seed": rows.to_dict("records"),
    }
    path = PROJECT_ROOT / "docs" / "data" / "sensitivity.json"
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("\nSaved results/sensitivity.csv and docs/data/sensitivity.json")


if __name__ == "__main__":
    main()
