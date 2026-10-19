"""How wide should the feature model's posterior be? Reproduces the choice of ``scale``.

Thompson sampling on customer features explores by drawing weights from a Gaussian whose
width is ``scale`` times the posterior's. Too narrow and it stops testing offers; too wide
and it keeps showing poor ones. This runs each width through the same simulated world on
seeds *outside* the ten the published results use (100 and up), so the choice can't
flatter those results.

Usage:  python scripts/experiment_features.py [--seeds 1] [--rounds 50000]
                                                                (make experiment-features)
"""

from __future__ import annotations

import argparse

import pandas as pd

from recommender.algorithms.factory import policy_rng
from recommender.algorithms.linear_thompson import LinearThompsonPolicy
from recommender.config import PROJECT_ROOT, load_settings
from recommender.evaluation.metrics import summarize
from recommender.simulator.engine import World, simulate

SCALES = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4)
FIRST_SEED = 100  # the published comparison uses seeds 0-9


def main() -> None:
    """Run every width on the held-out seeds and print share of oracle and regret."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=50_000)
    args = parser.parse_args()

    settings, world = load_settings(), World.load()
    spec = settings.policies["thompson_features"]
    reward = settings.reward_function("thompson_features", world.offers)
    rows = []
    for scale in SCALES:
        for seed in range(FIRST_SEED, FIRST_SEED + args.seeds):
            policy = LinearThompsonPolicy(
                "thompson_features",
                rng=policy_rng(seed, "thompson_features"),
                decay=spec.decay,
                scale=scale,
            )
            log = simulate(
                policy,
                world,
                rounds=args.rounds,
                shift_round=args.rounds // 2,
                n_offers=settings.recommendation.number_of_offers,
                reward=reward,
                seed=seed,
            )
            rows.append({"scale": scale, "seed": seed} | summarize(log))
            print(
                f"  scale {scale:<6} seed {seed}: share of oracle {rows[-1]['share_of_oracle']:.3f}"
            )

    table = (
        pd.DataFrame(rows)
        .groupby("scale")[["share_of_oracle", "regret", "regret_after_shift"]]
        .mean()
    )
    print(f"\nMean over {args.seeds} held-out seed(s); configured scale: {spec.scale}\n")
    print(table.round(3).to_string())
    out = PROJECT_ROOT / "results" / "experiment_features.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out)
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
