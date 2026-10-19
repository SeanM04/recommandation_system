"""Does the ranking of policies survive a different world? A sensitivity sweep.

The published results come from one simulated world that this project wrote, so the fair
challenge is "you built the world your method wins in". Each scenario here changes one
assumption of that world (how much context matters, how distinct preferences are, whether
and when behaviour shifts) and reruns the same policies on the same seeds. If the
conclusions only held in the original world, this is where it would show.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import pandas as pd

from recommender.config import Settings
from recommender.evaluation.compare import run_comparison
from recommender.simulator.engine import World
from recommender.simulator.ground_truth import GroundTruth


def scale_effects(truth: GroundTruth, strength: float) -> GroundTruth:
    """Every context effect ``strength`` times as strong: a multiplier m becomes
    1 + strength * (m - 1), never below zero. 0 removes context; 1 changes nothing."""
    effects = tuple(
        replace(
            effect,
            multiply={k: max(0.0, 1 + strength * (m - 1)) for k, m in effect.multiply.items()},
        )
        for effect in truth.effects
    )
    return replace(truth, effects=effects)


def flatten_preferences(truth: GroundTruth, spread: float) -> GroundTruth:
    """Pull each segment's take-up rates towards that segment's average: a rate p becomes
    mean + spread * (p - mean). Below 1 the best offer stands out less and is harder to
    find. The shift's new rates move the same way, around the same averages."""
    means = {s: sum(rates.values()) / len(rates) for s, rates in truth.base_take_up.items()}

    def pull(segment: str, rates: dict[str, float]) -> dict[str, float]:
        """One segment's rates, pulled towards its average."""
        mean = means[segment]
        return {offer: mean + spread * (p - mean) for offer, p in rates.items()}

    return replace(
        truth,
        base_take_up={s: pull(s, dict(r)) for s, r in truth.base_take_up.items()},
        shift_base_take_up={s: pull(s, dict(r)) for s, r in truth.shift_base_take_up.items()},
    )


@dataclass(frozen=True)
class Scenario:
    """One variant of the world: what it changes, and when (if ever) behaviour shifts."""

    name: str
    change: str
    truth: Callable[[GroundTruth], GroundTruth]
    shift_at: float | None = 0.5  # share of the run; None: no shift


SCENARIOS = (
    Scenario("As published", "the world the results use", lambda t: t),
    Scenario("No behaviour shift", "customers never change", lambda t: t, shift_at=None),
    Scenario("Weaker context", "context effects at half strength", lambda t: scale_effects(t, 0.5)),
    Scenario(
        "Stronger context", "context effects at double strength", lambda t: scale_effects(t, 2.0)
    ),
    Scenario(
        "Flatter preferences",
        "offers' take-up rates half as far apart",
        lambda t: flatten_preferences(t, 0.5),
    ),
    Scenario("Earlier shift", "behaviour shifts after 20% of the run", lambda t: t, shift_at=0.2),
)


def run_sweep(
    settings: Settings,
    world: World,
    policies: Sequence[str],
    *,
    rounds: int,
    seeds: Sequence[int],
    scenarios: Sequence[Scenario] = SCENARIOS,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Every scenario, every chosen policy, every seed: one row each, with the policy's
    share of what was possible, its regret and its take-up rate."""
    chosen = settings.model_copy(
        update={"policies": {n: s for n, s in settings.policies.items() if n in policies}}
    )
    rows = []
    for scenario in scenarios:
        progress(f"{scenario.name}: {scenario.change}")
        variant = replace(world, truth=scenario.truth(world.truth))
        shift_round = int(rounds * scenario.shift_at) if scenario.shift_at else None
        comparison = run_comparison(
            chosen,
            variant,
            rounds=rounds,
            shift_round=shift_round,
            seeds=seeds,
            snapshot_every=rounds + 1,  # beliefs aren't needed here
            progress=lambda _: None,
        )
        for row in comparison.summary.to_dict("records"):
            rows.append(
                {
                    "scenario": scenario.name,
                    "change": scenario.change,
                    "policy": row["policy"],
                    "seed": row["seed"],
                    "share_of_oracle": row["share_of_oracle"],
                    "regret": row["regret"],
                    "take_up_rate": row["take_up_rate"],
                }
            )
    return pd.DataFrame(rows)
