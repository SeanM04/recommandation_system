"""The simulation loop: app opens, ranked offers, simulated responses, learning.

Two ways to drive the same loop:

* ``simulate`` (fast mode): the policy is called directly, which is what large
  comparisons need.
* ``simulate_via_api``: every app open is a real ``POST /invocations`` and every outcome
  a real ``POST /response``, exactly as the app would do it.

Every round is scored against an oracle that knows the true probabilities. Regret uses
*expected* values from the response model rather than the single random draw, so the
curves measure the policy's choices, not luck.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from recommender.algorithms.base import Policy
from recommender.algorithms.rewards import binary_reward
from recommender.config import load_offers, load_settings
from recommender.domain import TAKEN_UP, Context, Customer, Offer
from recommender.simulator.ground_truth import GroundTruth
from recommender.simulator.population import generate_customers, sample_context
from recommender.simulator.settings import load_simulation_settings

RewardFn = Callable[[Offer, str], float | None]
Snapshot = Callable[[int], None]


@dataclass(frozen=True)
class World:
    """Everything a simulation needs: the hidden truth, the offers, who opens the app, when."""

    truth: GroundTruth
    offers: list[Offer]
    customers: list[Customer]
    days: list[date]
    offers_by_id: dict[str, Offer] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "offers_by_id", {o.offer_id: o for o in self.offers})

    @classmethod
    def load(cls) -> World:
        """The configured world. Customers are regenerated from the seed, which gives the
        same population as ``data/customers.csv`` and the customers table."""
        settings, simulation = load_settings(), load_simulation_settings()
        offers = load_offers()
        population = simulation.population
        frame = generate_customers(population.n_customers, population.segment_mix, settings.seed)
        return cls(
            truth=GroundTruth.from_yaml(simulation.ground_truth, offers),
            offers=offers,
            customers=[Customer(**record) for record in frame.to_dict("records")],
            days=simulation.calendar.days(),
        )

    def app_open(self, rng: np.random.Generator, round_: int, rounds: int) -> Context:
        """Who opens the app in this round, and when. Rounds spread evenly over the calendar."""
        customer = self.customers[rng.integers(len(self.customers))]
        return sample_context(customer, rng, self.days[round_ * len(self.days) // rounds])

    def eligible(self, segment: str) -> list[Offer]:
        return [offer for offer in self.offers if offer.is_eligible(segment)]


def random_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Separate streams for arrivals and outcomes. Every policy run with the same seed
    therefore faces the identical sequence of app opens (common random numbers)."""
    return np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])


def score_round(
    world: World,
    truth: GroundTruth,
    context: Context,
    probabilities: dict[str, float],
    shown: Sequence[str],
    outcomes: dict[str, str],
) -> dict[str, Any]:
    """What happened in one round, and what the oracle would have expected.

    The take-up oracle ranks eligible offers by true probability; the revenue oracle by
    true probability x margin. Both are scored with the same response model.
    """
    response = truth.response
    margins = {offer_id: world.offers_by_id[offer_id].margin_zar for offer_id in probabilities}
    n = len(shown)
    by_take_up = sorted(probabilities, key=probabilities.get, reverse=True)[:n]
    by_revenue = sorted(probabilities, key=lambda o: probabilities[o] * margins[o], reverse=True)[
        :n
    ]
    taken = [offer_id for offer_id, outcome in outcomes.items() if outcome == TAKEN_UP]
    return {
        "segment": context.segment,
        "is_payday_week": context.is_payday_week,
        "top_offer": shown[0],
        "take_ups": len(taken),
        "revenue": sum(margins[offer_id] for offer_id in taken),
        "expected_take_ups": response.expected_value([probabilities[o] for o in shown]),
        "oracle_take_ups": response.expected_value([probabilities[o] for o in by_take_up]),
        "expected_revenue": response.expected_value(
            [probabilities[o] for o in shown], [margins[o] for o in shown]
        ),
        "oracle_revenue": response.expected_value(
            [probabilities[o] for o in by_revenue], [margins[o] for o in by_revenue]
        ),
    }


def simulate(
    policy: Policy,
    world: World,
    *,
    rounds: int,
    shift_round: int | None = None,
    n_offers: int = 3,
    reward: RewardFn = binary_reward,
    seed: int = 0,
    snapshot: Snapshot | None = None,
    snapshot_every: int = 0,
) -> pd.DataFrame:
    """Fast mode: run ``rounds`` app opens through ``policy`` directly. One row per round.

    From ``shift_round`` on, customers behave according to the shifted truth.
    ``snapshot(round)`` is called every ``snapshot_every`` rounds (e.g. to record beliefs).
    """
    arrivals, outcomes_rng = random_streams(seed)
    shifted_truth = world.truth.after_shift()
    records = []
    for round_ in range(rounds):
        shifted = shift_round is not None and round_ >= shift_round
        truth = shifted_truth if shifted else world.truth
        context = world.app_open(arrivals, round_, rounds)
        eligible = world.eligible(context.segment)
        probabilities = truth.probabilities(context, eligible)

        shown = [scored.offer_id for scored in policy.rank(context, eligible, n_offers)]
        drawn = truth.response.draw([probabilities[o] for o in shown], outcomes_rng)
        outcomes = dict(zip(shown, drawn, strict=True))
        for offer_id, outcome in outcomes.items():
            value = reward(world.offers_by_id[offer_id], outcome)
            if value is not None:
                policy.update(context, offer_id, value)

        records.append(
            {"round": round_, "day": context_day(world, round_, rounds), "shifted": shifted}
            | score_round(world, truth, context, probabilities, shown, outcomes)
        )
        if snapshot is not None and snapshot_every and (round_ + 1) % snapshot_every == 0:
            snapshot(round_ + 1)
    return pd.DataFrame.from_records(records)


def context_day(world: World, round_: int, rounds: int) -> date:
    return world.days[round_ * len(world.days) // rounds]


def simulate_via_api(
    client: Any,
    world: World,
    *,
    rounds: int,
    shift_round: int | None = None,
    n_offers: int = 3,
    seed: int = 0,
    policy: str | None = None,
    base_url: str = "",
    progress: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """API mode: each app open is ``POST /invocations``, each outcome ``POST /response``.

    ``client`` is anything with ``post(url, json=..., params=...)`` returning a response with
    ``raise_for_status()`` and ``json()``: a ``requests.Session``, or FastAPI's TestClient.
    The simulated moment (day, time of day) is sent as a context override, so the API sees
    the simulated calendar rather than today's date.
    """
    arrivals, outcomes_rng = random_streams(seed)
    shifted_truth = world.truth.after_shift()
    params = {"policy": policy} if policy else None
    records = []
    for round_ in range(rounds):
        shifted = shift_round is not None and round_ >= shift_round
        truth = shifted_truth if shifted else world.truth
        context = world.app_open(arrivals, round_, rounds)
        day = context_day(world, round_, rounds)
        body = {
            "customer_id": context.customer_id,
            "channel": context.channel,
            "number_of_offers": n_offers,
            "context": {"day": day.isoformat(), "time_of_day": context.time_of_day},
        }
        reply = client.post(f"{base_url}/invocations", json=body, params=params)
        reply.raise_for_status()
        invocation = reply.json()

        eligible = world.eligible(context.segment)
        probabilities = truth.probabilities(context, eligible)
        shown = [offer["offer_id"] for offer in invocation["offers"]]
        drawn = truth.response.draw([probabilities[o] for o in shown], outcomes_rng)
        outcomes = dict(zip(shown, drawn, strict=True))
        for offer_id, outcome in outcomes.items():
            payload = {
                "invocation_id": invocation["invocation_id"],
                "offer_id": offer_id,
                "outcome": outcome,
            }
            client.post(f"{base_url}/response", json=payload).raise_for_status()

        records.append(
            {"round": round_, "day": day, "shifted": shifted}
            | score_round(world, truth, context, probabilities, shown, outcomes)
        )
        if progress is not None:
            progress(round_ + 1)
    return pd.DataFrame.from_records(records)
