"""The simulation loop: app opens, ranked offers, simulated responses, learning.

This is fast mode: the policy is called directly, without HTTP, which is what large
experiments need. Phase 4 adds the same loop driven through the API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from recommender.algorithms.base import Policy
from recommender.algorithms.rewards import binary_reward
from recommender.config import load_offers, load_settings
from recommender.domain import TAKEN_UP, Customer, Offer
from recommender.simulator.ground_truth import GroundTruth
from recommender.simulator.population import generate_customers, sample_context
from recommender.simulator.settings import load_simulation_settings


@dataclass(frozen=True)
class World:
    """Everything a simulation needs: the hidden truth, the offers, who opens the app, when."""

    truth: GroundTruth
    offers: list[Offer]
    customers: list[Customer]
    days: list[date]

    @classmethod
    def load(cls) -> World:
        """The configured world. Customers are regenerated from the seed, which gives the
        same population as ``data/customers.csv``."""
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


def simulate(
    policy: Policy,
    world: World,
    *,
    rounds: int,
    shift_round: int | None = None,
    n_offers: int = 3,
    reward: Callable[[Offer, str], float | None] = binary_reward,
    seed: int = 0,
) -> pd.DataFrame:
    """Run ``rounds`` app opens and return one row per app open.

    Rounds are spread evenly over the simulated calendar. From ``shift_round`` on, the
    customers behave according to the shifted truth.
    """
    rng = np.random.default_rng(seed)
    offers_by_id = {offer.offer_id: offer for offer in world.offers}
    shifted_truth = world.truth.after_shift()
    records = []
    for round_ in range(rounds):
        shifted = shift_round is not None and round_ >= shift_round
        truth = shifted_truth if shifted else world.truth
        day = world.days[round_ * len(world.days) // rounds]
        customer = world.customers[rng.integers(len(world.customers))]
        context = sample_context(customer, rng, day)

        eligible = [offer for offer in world.offers if offer.is_eligible(context.segment)]
        ranked = policy.rank(context, eligible, n_offers)
        outcomes = truth.respond(context, [offers_by_id[s.offer_id] for s in ranked], rng)
        for offer_id, outcome in outcomes.items():
            value = reward(offers_by_id[offer_id], outcome)
            if value is not None:
                policy.update(context, offer_id, value)

        records.append(
            {
                "round": round_,
                "day": day,
                "segment": context.segment,
                "is_payday_week": context.is_payday_week,
                "shifted": shifted,
                "top_offer": ranked[0].offer_id,
                "take_ups": sum(outcome == TAKEN_UP for outcome in outcomes.values()),
            }
        )
    return pd.DataFrame.from_records(records)
