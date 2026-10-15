"""Reward functions: turn the outcome of one shown offer into what a policy learns from.

A reward function returns a number in [0, 1], or ``None`` for "no update". By default
``ignored`` gives ``None``: an offer the customer never noticed says nothing about how
appealing it is, so counting it as a failure would punish offers that were ranked low.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from recommender.domain import IGNORED, OUTCOMES, TAKEN_UP, VIEWED, Offer

RewardFn = Callable[[Offer, str], float | None]
REWARDS = ("binary", "margin")


def binary_reward(offer: Offer, outcome: str) -> float | None:
    """1 for a take-up, 0 for an offer seen and passed over."""
    return {TAKEN_UP: 1.0, VIEWED: 0.0}.get(outcome)


def margin_reward(offers: Sequence[Offer]) -> RewardFn:
    """Revenue-aware: a take-up earns its margin, scaled by the catalogue's top margin."""
    top_margin = max(offer.margin_zar for offer in offers)

    def reward(offer: Offer, outcome: str) -> float | None:
        if outcome == TAKEN_UP:
            return offer.margin_zar / top_margin
        return 0.0 if outcome == VIEWED else None

    return reward


def make_reward(name: str, offers: Sequence[Offer], learn_from_ignored: bool = False) -> RewardFn:
    """Build the configured reward function (``binary`` or ``margin``)."""
    if name not in REWARDS:
        raise ValueError(f"unknown reward {name!r}; expected one of {REWARDS}")
    base = binary_reward if name == "binary" else margin_reward(offers)

    def reward(offer: Offer, outcome: str) -> float | None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
        if outcome == IGNORED and learn_from_ignored:
            return 0.0
        return base(offer, outcome)

    return reward
