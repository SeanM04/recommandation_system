"""Random baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from recommender.algorithms.base import Policy, ScoredOffer, check_slate_size
from recommender.domain import Context, Offer


class RandomPolicy(Policy):
    """The floor every other policy must beat: offers in random order. Learns nothing."""

    def __init__(self, name: str = "random", *, rng: np.random.Generator) -> None:
        self.name = name
        self.rng = rng

    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        check_slate_size(n)
        draws = self.rng.random(len(candidates))
        order = np.argsort(-draws)[:n]
        return [ScoredOffer(candidates[i].offer_id, float(draws[i])) for i in order]
