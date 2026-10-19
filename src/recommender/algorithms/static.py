"""Static, batch-trained baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from recommender.algorithms.base import Policy, ScoredOffer, check_slate_size
from recommender.domain import Context, Offer


class StaticPolicy(Policy):
    """The batch-model baseline: rank by one popularity table, computed offline from
    historical responses and then frozen.

    Every customer sees the same order, and new responses change nothing until the next
    batch retrain. With an empty table the order is the catalogue order.
    """

    def __init__(self, popularity: Mapping[str, float] | None = None, name: str = "static") -> None:
        """A frozen ranking from ``popularity``: average reward per offer."""
        self.name = name
        self.popularity = dict(popularity or {})

    @classmethod
    def fit(cls, history: Iterable[tuple[str, float]], name: str = "static") -> StaticPolicy:
        """The batch training job: average reward per offer over a log of (offer_id, reward)."""
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for offer_id, reward in history:
            totals[offer_id] += reward
            counts[offer_id] += 1
        return cls({offer_id: totals[offer_id] / counts[offer_id] for offer_id in counts}, name)

    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        """The eligible offers by popularity, the same order for every customer."""
        check_slate_size(n)
        scores = {offer.offer_id: self.popularity.get(offer.offer_id, 0.0) for offer in candidates}
        best = sorted(scores, key=scores.__getitem__, reverse=True)[
            :n
        ]  # stable: ties keep catalogue order
        return [ScoredOffer(offer_id, scores[offer_id]) for offer_id in best]
