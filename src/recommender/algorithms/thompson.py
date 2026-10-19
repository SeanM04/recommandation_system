"""Thompson sampling."""

from __future__ import annotations

from collections.abc import Sequence

from recommender.algorithms.base import BanditPolicy, ScoredOffer, check_slate_size
from recommender.domain import Context, Offer


class ThompsonSamplingPolicy(BanditPolicy):
    """Keep a Beta(alpha, beta) belief about each offer's take-up rate, draw one
    plausible rate from every belief, and rank the offers by their draws.

    An offer we know little about has a wide belief, so it sometimes draws high and
    gets shown: exploration happens in proportion to uncertainty, with nothing to tune.
    """

    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        """Draw a take-up rate from each offer's Beta belief and show the highest draws."""
        check_slate_size(n)
        stats = self.stats(context, candidates)
        draws = {
            offer_id: float(self.rng.beta(arm.alpha, arm.beta)) for offer_id, arm in stats.items()
        }
        best = sorted(draws, key=draws.__getitem__, reverse=True)[:n]
        return [ScoredOffer(offer_id, draws[offer_id]) for offer_id in best]
