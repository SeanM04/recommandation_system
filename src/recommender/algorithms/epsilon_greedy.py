"""Epsilon-greedy."""

from __future__ import annotations

from collections.abc import Sequence

from recommender.algorithms.base import BanditPolicy, ScoredOffer, check_slate_size
from recommender.domain import Context, Offer
from recommender.store.options_store import ArmStats, OptionsStore

UNTRIED_SCORE = 1.0  # an offer never shown looks perfect, so every offer gets tried early


class EpsilonGreedyPolicy(BanditPolicy):
    """Fill each slot with the offer that has the best average reward so far, except
    that with probability epsilon the slot goes to a random remaining offer instead.

    With ``epsilon_decay`` < 1, epsilon shrinks as evidence builds up in a context:
    epsilon x epsilon_decay ** (rewards seen), never below ``min_epsilon``.
    """

    def __init__(
        self,
        name: str,
        store: OptionsStore,
        *,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.0,
        **kwargs,
    ) -> None:
        """An epsilon-greedy policy; ``epsilon`` is the share of slots it explores."""
        super().__init__(name, store, **kwargs)
        if not 0 <= min_epsilon <= epsilon <= 1:
            raise ValueError("need 0 <= min_epsilon <= epsilon <= 1")
        if not 0 < epsilon_decay <= 1:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}")
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def current_epsilon(self, stats: dict[str, ArmStats]) -> float:
        """The exploration rate now: epsilon, shrunk by the evidence so far if decay is set."""
        evidence = sum(arm.pulls for arm in stats.values())
        return max(self.min_epsilon, self.epsilon * self.epsilon_decay**evidence)

    @staticmethod
    def estimate(arm: ArmStats) -> float:
        """An offer's average reward so far; one never shown scores 1.0 so it gets tried."""
        return arm.total_reward / arm.pulls if arm.pulls > 0 else UNTRIED_SCORE

    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        """Fill each slot with the best remaining average, or a random offer epsilon of the time."""
        check_slate_size(n)
        stats = self.stats(context, candidates)
        epsilon = self.current_epsilon(stats)
        estimates = {offer_id: self.estimate(arm) for offer_id, arm in stats.items()}

        # Shuffle, then sort best first: offers with equal estimates end up in random order.
        offer_ids = list(estimates)
        shuffled = [offer_ids[i] for i in self.rng.permutation(len(offer_ids))]
        remaining = sorted(shuffled, key=estimates.__getitem__, reverse=True)

        ranked: list[ScoredOffer] = []
        while remaining and len(ranked) < n:
            if self.rng.random() < epsilon:
                pick = remaining[self.rng.integers(len(remaining))]  # explore
            else:
                pick = remaining[0]  # exploit
            remaining.remove(pick)
            ranked.append(ScoredOffer(pick, estimates[pick]))
        return ranked
