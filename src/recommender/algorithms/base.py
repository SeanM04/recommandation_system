"""The policy interface, context keys, and the learning step shared by bandit policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from recommender.domain import CONTEXT_MODES, Context, Offer
from recommender.store.options_store import ArmStats, OptionsStore


@dataclass(frozen=True)
class ScoredOffer:
    """One ranked offer and the score the policy ranked it by."""

    offer_id: str
    score: float


def context_key(context: Context, mode: str) -> str:
    """Whose experience a policy pools for this app open.

    ``global`` -> ``"all"``, ``segment`` -> ``"family"``,
    ``segment_payday`` -> ``"family|payday"`` or ``"family|rest"``.
    """
    if mode == "global":
        return "all"
    if mode == "segment":
        return context.segment
    if mode == "segment_payday":
        return f"{context.segment}|{'payday' if context.is_payday_week else 'rest'}"
    raise ValueError(f"unknown context mode {mode!r}; expected one of {CONTEXT_MODES}")


def check_slate_size(n: int) -> None:
    """Refuse a request for fewer than one offer."""
    if n < 1:
        raise ValueError(f"number of offers must be at least 1, got {n}")


class Policy(ABC):
    """Ranks offers for an app open, and may learn from the rewards that follow."""

    name: str
    learns = False  # True for policies whose update() changes what they recommend
    servable = True  # False for policies whose state the options store can't hold

    @abstractmethod
    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        """Return up to ``n`` of the ``candidates``, best first."""

    def update(self, context: Context, offer_id: str, reward: float) -> None:  # noqa: B027
        """Learn from one reward in [0, 1]. Baselines that never learn ignore it."""


class BanditPolicy(Policy):
    """A policy that learns, keeping its statistics in the options store.

    ``context_mode`` sets whose experience is pooled (see ``context_key``). ``decay`` < 1
    switches on forgetting: before each update, every offer's evidence in that context
    shrinks by ``decay``, so recent responses outweigh old ones.
    """

    learns = True

    def __init__(
        self,
        name: str,
        store: OptionsStore,
        *,
        rng: np.random.Generator,
        context_mode: str = "global",
        decay: float = 1.0,
    ) -> None:
        """A learning policy reading and writing its statistics in ``store``."""
        if context_mode not in CONTEXT_MODES:
            raise ValueError(
                f"unknown context mode {context_mode!r}; expected one of {CONTEXT_MODES}"
            )
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.name = name
        self.store = store
        self.rng = rng
        self.context_mode = context_mode
        self.decay = decay

    def key(self, context: Context) -> str:
        """The context key this app open's evidence is pooled under."""
        return context_key(context, self.context_mode)

    def stats(self, context: Context, candidates: Sequence[Offer]) -> dict[str, ArmStats]:
        """Current statistics for each candidate; untried offers start at the prior."""
        return self.store.get(self.name, self.key(context), [o.offer_id for o in candidates])

    def update(self, context: Context, offer_id: str, reward: float) -> None:
        """Forget a little (if decay is on), then add the new evidence.

        The store does both in one atomic step, so concurrent requests never overwrite
        each other's learning.
        """
        if not 0 <= reward <= 1:
            raise ValueError(f"reward must be in [0, 1], got {reward}")
        self.store.learn(self.name, self.key(context), offer_id, reward, decay=self.decay)
