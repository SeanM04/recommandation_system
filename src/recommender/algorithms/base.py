"""The policy interface, context keys, and the learning step shared by bandit policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from recommender.domain import Context, Offer
from recommender.store.options_store import ArmStats, OptionsStore

CONTEXT_MODES = ("global", "segment", "segment_payday")


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
    if n < 1:
        raise ValueError(f"number of offers must be at least 1, got {n}")


class Policy(ABC):
    """Ranks offers for an app open, and may learn from the rewards that follow."""

    name: str

    @abstractmethod
    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        """Return up to ``n`` of the ``candidates``, best first."""

    def update(self, context: Context, offer_id: str, reward: float) -> None:  # noqa: B027
        """Learn from one reward in [0, 1]. Baselines that never learn ignore it."""


class BanditPolicy(Policy):
    """A policy that learns, keeping its statistics in the options store.

    ``context`` sets whose experience is pooled (see ``context_key``). ``decay`` < 1
    switches on forgetting: before each update, every offer's evidence in that context
    shrinks by ``decay``, so recent responses outweigh old ones.
    """

    def __init__(
        self,
        name: str,
        store: OptionsStore,
        *,
        rng: np.random.Generator,
        context: str = "global",
        decay: float = 1.0,
    ) -> None:
        if context not in CONTEXT_MODES:
            raise ValueError(f"unknown context mode {context!r}; expected one of {CONTEXT_MODES}")
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.name = name
        self.store = store
        self.rng = rng
        self.context = context
        self.decay = decay

    def key(self, context: Context) -> str:
        return context_key(context, self.context)

    def stats(self, context: Context, candidates: Sequence[Offer]) -> dict[str, ArmStats]:
        """Current statistics for each candidate; untried offers start at the prior."""
        return self.store.get(self.name, self.key(context), [o.offer_id for o in candidates])

    def update(self, context: Context, offer_id: str, reward: float) -> None:
        """Forget a little (if decay is on), then add the new evidence."""
        if not 0 <= reward <= 1:
            raise ValueError(f"reward must be in [0, 1], got {reward}")
        key = self.key(context)
        arms = self.store.get_context(self.name, key)
        if self.decay < 1:
            arms = {offer: stats.decayed(self.decay) for offer, stats in arms.items()}
        arms[offer_id] = arms.get(offer_id, ArmStats()).updated(reward)
        changed = arms if self.decay < 1 else {offer_id: arms[offer_id]}
        self.store.put(self.name, key, changed)
