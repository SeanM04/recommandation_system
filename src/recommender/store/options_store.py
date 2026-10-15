"""The options store: the model's state.

One row of statistics per (policy, context key, offer). Policies hold no state of their
own; they read and write these rows. ``InMemoryOptionsStore`` serves fast simulations
and tests; ``SqlOptionsStore`` (see ``sql.py``) persists the same rows for the API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

PRIOR_ALPHA = 1.0  # Beta(1, 1): before any evidence, every take-up rate is equally likely
PRIOR_BETA = 1.0


@dataclass(frozen=True)
class ArmStats:
    """What a policy knows about one offer in one context.

    ``pulls`` counts the rewards seen and ``total_reward`` sums them. With forgetting
    switched on both shrink over time, so ``pulls`` is an *effective* count, not an
    integer. The Beta belief follows from them, so it can never disagree with them.
    """

    pulls: float = 0.0
    total_reward: float = 0.0

    @property
    def alpha(self) -> float:
        return PRIOR_ALPHA + self.total_reward

    @property
    def beta(self) -> float:
        return PRIOR_BETA + self.pulls - self.total_reward

    @property
    def mean(self) -> float:
        """Posterior mean take-up rate: alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def updated(self, reward: float) -> ArmStats:
        """Add one reward in [0, 1]: alpha += reward, beta += 1 - reward."""
        return ArmStats(self.pulls + 1, self.total_reward + reward)

    def decayed(self, gamma: float) -> ArmStats:
        """Forget: shrink the evidence by ``gamma`` towards the Beta(1, 1) prior."""
        return ArmStats(self.pulls * gamma, self.total_reward * gamma)


@dataclass(frozen=True)
class OptionRow:
    """One row of the options store, as shown by the state endpoint and dashboard."""

    policy: str
    context_key: str
    offer_id: str
    stats: ArmStats
    updated_at: datetime


class OptionsStore(ABC):
    """Where arm statistics live. Implementations must behave identically."""

    def get(self, policy: str, context_key: str, offer_ids: Sequence[str]) -> dict[str, ArmStats]:
        """Statistics for each offer, in the order given; offers never updated get the prior."""
        known = self.get_context(policy, context_key)
        return {offer_id: known.get(offer_id, ArmStats()) for offer_id in offer_ids}

    @abstractmethod
    def get_context(self, policy: str, context_key: str) -> dict[str, ArmStats]:
        """Every offer with statistics for this policy and context key."""

    @abstractmethod
    def put(self, policy: str, context_key: str, arms: Mapping[str, ArmStats]) -> None:
        """Insert or replace the rows for these offers."""

    @abstractmethod
    def rows(self, policy: str | None = None) -> list[OptionRow]:
        """All rows, or one policy's rows, sorted by policy, context key and offer."""

    @abstractmethod
    def reset(self, policy: str | None = None) -> None:
        """Forget everything, or everything one policy has learned."""


class InMemoryOptionsStore(OptionsStore):
    """Options store in a Python dict: no I/O, for simulations and tests."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], dict[str, tuple[ArmStats, datetime]]] = {}

    def get_context(self, policy: str, context_key: str) -> dict[str, ArmStats]:
        rows = self._rows.get((policy, context_key), {})
        return {offer_id: stats for offer_id, (stats, _) in rows.items()}

    def put(self, policy: str, context_key: str, arms: Mapping[str, ArmStats]) -> None:
        now = datetime.now(UTC)
        rows = self._rows.setdefault((policy, context_key), {})
        rows.update({offer_id: (stats, now) for offer_id, stats in arms.items()})

    def rows(self, policy: str | None = None) -> list[OptionRow]:
        return sorted(
            (
                OptionRow(name, key, offer_id, stats, updated_at)
                for (name, key), rows in self._rows.items()
                if policy is None or name == policy
                for offer_id, (stats, updated_at) in rows.items()
            ),
            key=lambda row: (row.policy, row.context_key, row.offer_id),
        )

    def reset(self, policy: str | None = None) -> None:
        if policy is None:
            self._rows.clear()
        else:
            self._rows = {key: rows for key, rows in self._rows.items() if key[0] != policy}
