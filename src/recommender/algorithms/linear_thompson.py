"""Thompson sampling on customer features: the step beyond segments.

Instead of one Beta belief per segment and offer, each offer gets a Bayesian linear model
of its take-up rate over the customer's features and the moment (segment, device,
contract, spend band, time of day, weekend, payday week, border province). It can learn
that flagship handsets like device insurance in every segment, or that evenings favour
entertainment, and share that across segments instead of learning it five times.

The model per offer is ridge regression with a Gaussian posterior: precision
A = prior * I + sum of x x^T, and b = sum of reward * x. To rank, draw one weight vector
per offer from N(A^-1 b, scale^2 A^-1) and score x . w. Forgetting shrinks A towards the
prior and b towards zero before each update, as the Beta policies do with their counts.

The state is a matrix per offer, which the options store (counts per row) can't hold, so
this policy runs in the simulator only: ``servable`` is False and the API skips it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from recommender.algorithms.base import Policy, ScoredOffer, check_slate_size
from recommender.domain import DEVICE_TIERS, SEGMENTS, TIMES_OF_DAY, Context, Offer

SPEND_BANDS = ("low", "medium", "high")
BORDER_PROVINCES = ("Limpopo", "Mpumalanga", "North West")
FEATURES = (
    ["bias"]
    + [f"segment={s}" for s in SEGMENTS]
    + [f"device={d}" for d in DEVICE_TIERS]
    + ["contract"]
    + [f"spend={s}" for s in SPEND_BANDS]
    + [f"time={t}" for t in TIMES_OF_DAY]
    + ["weekend", "payday_week", "border_province"]
)


def features(context: Context) -> np.ndarray:
    """The feature vector of one app open: indicators for the customer and the moment."""
    x = np.zeros(len(FEATURES))
    on = [
        "bias",
        f"segment={context.segment}",
        f"device={context.device_tier}",
        f"spend={context.spend_band}",
        f"time={context.time_of_day}",
    ]
    if context.prepaid_vs_contract == "contract":
        on.append("contract")
    if context.is_weekend:
        on.append("weekend")
    if context.is_payday_week:
        on.append("payday_week")
    if context.province in BORDER_PROVINCES:
        on.append("border_province")
    for name in on:
        x[FEATURES.index(name)] = 1.0
    return x


class LinearThompsonPolicy(Policy):
    """Thompson sampling with a Bayesian linear model of take-up per offer (see module)."""

    learns = True
    servable = False  # its matrices live in memory; the API's options store holds counts

    def __init__(
        self,
        name: str,
        *,
        rng: np.random.Generator,
        decay: float = 1.0,
        prior: float = 1.0,
        scale: float = 0.05,
    ) -> None:
        """A policy with no evidence yet: every offer's weights start at zero."""
        if not 0 < decay <= 1:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.name = name
        self.rng = rng
        self.decay = decay
        self.prior = prior  # ridge strength: how firmly weights start at zero
        self.scale = scale  # posterior width: how far draws stray, i.e. how much it explores
        self._index: dict[str, int] = {}  # offer id -> row in the arrays below
        d = len(FEATURES)
        self._evidence = np.zeros((0, d, d))  # sum of x x^T per offer (forgetting applied)
        self._targets = np.zeros((0, d))  # sum of reward * x per offer

    def _rows(self, offer_ids: Sequence[str]) -> np.ndarray:
        """Array rows for these offers, adding empty models for offers never seen."""
        d = len(FEATURES)
        for offer_id in offer_ids:
            if offer_id not in self._index:
                self._index[offer_id] = len(self._index)
                self._evidence = np.concatenate([self._evidence, np.zeros((1, d, d))])
                self._targets = np.concatenate([self._targets, np.zeros((1, d))])
        return np.array([self._index[o] for o in offer_ids])

    def rank(self, context: Context, candidates: Sequence[Offer], n: int) -> list[ScoredOffer]:
        """Draw a weight vector per offer from its posterior; rank by predicted take-up."""
        check_slate_size(n)
        ids = [offer.offer_id for offer in candidates]
        rows = self._rows(ids)
        x = features(context)
        precision = self._evidence[rows] + self.prior * np.eye(len(FEATURES))
        mean = np.linalg.solve(precision, self._targets[rows][..., None])[..., 0]
        lower = np.linalg.cholesky(precision)  # precision = L L^T, so cov = L^-T L^-1
        noise = self.rng.standard_normal(mean.shape)
        spread = np.linalg.solve(np.swapaxes(lower, 1, 2), noise[..., None])[..., 0]
        scores = (mean + self.scale * spread) @ x
        best = np.argsort(-scores)[:n]
        return [ScoredOffer(ids[i], float(scores[i])) for i in best]

    def update(self, context: Context, offer_id: str, reward: float) -> None:
        """Forget a little across every offer (if decay is on), then add this response."""
        if not 0 <= reward <= 1:
            raise ValueError(f"reward must be in [0, 1], got {reward}")
        (row,) = self._rows([offer_id])
        if self.decay < 1:
            self._evidence *= self.decay
            self._targets *= self.decay
        x = features(context)
        self._evidence[row] += np.outer(x, x)
        self._targets[row] += reward * x

    def weights(self, offer_id: str) -> dict[str, float]:
        """The posterior mean weight of each feature for one offer (for inspection)."""
        if offer_id not in self._index:
            return dict.fromkeys(FEATURES, 0.0)
        row = self._index[offer_id]
        precision = self._evidence[row] + self.prior * np.eye(len(FEATURES))
        return dict(zip(FEATURES, np.linalg.solve(precision, self._targets[row]), strict=True))
