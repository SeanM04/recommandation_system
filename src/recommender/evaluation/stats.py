"""Confidence intervals for paired, seed-by-seed comparisons.

Every policy faces the identical app opens in a given seed, so two policies' results come
in pairs, one per seed. A bootstrap resamples those pairs (seeds) with replacement and
recomputes the statistic each time; the middle 95% of the results is the interval. With
ten seeds the intervals are approximate, but they show how much a claim could move if
the runs had come out differently, which a single average can't.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

LEVEL = 0.95
DRAWS = 10_000


def _pairs(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The two series as arrays, checked to be the same length (one value per seed)."""
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(first) == 0:
        raise ValueError("need two equally long, non-empty series: one value per seed each")
    return first, second


def _resamples(n: int, draws: int, seed: int) -> np.ndarray:
    """``draws`` bootstrap samples of seed positions, each of size ``n``."""
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def _bounds(values: np.ndarray, level: float) -> tuple[float, float]:
    """The central ``level`` share of the bootstrap distribution."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def difference_interval(
    a: ArrayLike,
    b: ArrayLike,
    *,
    level: float = LEVEL,
    draws: int = DRAWS,
    seed: int = 0,
) -> tuple[float, float]:
    """Interval for the mean of ``a - b``, paired by seed."""
    first, second = _pairs(a, b)
    picks = _resamples(len(first), draws, seed)
    return _bounds((first - second)[picks].mean(axis=1), level)


def ratio_interval(
    a: ArrayLike,
    b: ArrayLike,
    *,
    level: float = LEVEL,
    draws: int = DRAWS,
    seed: int = 0,
) -> tuple[float, float]:
    """Interval for ``mean(a) / mean(b)``, resampling the seeds as pairs."""
    first, second = _pairs(a, b)
    picks = _resamples(len(first), draws, seed)
    return _bounds(first[picks].mean(axis=1) / second[picks].mean(axis=1), level)


def reduction_interval(a: ArrayLike, b: ArrayLike, *, seed: int = 0) -> tuple[float, float]:
    """Interval for ``1 - mean(a) / mean(b)``: how much less ``a`` is than ``b``."""
    low, high = ratio_interval(a, b, seed=seed)
    return 1 - high, 1 - low


def uplift_interval(a: ArrayLike, b: ArrayLike, *, seed: int = 0) -> tuple[float, float]:
    """Interval for ``mean(a) / mean(b) - 1``: how much more ``a`` is than ``b``."""
    low, high = ratio_interval(a, b, seed=seed)
    return low - 1, high - 1


def percent(interval: tuple[float, float]) -> str:
    """'78–84%': an interval of shares, for text."""
    return f"{interval[0]:.0%}–{interval[1]:.0%}"


def count(interval: tuple[float, float]) -> str:
    """'102–158': an interval of counts, for text."""
    return f"{interval[0]:,.0f}–{interval[1]:,.0f}"
