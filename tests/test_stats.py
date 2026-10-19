"""Bootstrap intervals for paired, seed-by-seed comparisons."""

import pytest

from recommender.evaluation.stats import (
    count,
    difference_interval,
    percent,
    reduction_interval,
    uplift_interval,
)

LEARNER = [1000.0, 1050.0, 980.0, 1100.0, 1020.0, 990.0, 1060.0, 1010.0, 1040.0, 1000.0]
BASELINE = [5500.0, 5600.0, 5400.0, 5700.0, 5450.0, 5550.0, 5650.0, 5500.0, 5600.0, 5480.0]


def test_the_interval_contains_the_estimate_and_is_reproducible() -> None:
    low, high = reduction_interval(LEARNER, BASELINE)
    estimate = 1 - sum(LEARNER) / sum(BASELINE)
    assert low < estimate < high
    assert high - low < 0.02  # ten tight pairs: a narrow interval
    assert reduction_interval(LEARNER, BASELINE) == (low, high)  # fixed seed


def test_a_noisy_difference_gets_a_wide_interval() -> None:
    noisy = [100.0, -80.0, 60.0, -120.0, 90.0, -40.0, 70.0, -100.0, 20.0, 10.0]
    low, high = difference_interval(noisy, [0.0] * 10)
    assert low < 0 < high  # the data can't tell which way it goes


def test_uplift_and_the_text_helpers() -> None:
    low, high = uplift_interval([2.0, 2.2, 1.8], [1.0, 1.0, 1.0])
    assert 0.8 - 1e-9 <= low <= high <= 1.2 + 1e-9  # resampled means of 1.8, 2.0, 2.2
    assert percent((0.781, 0.842)) == "78%–84%"
    assert count((101.6, 1585.2)) == "102–1,585"


def test_series_must_pair_up() -> None:
    with pytest.raises(ValueError, match="equally long"):
        difference_interval([1.0, 2.0], [1.0])
