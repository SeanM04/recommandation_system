"""Build the policies configured in ``settings.yaml``, by name."""

from __future__ import annotations

import zlib
from collections.abc import Mapping

import numpy as np

from recommender.algorithms.base import Policy
from recommender.algorithms.epsilon_greedy import EpsilonGreedyPolicy
from recommender.algorithms.random_policy import RandomPolicy
from recommender.algorithms.static import StaticPolicy
from recommender.algorithms.thompson import ThompsonSamplingPolicy
from recommender.config import PolicySpec, Settings
from recommender.store.options_store import OptionsStore


def policy_rng(seed: int, name: str) -> np.random.Generator:
    """A reproducible random stream per policy, so adding one policy never changes
    another policy's results."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def build_policy(
    name: str,
    spec: PolicySpec,
    store: OptionsStore,
    *,
    seed: int,
    popularity: Mapping[str, float] | None = None,
) -> Policy:
    """One policy from its spec. ``popularity`` is the static baseline's batch-trained table."""
    rng = policy_rng(seed, name)
    if spec.type == "random":
        return RandomPolicy(name, rng=rng)
    if spec.type == "static":
        return StaticPolicy(popularity, name=name)
    if spec.type == "epsilon_greedy":
        return EpsilonGreedyPolicy(
            name,
            store,
            rng=rng,
            context_mode=spec.context,
            decay=spec.decay,
            epsilon=spec.epsilon,
            epsilon_decay=spec.epsilon_decay,
            min_epsilon=spec.min_epsilon,
        )
    if spec.type == "thompson":
        return ThompsonSamplingPolicy(
            name, store, rng=rng, context_mode=spec.context, decay=spec.decay
        )
    raise ValueError(f"unknown policy type {spec.type!r}")


def build_policies(
    settings: Settings,
    store: OptionsStore,
    *,
    popularity: Mapping[str, float] | None = None,
    seed: int | None = None,
) -> dict[str, Policy]:
    """Every policy in ``settings.policies``, sharing one options store.

    ``seed`` defaults to the configured seed; the API passes one per server process.
    """
    seed = settings.seed if seed is None else seed
    return {
        name: build_policy(name, spec, store, seed=seed, popularity=popularity)
        for name, spec in settings.policies.items()
    }
