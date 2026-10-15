"""Hidden ground truth: the real chance that a customer takes up an offer.

ONLY the simulator and the offline regret oracle may use this module. The recommender
never sees these numbers; it has to learn them from customer responses.

    p(take-up) = base_take_up[segment][offer] x every matching context effect   (capped)

The numbers live in ``config/ground_truth.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.config import read_yaml
from recommender.domain import SEGMENTS, Context, Customer, Offer
from recommender.simulator.population import sample_context

CONTEXT_ATTRIBUTES = {f.name for f in fields(Context)} | {"is_weekend", "spend_band"}


def _matches(value: Any, expected: Any) -> bool:
    """An effect condition is either a single value or a list of allowed values."""
    return value in expected if isinstance(expected, list) else value == expected


@dataclass(frozen=True)
class Effect:
    """A multiplicative nudge, for example "in payday week, insurance take-up x1.3"."""

    name: str
    when: Mapping[str, Any]  # context attribute -> required value (or list of values)
    multiply: Mapping[str, float]  # offer_id or category -> multiplier

    def applies_to(self, context: Context) -> bool:
        return all(_matches(getattr(context, attr), want) for attr, want in self.when.items())

    def multiplier_for(self, offer: Offer) -> float:
        return self.multiply.get(offer.offer_id, self.multiply.get(offer.category, 1.0))


@dataclass(frozen=True)
class GroundTruth:
    """True take-up probabilities, before or after the behaviour shift."""

    base_take_up: Mapping[str, Mapping[str, float]]
    effects: tuple[Effect, ...]
    max_probability: float
    shift_description: str = ""
    shift_base_take_up: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    is_shifted: bool = False

    @classmethod
    def from_yaml(cls, path: Path, offers: Sequence[Offer]) -> GroundTruth:
        """Load ``ground_truth.yaml`` and validate it against the offer catalogue."""
        raw = read_yaml(path)
        shift = raw.get("behaviour_shift", {})
        truth = cls(
            base_take_up=raw["base_take_up"],
            effects=tuple(Effect(**effect) for effect in raw.get("effects", [])),
            max_probability=raw["max_probability"],
            shift_description=shift.get("description", ""),
            shift_base_take_up=shift.get("base_take_up", {}),
        )
        truth.validate(offers)
        return truth

    def validate(self, offers: Sequence[Offer]) -> None:
        """Fail fast on typos: unknown segments or offers, ineligible pairs, bad effects."""
        offer_ids = {offer.offer_id for offer in offers}
        categories = {offer.category for offer in offers}

        if set(self.base_take_up) != set(SEGMENTS):
            raise ValueError(f"base_take_up must define exactly these segments: {SEGMENTS}")
        for segment in SEGMENTS:
            eligible = {offer.offer_id for offer in offers if offer.is_eligible(segment)}
            defined = set(self.base_take_up[segment])
            if defined != eligible:
                raise ValueError(
                    f"{segment}: base_take_up must cover exactly the eligible offers "
                    f"(missing {sorted(eligible - defined)}, "
                    f"not eligible {sorted(defined - eligible)})"
                )
        for segment, rates in self.shift_base_take_up.items():
            unknown = set(rates) - set(self.base_take_up.get(segment, {}))
            if unknown:
                raise ValueError(f"behaviour_shift.{segment}: not eligible {sorted(unknown)}")

        tables = [*self.base_take_up.values(), *self.shift_base_take_up.values()]
        if any(not 0 <= p <= 1 for rates in tables for p in rates.values()):
            raise ValueError("take-up probabilities must be between 0 and 1")

        for effect in self.effects:
            unknown_attributes = set(effect.when) - CONTEXT_ATTRIBUTES
            if unknown_attributes:
                raise ValueError(f"effect '{effect.name}': unknown {sorted(unknown_attributes)}")
            unknown_keys = set(effect.multiply) - offer_ids - categories
            if unknown_keys:
                raise ValueError(f"effect '{effect.name}': unknown offers {sorted(unknown_keys)}")
            if any(m < 0 for m in effect.multiply.values()):
                raise ValueError(f"effect '{effect.name}': multipliers must be non-negative")

    def after_shift(self) -> GroundTruth:
        """The world after the behaviour shift: overridden base rates, same effects."""
        shifted = {
            segment: {**rates, **self.shift_base_take_up.get(segment, {})}
            for segment, rates in self.base_take_up.items()
        }
        return replace(self, base_take_up=shifted, is_shifted=True)

    def probabilities(self, context: Context, offers: Sequence[Offer]) -> dict[str, float]:
        """True take-up probability of each offer in this context (0 if not eligible)."""
        base = self.base_take_up.get(context.segment, {})
        active_effects = [effect for effect in self.effects if effect.applies_to(context)]
        result = {}
        for offer in offers:
            p = base.get(offer.offer_id, 0.0)
            for effect in active_effects:
                p *= effect.multiplier_for(offer)
            result[offer.offer_id] = min(p, self.max_probability)
        return result

    def take_up_probability(self, context: Context, offer: Offer) -> float:
        """True probability that this customer, in this moment, takes up ``offer``."""
        return self.probabilities(context, [offer])[offer.offer_id]


def take_up_table(
    truth: GroundTruth,
    customers: Sequence[Customer],
    offers: Sequence[Offer],
    days: Sequence[date],
    *,
    by: Sequence[str] = ("segment",),
    samples_per_segment: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """Average true take-up per offer, grouped by context attributes (segment first).

    Contexts are sampled like real app opens: random customers of each segment on random
    days of the simulated calendar. This is the yardstick for "the best offer" per
    segment, or per segment x payday with ``by=("segment", "is_payday_week")``.
    Ineligible offers are NaN.
    """
    if not by or by[0] != "segment":
        raise ValueError("group by 'segment' first: contexts are sampled per segment")
    rng = np.random.default_rng(seed)
    records = []
    for segment in SEGMENTS:
        members = [customer for customer in customers if customer.segment == segment]
        eligible = [offer for offer in offers if offer.is_eligible(segment)]
        for _ in range(samples_per_segment):
            customer = members[rng.integers(len(members))]
            context = sample_context(customer, rng, days[rng.integers(len(days))])
            keys = {attribute: getattr(context, attribute) for attribute in by}
            records.append(keys | truth.probabilities(context, eligible))
    columns = [offer.offer_id for offer in offers]
    return pd.DataFrame(records).groupby(list(by)).mean().reindex(columns=columns)


def winners(table: pd.DataFrame) -> pd.DataFrame:
    """Best and runner-up offer per row of a take-up (or revenue) table.

    ``lead`` is best / runner-up: how clearly the favourite wins.
    """
    rows = {}
    for key, row in table.iterrows():
        ranked = row.dropna().sort_values(ascending=False)
        rows[key] = {
            "best": ranked.index[0],
            "value": ranked.iloc[0],
            "runner_up": ranked.index[1],
            "lead": ranked.iloc[0] / ranked.iloc[1],
        }
    return pd.DataFrame.from_dict(rows, orient="index")
