"""Request and response bodies for the API, validated by Pydantic."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Kept as literals so the OpenAPI docs list the allowed values; a test checks they match
# the constants in recommender.domain.
Channel = Literal["app", "ussd", "sms"]
TimeOfDay = Literal["morning", "afternoon", "evening", "night"]
Outcome = Literal["taken_up", "viewed", "ignored"]


class ContextOverrides(BaseModel):
    """Optional overrides for the moment of the app open, for simulations and testing.
    By default the API uses today's date and the current time in the configured timezone."""

    model_config = ConfigDict(extra="forbid")

    day: date | None = Field(None, description="Calendar day; sets day_of_week and payday week")
    time_of_day: TimeOfDay | None = None
    is_payday_week: bool | None = Field(None, description="Overrides the calendar's answer")


class InvocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=16, examples=["C00042"])
    channel: Channel = "app"
    number_of_offers: int | None = Field(
        None, ge=1, description="Defaults to recommendation.number_of_offers"
    )
    context: ContextOverrides | None = None


class RankedOffer(BaseModel):
    rank: int
    offer_id: str
    name: str
    category: str
    price_zar: float
    score: float = Field(description="What the policy ranked by, e.g. a Thompson draw")


class ContextOut(BaseModel):
    segment: str
    device_tier: str
    prepaid_vs_contract: str
    avg_monthly_spend: float
    province: str
    channel: str
    time_of_day: str
    day_of_week: str
    is_payday_week: bool


class InvocationResponse(BaseModel):
    invocation_id: str
    policy: str
    customer_id: str
    context: ContextOut
    offers: list[RankedOffer]


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invocation_id: str = Field(min_length=1, max_length=36)
    offer_id: str = Field(min_length=1, max_length=64)
    outcome: Outcome
    value: float | None = Field(None, ge=0, description="Optional value, e.g. rand spent")


class ResponseResult(BaseModel):
    invocation_id: str
    offer_id: str
    outcome: Outcome
    policy: str
    reward: float | None = Field(description="What the policy learned from; null = no update")
    model_updated: bool


class OfferOut(BaseModel):
    offer_id: str
    name: str
    category: str
    price_zar: float
    margin_zar: float
    eligible_segments: list[str]
    description: str


class CustomerOut(BaseModel):
    customer_id: str
    segment: str


class PolicyInfo(BaseModel):
    name: str
    type: str
    context: str
    decay: float
    learns: bool
    active: bool


class ArmState(BaseModel):
    context_key: str
    offer_id: str
    pulls: float
    total_reward: float
    alpha: float
    beta: float
    mean: float = Field(description="Posterior mean take-up rate")
    updated_at: datetime


class PolicyState(BaseModel):
    policy: str
    type: str
    learns: bool
    arms: list[ArmState]
    popularity: dict[str, float] | None = Field(
        None, description="The static baseline's frozen table"
    )


class PolicyMetrics(BaseModel):
    invocations: int
    impressions: int
    responses: dict[str, int]
    take_up_rate: float = Field(description="Take-ups per app open")
    revenue_zar: float


class LatencyStats(BaseModel):
    count: int
    p50_ms: float
    p95_ms: float
    max_ms: float


class MetricsResponse(BaseModel):
    policies: dict[str, PolicyMetrics]
    latency: dict[str, LatencyStats]
