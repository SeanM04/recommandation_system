"""Core domain types shared by the API, the bandit policies and the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SEGMENTS = ("student", "young_professional", "family", "small_business", "pensioner")
DEVICE_TIERS = ("low", "mid", "high")
CONTRACT_TYPES = ("prepaid", "contract")
PROVINCES = (
    "Gauteng",
    "KwaZulu-Natal",
    "Western Cape",
    "Eastern Cape",
    "Limpopo",
    "Mpumalanga",
    "North West",
    "Free State",
    "Northern Cape",
)
TIMES_OF_DAY = ("morning", "afternoon", "evening", "night")
DAYS_OF_WEEK = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
CHANNELS = ("app", "ussd", "sms")

SALARY_PAYDAY = 25  # most South African salaries land on the 25th
GRANT_PAYDAYS = range(1, 8)  # SASSA older-person grants are paid in the first week
LOW_SPEND_ZAR = 200
HIGH_SPEND_ZAR = 600


class Offer(BaseModel):
    """A product the app can recommend, as defined in ``config/offers.yaml``."""

    model_config = ConfigDict(frozen=True)

    offer_id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    category: str
    price_zar: float = Field(ge=0)
    margin_zar: float = Field(ge=0)
    eligible_segments: tuple[str, ...] = Field(min_length=1)
    description: str = ""

    @field_validator("eligible_segments")
    @classmethod
    def _segments_are_known(cls, segments: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(segments) - set(SEGMENTS)
        if unknown:
            raise ValueError(f"unknown segments: {sorted(unknown)}")
        return segments

    @model_validator(mode="after")
    def _margin_within_price(self) -> Offer:
        if self.margin_zar > self.price_zar:
            raise ValueError(f"{self.offer_id}: margin_zar cannot exceed price_zar")
        return self

    def is_eligible(self, segment: str) -> bool:
        """True if the offer may be shown to customers in ``segment``."""
        return segment in self.eligible_segments


@dataclass(frozen=True)
class Customer:
    """Static profile of an app user: what the business already knows about them."""

    customer_id: str
    segment: str
    device_tier: str
    prepaid_vs_contract: str
    avg_monthly_spend: float
    province: str
    preferred_time_of_day: str


@dataclass(frozen=True)
class Context:
    """Everything known at the moment a customer opens the app: profile plus situation."""

    customer_id: str
    segment: str
    device_tier: str
    prepaid_vs_contract: str
    avg_monthly_spend: float
    province: str
    channel: str
    time_of_day: str
    day_of_week: str
    is_payday_week: bool

    @property
    def is_weekend(self) -> bool:
        return self.day_of_week in ("saturday", "sunday")

    @property
    def spend_band(self) -> str:
        """Coarse spend bucket: ``low`` (< R200), ``medium`` or ``high`` (> R600) per month."""
        if self.avg_monthly_spend < LOW_SPEND_ZAR:
            return "low"
        if self.avg_monthly_spend > HIGH_SPEND_ZAR:
            return "high"
        return "medium"

    @classmethod
    def build(
        cls, customer: Customer, *, day: date, time_of_day: str, channel: str = "app"
    ) -> Context:
        """Combine a customer's profile with the calendar day and time of an app open."""
        return cls(
            customer_id=customer.customer_id,
            segment=customer.segment,
            device_tier=customer.device_tier,
            prepaid_vs_contract=customer.prepaid_vs_contract,
            avg_monthly_spend=customer.avg_monthly_spend,
            province=customer.province,
            channel=channel,
            time_of_day=time_of_day,
            day_of_week=DAYS_OF_WEEK[day.weekday()],
            is_payday_week=is_payday_week(day, customer.segment),
        )


def time_of_day_for_hour(hour: int) -> str:
    """Map a clock hour (0-23) to a time-of-day bucket."""
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be between 0 and 23, got {hour}")
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def is_payday_week(day: date, segment: str) -> bool:
    """True when the customer has just been paid.

    Pensioners are paid a SASSA grant in the first week of the month; everyone else is
    treated as salaried, paid on the 25th, with the payday week running to month end.
    """
    if segment == "pensioner":
        return day.day in GRANT_PAYDAYS
    return day.day >= SALARY_PAYDAY
