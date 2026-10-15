"""Load and validate the YAML configuration: global settings and the offer catalogue."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.engine import URL

from recommender.algorithms.rewards import RewardFn, make_reward
from recommender.domain import CONTEXT_MODES, Offer

PROJECT_ROOT = Path(os.environ.get("RECOMMENDER_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

# Local secrets such as the database password live in .env (git-ignored). Real
# environment variables win over the file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_path(path: Path) -> Path:
    """Interpret relative paths in config files as relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


class PathSettings(BaseModel):
    offers: Path
    customers: Path
    database: Path

    @field_validator("*")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        return resolve_path(path)


class PolicySpec(BaseModel):
    """One configured policy. Settings that don't apply to its type are ignored."""

    type: Literal["random", "static", "epsilon_greedy", "thompson"]
    context: str = "global"  # one of CONTEXT_MODES
    decay: float = Field(1.0, gt=0, le=1)  # 1 = never forget
    epsilon: float = Field(0.1, ge=0, le=1)
    epsilon_decay: float = Field(1.0, gt=0, le=1)
    min_epsilon: float = Field(0.0, ge=0, le=1)
    reward: Literal["binary", "margin"] | None = None  # None: recommendation.reward

    @field_validator("context")
    @classmethod
    def _known_context_mode(cls, mode: str) -> str:
        if mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context mode {mode!r}; expected one of {CONTEXT_MODES}")
        return mode


class RecommendationSettings(BaseModel):
    number_of_offers: int = Field(3, ge=1)
    max_offers: int = Field(5, ge=1)
    reward: Literal["binary", "margin"] = "binary"
    learn_from_ignored: bool = False
    active_policy: str


class Settings(BaseModel):
    seed: int
    timezone: str = "Africa/Johannesburg"
    paths: PathSettings
    recommendation: RecommendationSettings
    policies: dict[str, PolicySpec]

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, name: str) -> str:
        ZoneInfo(name)  # raises for an unknown zone
        return name

    @model_validator(mode="after")
    def _active_policy_is_configured(self) -> Settings:
        if self.recommendation.active_policy not in self.policies:
            raise ValueError(
                f"active_policy {self.recommendation.active_policy!r} is not in policies"
            )
        return self

    def reward_for(self, policy: str) -> str:
        """The reward a policy learns from: its own, or the global default."""
        return self.policies[policy].reward or self.recommendation.reward

    def reward_function(self, policy: str, offers: Sequence[Offer]) -> RewardFn:
        """The configured reward function for ``policy``."""
        return make_reward(
            self.reward_for(policy),
            offers,
            learn_from_ignored=self.recommendation.learn_from_ignored,
        )

    @property
    def database_url(self) -> str:
        """Where the database is, from the environment (normally ``.env``). First match wins:

        1. ``RECOMMENDER_DATABASE_URL``, a full SQLAlchemy URL;
        2. ``DB_NAME`` and ``DB_USER`` (plus ``DB_PASSWORD``, ``DB_HOST``, ``DB_PORT``):
           Postgres, e.g. the ``Telco_Dynamic_Recomender`` database;
        3. the local SQLite file in ``paths.database``.
        """
        if url := os.environ.get("RECOMMENDER_DATABASE_URL"):
            return url
        if os.environ.get("DB_NAME") and os.environ.get("DB_USER"):
            return postgres_url(
                database=os.environ["DB_NAME"],
                user=os.environ["DB_USER"],
                password=os.environ.get("DB_PASSWORD"),
                host=os.environ.get("DB_HOST") or "localhost",
                port=int(os.environ.get("DB_PORT") or 5432),
            )
        return f"sqlite:///{self.paths.database.as_posix()}"

    @property
    def require_api_key(self) -> bool:
        """Whether API calls need an ``X-API-Key`` header: ``RECOMMENDER_REQUIRE_API_KEY``
        in the environment (``.env``). Off by default, so a clean machine runs as is."""
        return os.environ.get("RECOMMENDER_REQUIRE_API_KEY", "").lower() in ("1", "true", "yes")


def api_client_headers() -> dict[str, str]:
    """Headers for calling the API: the key in ``RECOMMENDER_API_KEY``, if one is set."""
    key = os.environ.get("RECOMMENDER_API_KEY")
    return {"X-API-Key": key} if key else {}


def postgres_url(*, database: str, user: str, password: str | None, host: str, port: int) -> str:
    """A psycopg URL from its parts; special characters in the password are escaped."""
    url = URL.create(
        "postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )
    return url.render_as_string(hide_password=False)


def read_yaml(path: Path) -> Any:
    """Parse a YAML file with the safe loader."""
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_settings(path: Path | str | None = None) -> Settings:
    """Load ``config/settings.yaml`` (or ``$RECOMMENDER_SETTINGS``, or ``path``)."""
    path = Path(path or os.environ.get("RECOMMENDER_SETTINGS", DEFAULT_SETTINGS_PATH))
    return Settings.model_validate(read_yaml(path))


def load_offers(path: Path | str | None = None) -> list[Offer]:
    """Load and validate the offer catalogue. Offer ids must be unique."""
    path = Path(path) if path else load_settings().paths.offers
    offers = [Offer.model_validate(item) for item in read_yaml(path)["offers"]]
    ids = [offer.offer_id for offer in offers]
    duplicates = sorted({offer_id for offer_id in ids if ids.count(offer_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate offer_id(s) in catalogue: {duplicates}")
    return offers
