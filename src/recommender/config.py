"""Load and validate the YAML configuration: global settings and the offer catalogue."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

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

    @field_validator("context")
    @classmethod
    def _known_context_mode(cls, mode: str) -> str:
        if mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context mode {mode!r}; expected one of {CONTEXT_MODES}")
        return mode


class RecommendationSettings(BaseModel):
    number_of_offers: int = Field(3, ge=1)
    reward: Literal["binary", "margin"] = "binary"
    learn_from_ignored: bool = False
    active_policy: str


class Settings(BaseModel):
    seed: int
    paths: PathSettings
    recommendation: RecommendationSettings
    policies: dict[str, PolicySpec]

    @model_validator(mode="after")
    def _active_policy_is_configured(self) -> Settings:
        if self.recommendation.active_policy not in self.policies:
            raise ValueError(
                f"active_policy {self.recommendation.active_policy!r} is not in policies"
            )
        return self

    @property
    def database_url(self) -> str:
        """``$RECOMMENDER_DATABASE_URL`` (Postgres, set in .env), else the local SQLite file."""
        default = f"sqlite:///{self.paths.database.as_posix()}"
        return os.environ.get("RECOMMENDER_DATABASE_URL", default)


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
