"""Load and validate the YAML configuration: global settings and the offer catalogue."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from recommender.domain import SEGMENTS, Offer

PROJECT_ROOT = Path(os.environ.get("RECOMMENDER_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _resolve(path: Path) -> Path:
    """Interpret relative paths in config files as relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


class PathSettings(BaseModel):
    offers: Path
    customers: Path
    database: Path

    @field_validator("*")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        return _resolve(path)


class PopulationSettings(BaseModel):
    n_customers: int = Field(gt=0)
    segment_mix: dict[str, float]

    @field_validator("segment_mix")
    @classmethod
    def _valid_mix(cls, mix: dict[str, float]) -> dict[str, float]:
        if set(mix) != set(SEGMENTS):
            raise ValueError(f"segment_mix must define exactly these segments: {SEGMENTS}")
        if any(share < 0 for share in mix.values()):
            raise ValueError("segment_mix shares must be non-negative")
        if not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"segment_mix must sum to 1, got {sum(mix.values()):.4f}")
        return mix


class SimulationSettings(BaseModel):
    ground_truth: Path

    @field_validator("ground_truth")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        return _resolve(path)


class Settings(BaseModel):
    seed: int
    paths: PathSettings
    population: PopulationSettings
    simulation: SimulationSettings


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
