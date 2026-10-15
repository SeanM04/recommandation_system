"""Load and validate the YAML configuration: global settings and the offer catalogue."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator

from recommender.domain import Offer

PROJECT_ROOT = Path(os.environ.get("RECOMMENDER_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


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


class Settings(BaseModel):
    seed: int
    paths: PathSettings


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
