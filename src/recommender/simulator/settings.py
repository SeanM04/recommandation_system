"""Settings for the simulated world (``config/simulation.yaml``).

Kept apart from the recommender's own settings so the recommender has no path to the
hidden ground truth.
"""

from __future__ import annotations

import math
import os
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from recommender.config import PROJECT_ROOT, read_yaml, resolve_path
from recommender.domain import SEGMENTS

DEFAULT_SIMULATION_PATH = PROJECT_ROOT / "config" / "simulation.yaml"


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


class CalendarSettings(BaseModel):
    start_date: date
    n_days: int = Field(gt=0)

    def days(self) -> list[date]:
        """Every simulated calendar day, in order."""
        return [self.start_date + timedelta(days=offset) for offset in range(self.n_days)]


class SimulationSettings(BaseModel):
    population: PopulationSettings
    ground_truth: Path
    calendar: CalendarSettings

    @field_validator("ground_truth")
    @classmethod
    def _absolute(cls, path: Path) -> Path:
        return resolve_path(path)


def load_simulation_settings(path: Path | str | None = None) -> SimulationSettings:
    """Load ``config/simulation.yaml`` (or ``$RECOMMENDER_SIMULATION``, or ``path``)."""
    path = Path(path or os.environ.get("RECOMMENDER_SIMULATION", DEFAULT_SIMULATION_PATH))
    return SimulationSettings.model_validate(read_yaml(path))
