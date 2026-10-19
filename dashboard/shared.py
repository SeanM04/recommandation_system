"""Data loading, colours and the API client shared by the dashboard pages."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from recommender.config import PROJECT_ROOT, api_client_headers, load_settings
from recommender.db import create_tables, make_engine

# One fixed colour per policy on every chart (categorical palette slots, in order).
POLICY_COLOURS = {
    "thompson_segment_decay": "#2a78d6",
    "static": "#eb6834",
    "random": "#1baf7a",
    "thompson": "#eda100",
    "epsilon_greedy_segment": "#e87ba4",
    "thompson_segment": "#008300",
    "thompson_segment_payday": "#6250d6",
    "epsilon_greedy": "#e34948",
}
OTHER_COLOUR = "#7a8691"  # a ninth series is never a new hue
KEY_POLICIES = ["thompson_segment_decay", "thompson_segment", "static", "random", "thompson"]


def colour(policy: str) -> str:
    """The policy's fixed chart colour (grey for any beyond the palette)."""
    return POLICY_COLOURS.get(policy, OTHER_COLOUR)


@st.cache_resource
def settings():
    """The project settings, loaded once per dashboard process."""
    return load_settings()


@st.cache_resource
def engine():
    """The project database, with every table in place (the dashboard may start first)."""
    connection = make_engine(settings().database_url)
    create_tables(connection)
    return connection


def results_dir() -> Path:
    """Where the simulation results are: results/, or $RECOMMENDER_RESULTS."""
    return Path(os.environ.get("RECOMMENDER_RESULTS", PROJECT_ROOT / "results"))


@st.cache_resource(max_entries=16)
def _read(path: str, modified: float) -> pd.DataFrame:
    """One shared, read-only copy of a results file per saved run.

    ``modified`` is part of the cache key, so a new run is picked up without a restart.
    A resource cache hands every page the same frame instead of a fresh copy per page load
    (the per-app-open log is 450,000 rows), so pages must never change it in place.
    """
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def results_file(name: str) -> pd.DataFrame | None:
    """A results file, or None if make simulate has not produced it yet."""
    path = results_dir() / name
    return _read(str(path), path.stat().st_mtime) if path.exists() else None


def need_results(*frames: pd.DataFrame | None) -> bool:
    """Show how to produce the results if any are missing; True when all are present."""
    if any(frame is None for frame in frames):
        st.info("No simulation results yet. Run `make simulate` (or `make compare`), then reload.")
        return False
    return True


def provenance() -> None:
    """Say which run the page is showing: seeds, app opens and when it was saved."""
    by_seed, rounds = (
        results_file("comparison_by_seed.csv"),
        results_file("comparison_rounds.parquet"),
    )
    if by_seed is None or rounds is None:
        return
    saved = dt.datetime.fromtimestamp((results_dir() / "comparison.csv").stat().st_mtime)
    st.caption(
        f"Showing a run of {by_seed['seed'].nunique()} seeds × {rounds['round'].max() + 1:,} "
        f"app opens, saved {saved:%Y-%m-%d %H:%M}. `make compare` produces the published "
        "10-seed results; `make simulate` a quicker 3-seed run."
    )


def style(figure: go.Figure, height: int = 420) -> go.Figure:
    """Recessive grid, readable hover, legend on top."""
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(128,128,128,0.18)")
    return figure


def mark_shift(figure: go.Figure, shift_round: int | None) -> go.Figure:
    """Draw the behaviour shift as a dashed vertical line, if there is one."""
    if shift_round:
        figure.add_vline(
            x=shift_round,
            line_dash="dash",
            line_color="#a35f08",
            annotation_text="behaviour shift",
            annotation_position="top left",
        )
    return figure


class Api:
    """Talks to the API over HTTP when it is running, otherwise runs the same FastAPI app
    inside the dashboard process (same code paths, no network)."""

    def __init__(self) -> None:
        """Use the API server if one answers; otherwise run the same app in-process."""
        self.url = os.environ.get("RECOMMENDER_API_URL", "http://localhost:8000")
        try:
            requests.get(f"{self.url}/health", timeout=1).raise_for_status()
            self.client, self.base, self.where = requests.Session(), self.url, self.url
            self.client.headers.update(api_client_headers())
        except requests.RequestException:
            from fastapi.testclient import TestClient

            from recommender.api.main import create_app

            self.client = TestClient(create_app(), headers=api_client_headers())
            self.client.__enter__()  # run the app's startup: database, policies
            self.base, self.where = "", "in-process (no API server running)"

    def get(self, path: str, **params) -> dict | list:
        """GET ``path`` with query parameters and return the JSON body."""
        response = self.client.get(f"{self.base}{path}", params=params or None)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, body: dict, **params):
        """POST ``body`` to ``path`` and return the response."""
        return self.client.post(f"{self.base}{path}", json=body, params=params or None)


@st.cache_resource
def api() -> Api:
    """The API client, created once per dashboard process."""
    return Api()


@st.cache_resource
def store():
    """The options store in the database (what the policies have learned)."""
    from recommender.store.sql import SqlOptionsStore

    return SqlOptionsStore(engine())
