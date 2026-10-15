"""FastAPI application entry point.

Run locally with:  uvicorn recommender.api.main:app --reload
"""

from fastapi import FastAPI

from recommender import __version__

app = FastAPI(
    title="Telco Dynamic Offer Recommender",
    version=__version__,
    description="Real-time offer ranking with multi-armed bandits that learn from every response.",
)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness check used by Docker and the dashboard."""
    return {"status": "ok", "version": __version__}
