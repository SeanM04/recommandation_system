"""Request timing: a response header, a JSON log line, and percentiles for /metrics."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

import numpy as np
from fastapi import FastAPI, Request, Response

logger = logging.getLogger("recommender.api")


class LatencyTracker:
    """Keeps the most recent request durations per endpoint."""

    def __init__(self, window: int = 5000) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._lock = threading.Lock()

    def record(self, endpoint: str, milliseconds: float) -> None:
        with self._lock:
            self._samples[endpoint].append(milliseconds)

    def summary(self) -> dict[str, dict[str, float]]:
        """Count, p50, p95 and max per endpoint, in milliseconds."""
        with self._lock:
            snapshot = {endpoint: list(samples) for endpoint, samples in self._samples.items()}
        return {
            endpoint: {
                "count": len(samples),
                "p50_ms": round(float(np.percentile(samples, 50)), 2),
                "p95_ms": round(float(np.percentile(samples, 95)), 2),
                "max_ms": round(max(samples), 2),
            }
            for endpoint, samples in sorted(snapshot.items())
        }


def add_timing(app: FastAPI, tracker: LatencyTracker) -> None:
    """Time every request: ``X-Response-Time-ms`` header, a log line, and the tracker."""

    @app.middleware("http")
    async def time_request(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        milliseconds = (time.perf_counter() - start) * 1000
        route = request.scope.get("route")
        endpoint = f"{request.method} {route.path if route else 'unmatched'}"
        tracker.record(endpoint, milliseconds)
        response.headers["X-Response-Time-ms"] = f"{milliseconds:.2f}"
        logger.info(
            "request",
            extra={
                "endpoint": endpoint,
                "status": response.status_code,
                "ms": round(milliseconds, 2),
            },
        )
        return response
