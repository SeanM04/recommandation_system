"""Request timing: a response header, a JSON log line, and percentiles for /metrics.

Written as plain ASGI middleware rather than FastAPI's ``BaseHTTPMiddleware``, which
wraps every request in extra tasks and streams and costs noticeably more per request.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque

import numpy as np
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("recommender.api")


class LatencyTracker:
    """Keeps the most recent request durations per endpoint (per server process)."""

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


class TimingMiddleware:
    """Times every HTTP request: ``X-Response-Time-ms`` header, a log line, the tracker."""

    def __init__(self, app: ASGIApp, tracker: LatencyTracker) -> None:
        self.app = app
        self.tracker = tracker

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = 500

        async def send_with_timing(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                elapsed = (time.perf_counter() - start) * 1000
                MutableHeaders(scope=message).append("X-Response-Time-ms", f"{elapsed:.2f}")
            await send(message)

        try:
            await self.app(scope, receive, send_with_timing)
        finally:
            milliseconds = (time.perf_counter() - start) * 1000
            route = scope.get("route")  # set by the router once the path has matched
            endpoint = f"{scope['method']} {route.path if route else 'unmatched'}"
            self.tracker.record(endpoint, milliseconds)
            logger.info(
                "request",
                extra={"endpoint": endpoint, "status": status, "ms": round(milliseconds, 2)},
            )
