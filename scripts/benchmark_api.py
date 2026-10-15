"""Measure API latency end to end: a real server, the real database, real HTTP.

Starts uvicorn on a spare port against the configured database. On Postgres it works
inside a throwaway ``benchmark`` schema, so the real tables are never touched. It sends
app opens, each followed by a response for every offer shown, then prints client-side
and server-side p50/p95 per endpoint, stops the server and drops the schema.

Usage:  python scripts/benchmark_api.py [--opens 1000]      (make benchmark)
Needs data/customers.csv (make data): the server loads it into the empty schema.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
from sqlalchemy import text
from sqlalchemy.engine import make_url

from recommender.config import PROJECT_ROOT, load_settings
from recommender.db import make_engine

SCHEMA = "benchmark"
TARGET_P95_MS = 50.0


def spare_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def benchmark_database() -> tuple[str, callable]:
    """A database URL for the server, and a function that cleans it up afterwards."""
    url = load_settings().database_url
    if not url.startswith("postgresql"):
        path = Path(tempfile.mkdtemp()) / "benchmark.db"
        return f"sqlite:///{path.as_posix()}", lambda: None

    engine = make_engine(url)
    with engine.begin() as connection:
        connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        connection.execute(text(f"CREATE SCHEMA {SCHEMA}"))

    def drop_schema() -> None:
        with engine.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        engine.dispose()

    scoped = make_url(url).update_query_dict({"options": f"-csearch_path={SCHEMA}"})
    return scoped.render_as_string(hide_password=False), drop_schema


def wait_until_healthy(base: str, timeout_s: float = 60) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base}/health", timeout=2)
            if response.ok:
                return response.json()
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s")


def percentile_row(samples: list[float]) -> dict[str, float]:
    return {
        "count": len(samples),
        "p50_ms": round(float(np.percentile(samples, 50)), 2),
        "p95_ms": round(float(np.percentile(samples, 95)), 2),
    }


def run_load(base: str, opens: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    session = requests.Session()
    customers = [c["customer_id"] for c in session.get(f"{base}/customers?limit=500").json()]
    timings: dict[str, list[float]] = {"POST /invocations": [], "POST /response": []}

    def timed(endpoint: str, payload: dict, measure: bool) -> dict:
        start = time.perf_counter()
        response = session.post(f"{base}{endpoint.split()[1]}", json=payload, timeout=10)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        if measure:
            timings[endpoint].append(elapsed)
        return response.json()

    warm_up = 20
    for i in range(warm_up + opens):
        measure = i >= warm_up
        customer_id = customers[rng.integers(len(customers))]
        invocation = timed("POST /invocations", {"customer_id": customer_id}, measure)
        for offer in invocation["offers"]:
            outcome = rng.choice(["taken_up", "viewed", "ignored"], p=[0.1, 0.6, 0.3])
            payload = {
                "invocation_id": invocation["invocation_id"],
                "offer_id": offer["offer_id"],
                "outcome": str(outcome),
            }
            timed("POST /response", payload, measure)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--opens", type=int, default=1000, help="app opens to send")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    database_url, cleanup = benchmark_database()
    port = spare_port()
    base = f"http://127.0.0.1:{port}"
    log_path = PROJECT_ROOT / "results" / "benchmark_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"RECOMMENDER_DATABASE_URL": database_url}
    command = [sys.executable, "-m", "uvicorn", "recommender.api.main:app",
               "--port", str(port), "--no-access-log", "--log-level", "warning"]  # fmt: skip

    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            health = wait_until_healthy(base)
            print(f"Server up on {base} ({health['database']}, policy {health['active_policy']})")
            client = run_load(base, args.opens, args.seed)
            server_side = requests.get(f"{base}/metrics", timeout=10).json()["latency"]
        finally:
            server.terminate()
            server.wait(timeout=10)
            cleanup()

    rows = []
    for endpoint, samples in client.items():
        rows.append(
            {
                "endpoint": endpoint,
                **{f"client_{k}": v for k, v in percentile_row(samples).items()},
                "server_p50_ms": server_side[endpoint]["p50_ms"],
                "server_p95_ms": server_side[endpoint]["p95_ms"],
            }
        )
    print(f"\n{args.opens:,} app opens on {health['database']}, after 20 warm-up opens\n")
    print(f"{'endpoint':<20}{'requests':>9}{'client p50':>12}{'client p95':>12}"
          f"{'server p50':>12}{'server p95':>12}")  # fmt: skip
    for row in rows:
        print(
            f"{row['endpoint']:<20}{row['client_count']:>9,}{row['client_p50_ms']:>10.1f}ms"
            f"{row['client_p95_ms']:>10.1f}ms{row['server_p50_ms']:>10.1f}ms"
            f"{row['server_p95_ms']:>10.1f}ms"
        )
    worst = max(row["client_p95_ms"] for row in rows)
    verdict = "met" if worst < TARGET_P95_MS else "missed"
    print(f"\nTarget p95 < {TARGET_P95_MS:.0f} ms: {verdict} (worst client p95 {worst:.1f} ms)")

    out = PROJECT_ROOT / "results" / "benchmark_api.json"
    out.write_text(
        json.dumps(
            {"opens": args.opens, "database": health["database"], "endpoints": rows}, indent=2
        )
    )
    print(f"Saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
