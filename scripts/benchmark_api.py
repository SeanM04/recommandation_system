"""Measure API latency end to end: a real server, the real database, real HTTP.

Runs the API either as a local uvicorn process (one worker) or in the Docker image with
several worker processes (``--docker``), against the configured database. On Postgres it
works inside a throwaway ``benchmark`` schema, so the real tables are never touched.
Or it measures an API that is already running (``--url``), such as the docker compose
stack, where the API and Postgres share one Docker network; that API logs and learns
from the benchmark traffic in its own database.
For each level of concurrency it sends app opens, each followed by a response for every
offer shown, and reports p50/p95 and throughput. Then it stops the server and drops the
schema.

Usage:
    python scripts/benchmark_api.py                               (make benchmark)
    python scripts/benchmark_api.py --docker --workers 4 --clients 1,4,8,16
                                                                  (make benchmark-docker)
    python scripts/benchmark_api.py --url http://localhost:8000 --clients 1,4,8,16
                                                                  (make benchmark-compose)
Needs data/customers.csv (make data); --docker also needs the image (make docker-build).
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from sqlalchemy import text
from sqlalchemy.engine import make_url

from recommender.config import PROJECT_ROOT, load_settings
from recommender.customers import CustomerRepository, read_customers
from recommender.db import make_engine

SCHEMA = "benchmark"
TARGET_P95_MS = 50.0
IMAGE = "telco-recommender:latest"
CONTAINER = "telco-recommender-benchmark"
ENDPOINTS = ("POST /invocations", "POST /response")


def spare_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def benchmark_database() -> tuple[str, callable]:
    """A database URL for the server (with customers loaded), and a cleanup function."""
    settings = load_settings()
    url = settings.database_url
    if url.startswith("postgresql"):
        admin = make_engine(url)
        with admin.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
            connection.execute(text(f"CREATE SCHEMA {SCHEMA}"))

        def cleanup() -> None:
            with admin.begin() as connection:
                connection.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
            admin.dispose()

        url = (
            make_url(url)
            .update_query_dict({"options": f"-csearch_path={SCHEMA}"})
            .render_as_string(hide_password=False)
        )
    else:
        url = f"sqlite:///{(Path(tempfile.mkdtemp()) / 'benchmark.db').as_posix()}"

        def cleanup() -> None:
            return None

    engine = make_engine(url)
    CustomerRepository(engine).load(read_customers(settings.paths.customers))
    engine.dispose()
    return url, cleanup


class LocalServer:
    """uvicorn in a child process: one worker (multi-worker uvicorn is unreliable on Windows)."""

    def __init__(self, database_url: str, port: int) -> None:
        self.log = (PROJECT_ROOT / "results" / "benchmark_server.log").open("w", encoding="utf-8")
        command = [sys.executable, "-m", "uvicorn", "recommender.api.main:app",
                   "--port", str(port), "--no-access-log", "--log-level", "warning"]  # fmt: skip
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.process = subprocess.Popen(
            command,
            env=os.environ | {"RECOMMENDER_DATABASE_URL": database_url},
            stdout=self.log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=15)
        self.log.close()


class RunningServer:
    """An API someone else started and will stop."""

    def stop(self) -> None:
        return None


class DockerServer:
    """The API image with several uvicorn workers, reaching the database on the host."""

    def __init__(self, database_url: str, port: int, workers: int) -> None:
        url = make_url(database_url)
        if url.host in ("localhost", "127.0.0.1"):
            url = url.set(host="host.docker.internal")
        # The URL holds the password: pass it in a temporary env file, not on the command line.
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as env_file:
            env_file.write(f"RECOMMENDER_DATABASE_URL={url.render_as_string(False)}\n")
            env_file.write(f"WEB_CONCURRENCY={workers}\n")
        self.env_path = Path(env_file.name)
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, check=False)
        subprocess.run(
            ["docker", "run", "-d", "--name", CONTAINER, "-p", f"{port}:8000",
             "--env-file", str(self.env_path), IMAGE],
            check=True, capture_output=True,
        )  # fmt: skip

    def stop(self) -> None:
        log = subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True)
        (PROJECT_ROOT / "results" / "benchmark_server.log").write_text(log.stdout + log.stderr)
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, check=False)
        self.env_path.unlink(missing_ok=True)


def wait_until_healthy(base: str, timeout_s: float = 90) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{base}/health", timeout=2)
            if response.ok:
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s")


def one_client(base: str, customers: list[str], opens: int, seed: int) -> dict[str, list[float]]:
    """App opens one after another, each followed by a response per offer shown."""
    rng = np.random.default_rng(seed)
    session = requests.Session()
    timings: dict[str, list[float]] = {endpoint: [] for endpoint in ENDPOINTS}

    def timed(endpoint: str, payload: dict) -> dict:
        start = time.perf_counter()
        response = session.post(f"{base}{endpoint.split()[1]}", json=payload, timeout=30)
        timings[endpoint].append((time.perf_counter() - start) * 1000)
        response.raise_for_status()
        return response.json()

    for _ in range(opens):
        customer_id = customers[rng.integers(len(customers))]
        invocation = timed("POST /invocations", {"customer_id": customer_id})
        for offer in invocation["offers"]:
            outcome = rng.choice(["taken_up", "viewed", "ignored"], p=[0.1, 0.6, 0.3])
            timed(
                "POST /response",
                {
                    "invocation_id": invocation["invocation_id"],
                    "offer_id": offer["offer_id"],
                    "outcome": str(outcome),
                },
            )
    return timings


def run_level(base: str, customers: list[str], clients: int, opens: int) -> dict:
    """``clients`` concurrent clients; returns percentiles per endpoint and throughput."""
    start = time.perf_counter()
    with ThreadPoolExecutor(clients) as pool:
        results = list(
            pool.map(lambda seed: one_client(base, customers, opens, seed), range(clients))
        )
    wall = time.perf_counter() - start
    level: dict = {"clients": clients}
    total = 0
    for endpoint in ENDPOINTS:
        samples = [ms for result in results for ms in result[endpoint]]
        total += len(samples)
        level[endpoint] = {
            "requests": len(samples),
            "p50_ms": round(float(np.percentile(samples, 50)), 1),
            "p95_ms": round(float(np.percentile(samples, 95)), 1),
        }
    level["requests_per_s"] = round(total / wall)
    return level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clients", default="1", help="comma-separated, e.g. 1,4,8")
    parser.add_argument("--opens", type=int, default=300, help="app opens per client")
    parser.add_argument("--docker", action="store_true", help="run the API in Docker")
    parser.add_argument("--workers", type=int, default=4, help="uvicorn workers (Docker)")
    parser.add_argument("--url", help="measure an API that is already running, e.g. compose")
    args = parser.parse_args()
    levels = [int(n) for n in args.clients.split(",")]

    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    if args.url:
        base, server, cleanup = args.url.rstrip("/"), RunningServer(), lambda: None
        where = f"running API at {base}"
    else:
        database_url, cleanup = benchmark_database()
        port = spare_port()
        base = f"http://127.0.0.1:{port}"
        if args.docker:
            server = DockerServer(database_url, port, args.workers)
            where = f"Docker, {args.workers} workers"
        else:
            server = LocalServer(database_url, port)
            where = "local process, 1 worker"
    try:
        health = wait_until_healthy(base)
        print(f"API up ({where}; {health['database']}; policy {health['active_policy']})")
        customers = [c["customer_id"] for c in requests.get(f"{base}/customers?limit=500").json()]
        run_level(base, customers, clients=2, opens=10)  # warm up connections and workers
        results = []
        for clients in levels:
            results.append(run_level(base, customers, clients, args.opens))
            print(f"  {clients} client(s) done", flush=True)
    finally:
        server.stop()
        cleanup()

    print(f"\n{where}, {args.opens} app opens per client, 3 offers each\n")
    print(f"{'clients':>7}  {'invocations p50/p95':>22}  {'responses p50/p95':>20}  {'req/s':>6}")
    for level in results:
        inv, resp = level["POST /invocations"], level["POST /response"]
        print(
            f"{level['clients']:>7}  {inv['p50_ms']:>9.1f} / {inv['p95_ms']:>6.1f} ms"
            f"  {resp['p50_ms']:>7.1f} / {resp['p95_ms']:>6.1f} ms  {level['requests_per_s']:>6}"
        )
    met = [lvl["clients"] for lvl in results
           if max(lvl[e]["p95_ms"] for e in ENDPOINTS) < TARGET_P95_MS]  # fmt: skip
    print(f"\np95 < {TARGET_P95_MS:.0f} ms at: {met or 'no level'} concurrent client(s)")

    name = (
        "benchmark_running.json" if args.url
        else "benchmark_docker.json" if args.docker
        else "benchmark_api.json"
    )  # fmt: skip
    out = PROJECT_ROOT / "results" / name
    payload = {"server": where, "database": health["database"], "levels": results}
    out.write_text(json.dumps(payload, indent=2))
    print(f"Saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
