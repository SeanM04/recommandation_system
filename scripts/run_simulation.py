"""Simulate customers using the app: fast mode, or through the real API.

Usage:
    python scripts/run_simulation.py                    fast mode: 50,000 app opens (make simulate)
    python scripts/run_simulation.py --mode api         every app open goes through the
                                                        running API (make run first)
    python scripts/run_simulation.py --mode api --in-process   the same, no server needed
    python scripts/run_simulation.py --policy static --rounds 10000 --no-shift

Fast mode calls the policy directly; --persist then copies what it learned into the
database. API mode sends real POST /invocations and POST /response calls, so the event log
and options store in the database fill up exactly as with live traffic. Either way the
behaviour shift happens halfway, and the run is scored against the oracle.
"""

from __future__ import annotations

import argparse
import time

import requests

from recommender.algorithms.factory import build_policy
from recommender.algorithms.static import StaticPolicy
from recommender.config import PROJECT_ROOT, api_client_headers, load_settings
from recommender.db import make_engine
from recommender.evaluation.compare import warm_up_history
from recommender.evaluation.metrics import summarize
from recommender.simulator.engine import World, simulate, simulate_via_api
from recommender.store.options_store import InMemoryOptionsStore
from recommender.store.sql import SqlOptionsStore, copy_options

DEFAULT_ROUNDS = {"fast": 50_000, "api": 5_000}  # API mode makes ~4 HTTP calls per app open


def main() -> None:
    """Simulate app opens through the fast loop or the API and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["fast", "api"], default="fast")
    parser.add_argument("--policy", help="a policy from settings.yaml (default: the active one)")
    parser.add_argument("--rounds", type=int, help="app opens (fast: 50,000; api: 5,000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shift", action="store_true", help="keep behaviour stable")
    parser.add_argument("--url", default="http://localhost:8000", help="API mode: the API")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="API mode without a server: run the API app inside this process",
    )
    parser.add_argument("--persist", action="store_true", help="fast mode: save the model")
    args = parser.parse_args()

    settings = load_settings()
    world = World.load()
    name = args.policy or settings.recommendation.active_policy
    if name not in settings.policies:
        parser.error(f"unknown policy {name!r}; configured: {list(settings.policies)}")
    rounds = args.rounds or DEFAULT_ROUNDS[args.mode]
    shift_round = None if args.no_shift else rounds // 2
    n_offers = settings.recommendation.number_of_offers
    print(f"{args.mode} mode, policy {name}, {rounds:,} app opens, shift at {shift_round}")

    start = time.perf_counter()
    if args.mode == "fast":
        store = InMemoryOptionsStore()
        popularity = StaticPolicy.fit(warm_up_history(world, 5000, args.seed)).popularity
        policy = build_policy(
            name, settings.policies[name], store, seed=args.seed, popularity=popularity
        )
        reward = settings.reward_function(name, world.offers)
        log = simulate(
            policy,
            world,
            rounds=rounds,
            shift_round=shift_round,
            n_offers=n_offers,
            reward=reward,
            seed=args.seed,
        )
        if args.persist:
            copied = copy_options(store, SqlOptionsStore(make_engine(settings.database_url)))
            print(f"Saved {copied} options rows for {name} to the database")
    else:
        if args.in_process:
            # The real FastAPI app (routing, validation, event log, learning), minus the network.
            from fastapi.testclient import TestClient

            from recommender.api.main import create_app

            client_context, base = TestClient(create_app(), headers=api_client_headers()), ""
        else:
            client_context, base = requests.Session(), args.url
            client_context.headers.update(api_client_headers())

        def progress(done: int) -> None:
            """Print the rate and time left every 500 app opens."""
            if done % 500 == 0 or done == rounds:
                rate = done / (time.perf_counter() - start)
                print(f"  {done:,} app opens ({rate:.0f}/s, {(rounds - done) / rate:.0f}s left)")

        with client_context as client:
            health = client.get(f"{base}/health").json()
            where = "in-process" if args.in_process else base
            print(f"API ({where}): {health['database']}, serving {health['active_policy']}")
            log = simulate_via_api(
                client,
                world,
                rounds=rounds,
                shift_round=shift_round,
                n_offers=n_offers,
                seed=args.seed,
                policy=name,
                base_url=base,
                progress=progress,
            )

    elapsed = time.perf_counter() - start
    out = PROJECT_ROOT / "results" / f"simulation_{args.mode}_{name}.parquet"
    out.parent.mkdir(exist_ok=True)
    log.to_parquet(out, index=False)
    print(f"\nDone in {elapsed:.0f}s ({rounds / elapsed:,.0f} app opens per second)")
    for metric, value in summarize(log).items():
        print(f"  {metric:<24} {value:,.3f}")
    print(f"Per-app-open log: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
