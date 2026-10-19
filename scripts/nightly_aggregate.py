"""Roll the event log up into ``daily_offer_stats``: one row per day, policy, segment and offer.

Meant to run once a night, shortly after midnight, e.g. from cron:
    5 0 * * *  cd /app && python scripts/nightly_aggregate.py
Rerunning a day replaces it, so a failed or late run is fixed by running it again.

Usage:  python scripts/nightly_aggregate.py                 yesterday (make nightly)
        python scripts/nightly_aggregate.py --day 2026-10-14
        python scripts/nightly_aggregate.py --all           every day in the log
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from recommender.config import load_offers, load_settings
from recommender.db import make_engine
from recommender.events.daily import aggregate_day, logged_days


def main() -> None:
    """Roll up yesterday, a given day or every day, and print a summary per policy."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", type=date.fromisoformat, help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--all", action="store_true", help="every day from the first app open")
    args = parser.parse_args()

    engine = make_engine(settings.database_url)
    if args.all:
        days = logged_days(engine, settings.timezone)
    else:
        today = datetime.now(ZoneInfo(settings.timezone)).date()
        days = [args.day or today - timedelta(days=1)]
    margins = {offer.offer_id: offer.margin_zar for offer in load_offers(settings.paths.offers)}

    for day in days:
        rows = aggregate_day(engine, day, settings.timezone, margins)
        if not rows:
            print(f"{day}: no app opens logged")
            continue
        frame = pd.DataFrame(rows)
        by_policy = frame.groupby("policy")[["shown_first", "impressions", "taken_up"]].sum()
        by_policy = by_policy.rename(columns={"shown_first": "app_opens"})
        by_policy["take_up_per_open"] = (by_policy["taken_up"] / by_policy["app_opens"]).round(3)
        print(f"{day}: {len(rows)} rows ({engine.dialect.name}, daily_offer_stats)")
        print(by_policy.to_string(), end="\n\n")


if __name__ == "__main__":
    main()
