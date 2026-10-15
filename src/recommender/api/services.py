"""Everything a request needs, wired once at startup."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import Engine, text

from recommender.algorithms.base import Policy
from recommender.algorithms.factory import build_policies
from recommender.algorithms.rewards import RewardFn, make_reward
from recommender.algorithms.static import StaticPolicy
from recommender.api.schemas import ContextOverrides
from recommender.config import Settings, load_offers
from recommender.customers import CustomerRepository, read_customers
from recommender.db import create_tables, make_engine
from recommender.domain import Context, Customer, Offer, time_of_day_for_hour
from recommender.events.log import EventLog
from recommender.store.sql import SqlOptionsStore

logger = logging.getLogger("recommender.api")


@dataclass
class Services:
    settings: Settings
    engine: Engine
    offers: dict[str, Offer]
    customers: CustomerRepository
    events: EventLog
    store: SqlOptionsStore
    policies: dict[str, Policy]
    reward: RewardFn
    timezone: ZoneInfo
    rank_locks: dict[str, threading.Lock]  # numpy generators are not thread-safe

    @classmethod
    def build(cls, settings: Settings, database_url: str) -> Services:
        engine = make_engine(database_url)
        create_tables(engine)
        offers = {offer.offer_id: offer for offer in load_offers(settings.paths.offers)}

        customers = CustomerRepository(engine)
        if customers.count() == 0 and settings.paths.customers.exists():
            loaded = customers.load(read_customers(settings.paths.customers))
            logger.info("customers loaded from csv", extra={"customers": loaded})

        events = EventLog(engine)
        store = SqlOptionsStore(engine)
        # The static baseline is "batch-trained" on everything logged so far, then frozen.
        popularity = StaticPolicy.fit(events.reward_history()).popularity
        policies = build_policies(settings, store, popularity=popularity)
        reward = make_reward(
            settings.recommendation.reward,
            list(offers.values()),
            learn_from_ignored=settings.recommendation.learn_from_ignored,
        )
        logger.info(
            "api ready",
            extra={
                "database": engine.dialect.name,
                "customers": customers.count(),
                "policies": list(policies),
                "active_policy": settings.recommendation.active_policy,
            },
        )
        return cls(
            settings=settings,
            engine=engine,
            offers=offers,
            customers=customers,
            events=events,
            store=store,
            policies=policies,
            reward=reward,
            timezone=ZoneInfo(settings.timezone),
            rank_locks={name: threading.Lock() for name in policies},
        )

    def context_for(
        self, customer: Customer, channel: str, overrides: ContextOverrides | None
    ) -> Context:
        """The customer's profile plus the moment: now in the configured timezone, unless
        the request overrides the day, the time of day or the payday flag."""
        overrides = overrides or ContextOverrides()
        now = datetime.now(self.timezone)
        context = Context.build(
            customer,
            day=overrides.day or now.date(),
            time_of_day=overrides.time_of_day or time_of_day_for_hour(now.hour),
            channel=channel,
        )
        if overrides.is_payday_week is not None:
            context = replace(context, is_payday_week=overrides.is_payday_week)
        return context

    def database_ok(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:  # noqa: BLE001 - any failure means "not healthy"
            return False
        return True
