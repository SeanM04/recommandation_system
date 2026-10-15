"""The real-time recommender API.

Run locally with:  make run   (uvicorn recommender.api.main:app)
Interactive docs:  http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status

from recommender import __version__
from recommender.algorithms.static import StaticPolicy
from recommender.api.schemas import (
    ArmState,
    ContextOut,
    CustomerOut,
    InvocationRequest,
    InvocationResponse,
    MetricsResponse,
    OfferOut,
    PolicyInfo,
    PolicyState,
    RankedOffer,
    ResponseRequest,
    ResponseResult,
)
from recommender.api.services import Services
from recommender.api.timing import LatencyTracker, TimingMiddleware
from recommender.config import Settings, load_settings
from recommender.domain import SEGMENTS
from recommender.events.log import DuplicateResponse
from recommender.observability import configure_logging


def create_app(settings: Settings | None = None, database_url: str | None = None) -> FastAPI:
    """Build the app. The database is connected on startup, not at import."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        resolved = settings or load_settings()
        app.state.services = Services.build(resolved, database_url or resolved.database_url)
        yield
        app.state.services.engine.dispose()

    app = FastAPI(
        title="Telco Dynamic Offer Recommender",
        version=__version__,
        description=(
            "Ranks offers for each app open with a multi-armed bandit and learns from every "
            "response immediately."
        ),
        lifespan=lifespan,
    )
    app.state.latency = LatencyTracker()
    app.add_middleware(TimingMiddleware, tracker=app.state.latency)
    register_routes(app)
    return app


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def not_found(message: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail=message)


def register_routes(app: FastAPI) -> None:  # noqa: C901 - a flat list of endpoints
    @app.get("/health", tags=["ops"])
    def health(services: ServicesDep) -> dict:
        """Liveness and readiness: is the database reachable, and what is being served."""
        database_ok = services.database_ok()
        return {
            "status": "ok" if database_ok else "degraded",
            "version": __version__,
            "database": services.engine.dialect.name,
            "database_ok": database_ok,
            "active_policy": services.settings.recommendation.active_policy,
        }

    @app.get("/offers", response_model=list[OfferOut], tags=["catalogue"])
    def list_offers(services: ServicesDep) -> list[OfferOut]:
        """The offer catalogue."""
        return [OfferOut(**offer.model_dump()) for offer in services.offers.values()]

    @app.get("/customers", response_model=list[CustomerOut], tags=["catalogue"])
    def list_customers(
        services: ServicesDep,
        segment: str | None = Query(None, description="Only customers in this segment"),
        limit: int = Query(20, ge=1, le=500),
    ) -> list[CustomerOut]:
        """A few customer ids, for trying the API out."""
        if segment is not None and segment not in SEGMENTS:
            raise not_found(f"unknown segment {segment!r}; expected one of {list(SEGMENTS)}")
        return [
            CustomerOut(customer_id=customer_id, segment=customer_segment)
            for customer_id, customer_segment in services.customers.list_customers(segment, limit)
        ]

    @app.post("/invocations", response_model=InvocationResponse, tags=["recommendations"])
    def invoke(
        body: InvocationRequest,
        services: ServicesDep,
        policy: str | None = Query(
            None, description="Serve this request with a named policy (A/B tests)"
        ),
    ) -> InvocationResponse:
        """Rank offers for one app open, and log the invocation and every offer shown."""
        settings = services.settings.recommendation
        name = policy or settings.active_policy
        if name not in services.policies:
            raise not_found(f"unknown policy {name!r}; configured: {list(services.policies)}")
        n = body.number_of_offers or settings.number_of_offers
        if n > settings.max_offers:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"number_of_offers must be at most {settings.max_offers}",
            )
        customer = services.customers.get(body.customer_id)
        if customer is None:
            raise not_found(f"unknown customer {body.customer_id!r}")

        context = services.context_for(customer, body.channel, body.context)
        eligible = [o for o in services.offers.values() if o.is_eligible(context.segment)]
        with services.rank_locks[name]:
            ranked = services.policies[name].rank(context, eligible, n)
        invocation_id = services.events.record_invocation(
            customer_id=customer.customer_id, policy=name, context=context, ranked=ranked
        )

        return InvocationResponse(
            invocation_id=invocation_id,
            policy=name,
            customer_id=customer.customer_id,
            context=ContextOut(**{k: getattr(context, k) for k in ContextOut.model_fields}),
            offers=[
                RankedOffer(
                    rank=rank,
                    offer_id=scored.offer_id,
                    name=services.offers[scored.offer_id].name,
                    category=services.offers[scored.offer_id].category,
                    price_zar=services.offers[scored.offer_id].price_zar,
                    score=scored.score,
                )
                for rank, scored in enumerate(ranked, start=1)
            ],
        )

    @app.post("/response", response_model=ResponseResult, tags=["recommendations"])
    def respond(body: ResponseRequest, services: ServicesDep) -> ResponseResult:
        """Record what happened to one shown offer, and update the model immediately.

        Idempotent: a second response for the same invocation and offer gets 409.
        """
        invocation = services.events.get_invocation(body.invocation_id)
        if invocation is None:
            raise not_found(f"unknown invocation_id {body.invocation_id!r}")
        offer = services.offers.get(body.offer_id)
        if offer is None:
            raise not_found(f"unknown offer {body.offer_id!r}")
        if body.offer_id not in invocation.shown:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"offer {body.offer_id!r} was not shown in invocation "
                f"{body.invocation_id!r}; shown: {list(invocation.shown)}",
            )

        reward_fn = (
            services.rewards.get(invocation.policy)
            or services.rewards[services.settings.recommendation.active_policy]
        )  # a policy since removed from the config still gets its response logged
        reward = reward_fn(offer, body.outcome)
        try:  # log first: the event log is the record the model can be rebuilt from
            services.events.record_response(
                invocation_id=body.invocation_id,
                offer_id=body.offer_id,
                outcome=body.outcome,
                reward=reward,
                value=body.value,
            )
        except DuplicateResponse as duplicate:
            raise HTTPException(status.HTTP_409_CONFLICT, detail=str(duplicate)) from None

        policy = services.policies.get(invocation.policy)
        model_updated = reward is not None and policy is not None and policy.learns
        if model_updated:
            policy.update(invocation.context, body.offer_id, reward)
        return ResponseResult(
            invocation_id=body.invocation_id,
            offer_id=body.offer_id,
            outcome=body.outcome,
            policy=invocation.policy,
            reward=reward,
            model_updated=model_updated,
        )

    @app.get("/policies", response_model=list[PolicyInfo], tags=["policies"])
    def list_policies(services: ServicesDep) -> list[PolicyInfo]:
        """Every configured policy; any of them can serve a request via ?policy=."""
        active = services.settings.recommendation.active_policy
        return [
            PolicyInfo(
                name=name,
                type=spec.type,
                context=spec.context,
                decay=spec.decay,
                reward=services.settings.reward_for(name),
                learns=services.policies[name].learns,
                active=name == active,
            )
            for name, spec in services.settings.policies.items()
        ]

    @app.get("/policies/{name}/state", response_model=PolicyState, tags=["policies"])
    def policy_state(
        name: str,
        services: ServicesDep,
        context_key: str | None = Query(None, description="Only this context key, e.g. family"),
    ) -> PolicyState:
        """What a policy has learned: its rows in the options store."""
        policy = services.policies.get(name)
        if policy is None:
            raise not_found(f"unknown policy {name!r}; configured: {list(services.policies)}")
        rows = [
            row
            for row in services.store.rows(name)
            if context_key is None or row.context_key == context_key
        ]
        return PolicyState(
            policy=name,
            type=services.settings.policies[name].type,
            learns=policy.learns,
            arms=[
                ArmState(
                    context_key=row.context_key,
                    offer_id=row.offer_id,
                    pulls=row.stats.pulls,
                    total_reward=row.stats.total_reward,
                    alpha=row.stats.alpha,
                    beta=row.stats.beta,
                    mean=row.stats.mean,
                    updated_at=row.updated_at,
                )
                for row in rows
            ],
            popularity=policy.popularity if isinstance(policy, StaticPolicy) else None,
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
    def metrics(request: Request, services: ServicesDep) -> MetricsResponse:
        """Per-policy take-up and revenue from the event log; latency per endpoint."""
        margins = {offer_id: offer.margin_zar for offer_id, offer in services.offers.items()}
        return MetricsResponse(
            policies=services.events.summary(margins),
            latency=request.app.state.latency.summary(),
        )


app = create_app()
