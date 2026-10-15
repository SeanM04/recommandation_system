import uuid
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from conftest import postgres_schema
from recommender.api.main import create_app
from recommender.api.schemas import Channel, Outcome, TimeOfDay
from recommender.config import load_offers
from recommender.customers import CustomerRepository
from recommender.db import make_engine
from recommender.domain import CHANNELS, OUTCOMES, TIMES_OF_DAY
from recommender.simulator.engine import World

OFFERS = {offer.offer_id: offer for offer in load_offers()}


def _client_for(url: str):
    engine = make_engine(url)
    CustomerRepository(engine).load(World.load().customers)
    engine.dispose()
    with TestClient(create_app(database_url=url)) as test_client:
        yield test_client


@pytest.fixture(scope="module", params=["sqlite", "postgres"])
def client(request, tmp_path_factory):
    """The API on a throwaway database holding the synthetic customers: a SQLite file,
    and a Postgres schema when Postgres is configured (the production database)."""
    if request.param == "sqlite":
        yield from _client_for(
            f"sqlite:///{(tmp_path_factory.mktemp('api') / 'api.db').as_posix()}"
        )
    else:
        with postgres_schema("pytest_api") as url:
            yield from _client_for(url)


def invoke(client: TestClient, customer_id="C00001", policy=None, **body) -> dict:
    params = {"policy": policy} if policy else None
    response = client.post("/invocations", json={"customer_id": customer_id, **body}, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def respond(client: TestClient, invocation_id: str, offer_id: str, outcome: str):
    return client.post(
        "/response",
        json={"invocation_id": invocation_id, "offer_id": offer_id, "outcome": outcome},
    )


# --- basics ---------------------------------------------------------------------------


def test_health_reports_the_database(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"] in ("sqlite", "postgresql")
    assert body["database_ok"] is True
    assert body["active_policy"] == "thompson_segment_decay"


def test_every_response_carries_its_timing(client: TestClient) -> None:
    response = client.get("/health")
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_offers_and_customers(client: TestClient) -> None:
    assert len(client.get("/offers").json()) == len(OFFERS)
    students = client.get("/customers", params={"segment": "student", "limit": 5}).json()
    assert len(students) == 5
    assert {c["segment"] for c in students} == {"student"}
    assert client.get("/customers", params={"segment": "aliens"}).status_code == 404


def test_schema_choices_match_the_domain() -> None:
    assert set(get_args(Channel)) == set(CHANNELS)
    assert set(get_args(TimeOfDay)) == set(TIMES_OF_DAY)
    assert set(get_args(Outcome)) == set(OUTCOMES)


# --- invocations ----------------------------------------------------------------------


def test_invocation_returns_ranked_eligible_offers(client: TestClient) -> None:
    body = invoke(client, "C00003", number_of_offers=3)  # C00003 is a pensioner
    uuid.UUID(body["invocation_id"])
    assert body["policy"] == "thompson_segment_decay"
    assert body["context"]["segment"] == "pensioner"
    offers = body["offers"]
    assert [o["rank"] for o in offers] == [1, 2, 3]
    assert all(OFFERS[o["offer_id"]].is_eligible("pensioner") for o in offers)
    scores = [o["score"] for o in offers]
    assert scores == sorted(scores, reverse=True)


def test_default_number_of_offers_comes_from_settings(client: TestClient) -> None:
    assert len(invoke(client)["offers"]) == 3


def test_context_overrides(client: TestClient) -> None:
    body = invoke(client, context={"day": "2026-09-26", "time_of_day": "night"})
    assert body["context"]["day_of_week"] == "saturday"
    assert body["context"]["is_payday_week"] is True  # salaried, 26th
    assert body["context"]["time_of_day"] == "night"
    forced = invoke(client, context={"day": "2026-09-26", "is_payday_week": False})
    assert forced["context"]["is_payday_week"] is False


def test_policy_can_be_chosen_per_request(client: TestClient) -> None:
    assert invoke(client, policy="random")["policy"] == "random"
    assert invoke(client, policy="static")["policy"] == "static"
    response = client.post("/invocations", json={"customer_id": "C00001"}, params={"policy": "x"})
    assert response.status_code == 404
    assert "unknown policy" in response.json()["detail"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"customer_id": "C99999"}, 404),
        ({"customer_id": "C00001", "channel": "fax"}, 422),
        ({"customer_id": "C00001", "number_of_offers": 0}, 422),
        ({"customer_id": "C00001", "number_of_offers": 6}, 422),
        ({"customer_id": "C00001", "context": {"time_of_day": "teatime"}}, 422),
        ({"customer_id": "C00001", "unexpected": True}, 422),
    ],
)
def test_invalid_invocations_get_clear_errors(client: TestClient, body, status) -> None:
    response = client.post("/invocations", json=body)
    assert response.status_code == status
    assert response.json()["detail"]


# --- responses ------------------------------------------------------------------------


def test_response_updates_the_model_immediately(client: TestClient) -> None:
    body = invoke(client, "C00005", policy="thompson_segment")
    offer_id = body["offers"][0]["offer_id"]
    segment = body["context"]["segment"]

    def pulls() -> float:
        state = client.get("/policies/thompson_segment/state", params={"context_key": segment})
        arms = {arm["offer_id"]: arm for arm in state.json()["arms"]}
        return arms.get(offer_id, {}).get("pulls", 0.0)

    before = pulls()
    result = respond(client, body["invocation_id"], offer_id, "taken_up")
    assert result.status_code == 200, result.text
    assert result.json()["policy"] == "thompson_segment"
    assert result.json()["reward"] == 1.0
    assert result.json()["model_updated"] is True
    assert pulls() == before + 1


def test_duplicate_response_is_rejected(client: TestClient) -> None:
    body = invoke(client)
    offer_id = body["offers"][0]["offer_id"]
    assert respond(client, body["invocation_id"], offer_id, "viewed").status_code == 200
    duplicate = respond(client, body["invocation_id"], offer_id, "taken_up")
    assert duplicate.status_code == 409
    assert "already recorded" in duplicate.json()["detail"]


def test_ignored_offers_are_logged_but_do_not_update_the_model(client: TestClient) -> None:
    body = invoke(client)
    result = respond(client, body["invocation_id"], body["offers"][2]["offer_id"], "ignored")
    assert result.status_code == 200
    assert result.json()["reward"] is None
    assert result.json()["model_updated"] is False


def test_baselines_log_responses_without_learning(client: TestClient) -> None:
    body = invoke(client, policy="random")
    result = respond(client, body["invocation_id"], body["offers"][0]["offer_id"], "taken_up")
    assert result.json()["model_updated"] is False


def test_invalid_responses_get_clear_errors(client: TestClient) -> None:
    body = invoke(client, "C00003")  # a pensioner: data_20gb is never shown
    shown = body["offers"][0]["offer_id"]
    assert respond(client, str(uuid.uuid4()), shown, "viewed").status_code == 404
    assert respond(client, body["invocation_id"], "no_such_offer", "viewed").status_code == 404
    not_shown = respond(client, body["invocation_id"], "data_20gb", "viewed")
    assert not_shown.status_code == 422
    assert "was not shown" in not_shown.json()["detail"]
    assert respond(client, body["invocation_id"], shown, "clicked").status_code == 422


# --- policies and metrics -------------------------------------------------------------


def test_policies_list_marks_the_active_one(client: TestClient) -> None:
    policies = {p["name"]: p for p in client.get("/policies").json()}
    assert policies["thompson_segment_decay"]["active"] is True
    assert policies["random"]["learns"] is False
    assert policies["thompson_segment"]["context"] == "segment"
    assert client.get("/policies/nope/state").status_code == 404


def test_static_policy_state_shows_its_frozen_table(client: TestClient) -> None:
    state = client.get("/policies/static/state").json()
    assert state["learns"] is False
    assert state["arms"] == []
    assert isinstance(state["popularity"], dict)


def test_metrics_count_events_and_report_latency(client: TestClient) -> None:
    body = invoke(client, policy="thompson")
    respond(client, body["invocation_id"], body["offers"][0]["offer_id"], "taken_up")
    metrics = client.get("/metrics").json()
    thompson = metrics["policies"]["thompson"]
    assert thompson["invocations"] >= 1
    assert thompson["impressions"] >= 3
    assert thompson["responses"]["taken_up"] >= 1
    assert thompson["revenue_zar"] > 0
    assert 0 < thompson["take_up_rate"] <= 1
    latency = metrics["latency"]["POST /invocations"]
    assert latency["count"] >= 1
    assert latency["p50_ms"] <= latency["p95_ms"] <= latency["max_ms"]
