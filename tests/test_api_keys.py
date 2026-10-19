"""The API with RECOMMENDER_REQUIRE_API_KEY on: every endpoint but /health needs a key."""

import pytest
from fastapi.testclient import TestClient

from recommender.accounts import Accounts
from recommender.api.main import create_app
from recommender.customers import CustomerRepository
from recommender.db import make_engine
from recommender.simulator.engine import World


@pytest.fixture(scope="module")
def secured(tmp_path_factory):
    """The API with keys required, on a throwaway database, and one valid key."""
    url = f"sqlite:///{(tmp_path_factory.mktemp('keys') / 'keys.db').as_posix()}"
    engine = make_engine(url)
    CustomerRepository(engine).load(World.load().customers)
    key = Accounts(engine).create_api_key("tests")
    engine.dispose()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RECOMMENDER_REQUIRE_API_KEY", "true")
        with TestClient(create_app(database_url=url)) as client:
            yield client, key


def test_calls_without_a_valid_key_are_refused(secured) -> None:
    client, _ = secured
    for headers in ({}, {"X-API-Key": "rk_not-a-real-key"}):
        response = client.post("/invocations", json={"customer_id": "C00001"}, headers=headers)
        assert response.status_code == 401
        assert "X-API-Key" in response.json()["detail"]
    assert client.get("/metrics").status_code == 401


def test_a_valid_key_is_accepted_and_health_needs_none(secured) -> None:
    client, key = secured
    response = client.post(
        "/invocations", json={"customer_id": "C00001"}, headers={"X-API-Key": key}
    )
    assert response.status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_keys_are_off_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RECOMMENDER_REQUIRE_API_KEY", raising=False)
    url = f"sqlite:///{(tmp_path / 'open.db').as_posix()}"
    with TestClient(create_app(database_url=url)) as client:
        assert client.get("/offers").status_code == 200
