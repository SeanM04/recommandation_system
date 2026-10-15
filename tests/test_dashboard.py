"""The dashboard, page by page, with Streamlit's test harness.

Every test runs against a throwaway database and results folder, never the real ones:
a SQLite file holding the customers and a few app opens sent through the API, and a small
comparison written exactly as `make compare` writes it.
"""

import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from recommender.api.main import create_app
from recommender.config import PROJECT_ROOT, load_settings
from recommender.customers import CustomerRepository
from recommender.db import make_engine
from recommender.evaluation.compare import run_comparison, save_results
from recommender.simulator.engine import World, simulate_via_api

DASHBOARD = PROJECT_ROOT / "dashboard"
PAGES = ["live", "comparison", "beliefs", "shift", "try_it"]
APP_OPENS = 40


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("dashboard")
    world = World.load()
    url = f"sqlite:///{(root / 'dashboard.db').as_posix()}"
    engine = make_engine(url)
    CustomerRepository(engine).load(world.customers)
    engine.dispose()
    with TestClient(create_app(database_url=url)) as client:
        simulate_via_api(client, world, rounds=APP_OPENS, seed=0)
    comparison = run_comparison(
        load_settings(),
        world,
        rounds=600,
        shift_round=300,
        seeds=[0, 1],
        warm_up_rounds=200,
        snapshot_every=300,
        progress=lambda _: None,
    )
    save_results(comparison, world, 300, root / "results")
    (root / "empty").mkdir()
    return {"url": url, "results": root / "results", "empty": root / "empty"}


@pytest.fixture
def dashboard(monkeypatch, sandbox) -> dict:
    """Point the dashboard at the sandbox and start every test with empty caches."""
    monkeypatch.syspath_prepend(str(DASHBOARD))  # as Streamlit does for the main script
    monkeypatch.setenv("RECOMMENDER_DATABASE_URL", sandbox["url"])
    monkeypatch.setenv("RECOMMENDER_RESULTS", str(sandbox["results"]))
    monkeypatch.setenv("RECOMMENDER_API_URL", "http://127.0.0.1:9")  # nothing there: in-process
    st.cache_data.clear()
    st.cache_resource.clear()
    yield sandbox
    st.cache_data.clear()
    st.cache_resource.clear()


def page(name: str) -> AppTest:
    return AppTest.from_file(str(DASHBOARD / "views" / f"{name}.py"), default_timeout=120)


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders(dashboard, name: str) -> None:
    app = page(name).run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.title


@pytest.mark.parametrize("name", ["comparison", "beliefs", "shift"])
def test_result_pages_explain_how_to_get_results(dashboard, monkeypatch, name: str) -> None:
    monkeypatch.setenv("RECOMMENDER_RESULTS", str(dashboard["empty"]))
    app = page(name).run()
    assert not app.exception
    assert "make simulate" in app.info[0].value


def test_live_view_counts_the_logged_app_opens(dashboard) -> None:
    app = page("live").run()
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["App opens logged"] == f"{APP_OPENS:,}"


def test_comparison_lists_every_policy_and_says_which_run(dashboard) -> None:
    app = page("comparison").run()
    assert set(app.dataframe[0].value.index) == set(load_settings().policies)
    assert any("2 seeds × 600 app opens" in caption.value for caption in app.caption)


def test_try_it_updates_the_model(dashboard) -> None:
    """Open the app as a customer, take up the first offer, see the belief move."""
    app = page("try_it").run()
    next(button for button in app.button if button.label == "Open the app").click().run()
    assert len([button for button in app.button if button.label == "Take up"]) == 3
    next(button for button in app.button if button.label == "Take up").click().run()
    assert not app.exception, [error.value for error in app.exception]
    assert "reward 1.00" in app.success[0].value
    assert "mean take-up" in app.success[0].value
