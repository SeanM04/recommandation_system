"""The dashboard, page by page, with Streamlit's test harness.

Every test runs against a throwaway database and results folder, never the real ones:
a SQLite file holding the customers and a few app opens sent through the API, and a small
comparison written exactly as `make compare` writes it.
"""

import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from recommender.accounts import Accounts, User
from recommender.api.main import create_app
from recommender.config import PROJECT_ROOT, load_settings
from recommender.customers import CustomerRepository
from recommender.db import make_engine
from recommender.evaluation.compare import run_comparison, save_results
from recommender.simulator.engine import World, simulate_via_api

DASHBOARD = PROJECT_ROOT / "dashboard"
PAGES = ["live", "comparison", "beliefs", "shift", "try_it", "users"]
ADMIN = User("admin", "admin")
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


def page(name: str, user: User | None = ADMIN) -> AppTest:
    """One page, run directly (as the navigation runs it) with ``user`` signed in."""
    app = AppTest.from_file(str(DASHBOARD / "views" / f"{name}.py"), default_timeout=120)
    if user is not None:
        app.session_state["user"] = user
    return app


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


# --- signing in -------------------------------------------------------------------------------


@pytest.fixture
def fresh(dashboard, monkeypatch, tmp_path) -> str:
    """The dashboard on an empty database: no accounts, no events."""
    url = f"sqlite:///{(tmp_path / 'fresh.db').as_posix()}"
    monkeypatch.setenv("RECOMMENDER_DATABASE_URL", url)
    return url


def whole_app() -> AppTest:
    return AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=120)


def sign_in(app: AppTest, username: str, password: str) -> AppTest:
    app.text_input[0].input(username)
    app.text_input[1].input(password)
    return app.button[0].click().run()


def test_first_run_creates_the_admin_and_signs_in(fresh) -> None:
    app = whole_app().run()
    assert "No accounts yet" in app.info[0].value
    assert "user" not in app.session_state
    app.text_input[1].input("a-long-password")
    app.text_input[2].input("a-long-password")
    app.button[0].click().run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.session_state["user"] == User("admin", "admin")
    assert app.title[0].value == "Live view"
    assert "Signed in as **admin**" in app.sidebar.caption[0].value


def test_sign_in_needs_the_right_password(fresh) -> None:
    Accounts(make_engine(fresh)).create_user("zanele", "her-password", role="viewer")
    app = sign_in(whole_app().run(), "zanele", "not-her-password")
    assert "Wrong username or password" in app.error[0].value
    assert "user" not in app.session_state
    app = sign_in(app, "zanele", "her-password")
    assert app.session_state["user"] == User("zanele", "viewer")
    assert not app.exception


def test_repeated_failures_lock_the_session(fresh) -> None:
    Accounts(make_engine(fresh)).create_user("zanele", "her-password")
    app = whole_app().run()
    for _ in range(5):
        app = sign_in(app, "zanele", "guess-again")
    assert "Too many failed sign-ins" in app.error[0].value


def test_viewers_cannot_use_admin_pages(dashboard) -> None:
    for name in ("try_it", "users"):
        app = page(name, user=User("zanele", "viewer")).run()
        assert "Only admins" in app.error[0].value
        assert not app.title


def test_admins_add_users(fresh) -> None:
    Accounts(make_engine(fresh)).create_user("admin", "admin-password", role="admin")
    app = page("users").run()
    form = app.text_input
    form[0].input("sipho")
    form[1].input("sipho-password")
    next(button for button in app.button if button.label == "Add user").click().run()
    assert not app.exception, [error.value for error in app.exception]
    assert Accounts(make_engine(fresh)).authenticate("sipho", "sipho-password") is not None
