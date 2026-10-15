import pytest
from streamlit.testing.v1 import AppTest

from recommender.config import PROJECT_ROOT

DASHBOARD = PROJECT_ROOT / "dashboard"
PAGES = ["live", "comparison", "beliefs", "shift", "try_it"]


@pytest.fixture(autouse=True)
def dashboard_on_path(monkeypatch) -> None:
    """Streamlit puts the main script's folder on the path; do the same for single pages."""
    monkeypatch.syspath_prepend(str(DASHBOARD))


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page: str) -> None:
    """Each page runs top to bottom with whatever data exists: results or none, events or none."""
    app = AppTest.from_file(str(DASHBOARD / "views" / f"{page}.py"), default_timeout=120).run()
    assert not app.exception, [error.value for error in app.exception]
    assert app.title, "every page has a title"
