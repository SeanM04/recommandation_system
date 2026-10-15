"""Screenshot every dashboard page, for the README and the docs.

Usage:  python scripts/screenshots.py [--url http://localhost:8501]      (make screenshots)
Needs the dashboard running (make dashboard) and Chrome or Edge installed; Playwright
drives the installed browser, so no browser download is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from recommender.config import PROJECT_ROOT

PAGES = {  # file name -> URL path
    "live": "",
    "comparison": "comparison",
    "beliefs": "beliefs",
    "shift": "shift",
    "try-it": "try_it",
}


def wait_until_drawn(page: Page) -> None:
    """Streamlit draws over a websocket after load: wait for the title, then for charts."""
    page.wait_for_selector("h1", timeout=90_000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=90_000
    )
    page.wait_for_timeout(3000)  # let Plotly finish its layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "docs" / "assets" / "screenshots"
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome")
        except Exception:  # noqa: BLE001 - fall back to Edge, which every Windows machine has
            browser = playwright.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1400, "height": 1500})
        for name, path in PAGES.items():
            page.goto(f"{args.url}/{path}")
            wait_until_drawn(page)
            if name == "try-it":  # show a real app open and one response
                page.get_by_role("button", name="Open the app").click()
                page.get_by_role("button", name="Take up").first.click()
                wait_until_drawn(page)
            page.screenshot(path=args.out / f"{name}.png", full_page=True)
            print(f"saved {(args.out / f'{name}.png').relative_to(PROJECT_ROOT)}")
        browser.close()


if __name__ == "__main__":
    main()
