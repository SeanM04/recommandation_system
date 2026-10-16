"""Screenshot every dashboard page, for the README and the docs.

Usage:  python scripts/screenshots.py [--url http://localhost:8501] [--username admin]
                                                                        (make screenshots)
Needs the dashboard running (make dashboard), an admin account to sign in with (the
password from RECOMMENDER_SCREENSHOT_PASSWORD, or asked for), and Chrome or Edge
installed; Playwright drives the installed browser, so no browser download is needed.
A sign-in lasts for one browser session, so the pages are reached through the sidebar,
not by URL.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from recommender.config import PROJECT_ROOT

PAGES = {  # file name -> the page's name in the sidebar
    "live": "Live view",
    "comparison": "Algorithm comparison",
    "beliefs": "Explore vs exploit",
    "shift": "Behaviour shift",
    "try-it": "Try it",
    "users": "Users and API keys",
}


def wait_until_drawn(page: Page) -> None:
    """Streamlit draws over a websocket after load: wait for the title, then for charts."""
    page.wait_for_selector("h1", timeout=90_000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=90_000
    )
    page.wait_for_timeout(3000)  # let Plotly finish its layout


def sign_in(page: Page, username: str, password: str) -> None:
    """Fill in the sign-in form and wait for the first page behind it."""
    page.get_by_role("textbox", name="Username").fill(username)
    page.get_by_role("textbox", name="Password").fill(password)
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_selector("text=Signed in as", timeout=90_000)


def main() -> None:
    """Sign in, then screenshot each page in turn."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "docs" / "assets" / "screenshots"
    )
    parser.add_argument("--username", default="admin", help="an admin account")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    password = os.environ.get("RECOMMENDER_SCREENSHOT_PASSWORD") or getpass.getpass(
        f"Password for {args.username}: "
    )

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome")
        except Exception:  # noqa: BLE001 - fall back to Edge, which every Windows machine has
            browser = playwright.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1400, "height": 1500})
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_selector("input[type=password]", timeout=90_000)
        page.screenshot(path=args.out / "sign-in.png")
        print("saved sign-in.png")
        sign_in(page, args.username, password)
        for name, title in PAGES.items():
            page.get_by_role("link", name=title).click()
            wait_until_drawn(page)
            page.wait_for_timeout(5000)  # the comparison story computes on first view
            if name == "try-it":  # show a real app open and one response
                page.get_by_role("button", name="Open the app").click()
                page.get_by_role("button", name="Take up").first.click()
                wait_until_drawn(page)
            page.screenshot(path=args.out / f"{name}.png", full_page=True)
            print(f"saved {name}.png")
        browser.close()


if __name__ == "__main__":
    main()
