import subprocess
import sys
from html.parser import HTMLParser

from recommender.config import PROJECT_ROOT

DOCS = PROJECT_ROOT / "docs"


class _PageIndex(HTMLParser):
    """Collects element ids and local links from one HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.append(attributes["id"])
        link = attributes.get("href") if tag in ("a", "link") else attributes.get("src")
        if link and not link.startswith(("http://", "https://")):
            self.links.append(link)


def _index_pages() -> dict[str, _PageIndex]:
    pages = {}
    for path in sorted(DOCS.glob("*.html")):
        index = _PageIndex()
        index.feed(path.read_text(encoding="utf-8"))
        pages[path.name] = index
    return pages


def test_generated_blocks_are_up_to_date() -> None:
    script = PROJECT_ROOT / "scripts" / "build_docs.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_every_link_between_pages_resolves() -> None:
    pages = _index_pages()
    assert len(pages) >= 6
    for name, page in pages.items():
        for link in page.links:
            target, _, anchor = link.partition("#")
            target = target or name
            assert (DOCS / target).exists(), f"{name}: link to missing file {link}"
            if anchor and target in pages:
                assert anchor in pages[target].ids, f"{name}: link to missing section {link}"


def test_element_ids_are_unique_within_each_page() -> None:
    for name, page in _index_pages().items():
        duplicates = {i for i in page.ids if page.ids.count(i) > 1}
        assert not duplicates, f"{name}: duplicate ids {duplicates}"
