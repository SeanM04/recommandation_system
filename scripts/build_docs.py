"""Regenerate the data-driven parts of the HTML docs in ``docs/`` from code and config.

Prose and diagrams in ``docs/*.html`` are written by hand. Everything between
``<!-- BEGIN generated:NAME -->`` and ``<!-- END generated:NAME -->`` is rebuilt here
from the source of truth (YAML config, the ground-truth model, DECISIONS.md,
requirements.txt and the test files), so the docs cannot drift from the code. README.md
and WRITEUP.md get the same treatment for their results tables and findings.

Usage:  python scripts/build_docs.py           rewrite the docs in place
        python scripts/build_docs.py --check   exit 1 if any page is out of date
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import re
import sys
import textwrap
from collections.abc import Callable
from functools import cached_property
from html import escape
from pathlib import Path

import pandas as pd

from recommender.algorithms.base import BanditPolicy
from recommender.algorithms.epsilon_greedy import EpsilonGreedyPolicy
from recommender.algorithms.rewards import make_reward
from recommender.algorithms.static import StaticPolicy
from recommender.algorithms.thompson import ThompsonSamplingPolicy
from recommender.config import PROJECT_ROOT, load_offers, load_settings
from recommender.domain import IGNORED, OUTCOMES, SEGMENTS, TAKEN_UP, Context, Customer, Offer
from recommender.simulator.ground_truth import GroundTruth, take_up_table, winners
from recommender.simulator.population import SEGMENT_PROFILES, generate_customers
from recommender.simulator.settings import load_simulation_settings
from recommender.store.options_store import ArmStats, InMemoryOptionsStore
from recommender.store.sql import SqlOptionsStore

DOCS = PROJECT_ROOT / "docs"
BLOCK = re.compile(
    r"(?P<begin><!-- BEGIN generated:(?P<name>[\w-]+) -->)(?P<body>.*?)"
    r"(?P<end><!-- END generated:(?P=name) -->)",
    re.S,
)

PAGES = [
    ("index.html", "Overview"),
    ("architecture.html", "Architecture"),
    ("simulation.html", "Simulated world"),
    ("algorithms.html", "Policies"),
    ("api.html", "API"),
    ("results.html", "Results"),
    ("dashboard.html", "Dashboard"),
    ("workflows.html", "Workflows"),
    ("decisions.html", "Decisions"),
    ("stack.html", "Tools & quality"),
]

# Update the status here when a phase is finished, then run `make docs`.
PHASES = [
    (
        1,
        "Domain & data",
        "built",
        "Offer catalogue, synthetic customers, hidden ground truth and the response model.",
    ),
    (
        2,
        "Algorithms",
        "built",
        "Random, static, epsilon-greedy and Thompson sampling policies, contextual keys, "
        "forgetting, pluggable rewards and the options store.",
    ),
    (
        3,
        "Real-time API",
        "built",
        "/invocations and /response with validation, idempotency, an event log, policy "
        "switching and latency timing.",
    ),
    (
        4,
        "Simulator",
        "built",
        "50,000 app opens through the API or in fast mode, with the behaviour shift halfway.",
    ),
    (
        5,
        "Evaluation & dashboard",
        "built",
        "Take-up, revenue, regret against an oracle and convergence; a five-page dashboard.",
    ),
    (
        6,
        "Quality & delivery",
        "built",
        "docker compose for the whole system, JSON logging, README and WRITEUP, and a "
        "nightly roll-up of the event log.",
    ),
]

# fmt: off
TOOLS = [  # name, package in requirements.txt, role, status, why it was chosen
    ("Python", None, "Language", "built",
     "Readable, typed code; the bandit maths is plain Python and NumPy."),
    ("FastAPI", "fastapi", "Real-time API", "built",
     "Eight endpoints with Pydantic validation and interactive docs generated at /docs."),
    ("Uvicorn", "uvicorn", "API server", "built",
     "The standard ASGI server for FastAPI (`make run`)."),
    ("Pydantic", "pydantic", "Validation", "built",
     "Rejects bad config and offers at load; request schemas in Phase 3."),
    ("PyYAML", "pyyaml", "Configuration", "built",
     "Offers, settings and the hidden truth stay human-readable."),
    ("NumPy", "numpy", "Random draws, maths", "built",
     "Seeded generators make every run reproducible; Beta draws for Thompson sampling."),
    ("pandas", "pandas", "Tables", "built",
     "Customer table, take-up tables and, later, metrics."),
    ("pytest", "pytest", "Tests", "built",
     "Plain asserts and fixtures; every phase ships with tests."),
    ("HTTPX 2", "httpx2", "Test client", "built",
     "Powers FastAPI's TestClient; replaces the deprecated httpx."),
    ("Ruff", "ruff", "Lint and format", "built",
     "One fast tool for style, import order and common bugs."),
    ("Git", None, "Version control", "built",
     "Small, logical commits for each step."),
    ("Make", None, "Task runner", "built",
     "One entry point for every command; each target is a plain command."),
    ("PostgreSQL", None, "Project database", "built",
     "Telco_Dynamic_Recomender (Postgres 18) holds the options store; `make db` sets it up."),
    ("psycopg 3", "psycopg", "Postgres driver", "built",
     "The modern Postgres driver for Python, used through SQLAlchemy."),
    ("SQLAlchemy", "sqlalchemy", "Database access", "built",
     "Core API only: one code path for Postgres and SQLite, upserts via ON CONFLICT."),
    ("SQLite", None, "Fallback database", "built",
     "Used when no Postgres URL is configured, so a clean machine runs with no server."),
    ("python-dotenv", "python-dotenv", "Local secrets", "built",
     "Loads the database URL from a git-ignored `.env` file."),
    ("Requests", "requests", "HTTP client", "built",
     "The latency benchmark (and later the simulator) call the API as an app would."),
    ("Streamlit", "streamlit", "Dashboard", "built",
     "Python-only dashboards: five pages, a live view refreshing every 3 seconds."),
    ("Plotly", "plotly", "Charts", "built",
     "Interactive regret, reward and Beta-distribution charts with hover."),
    ("Playwright", "playwright", "Screenshots", "built",
     "Drives the installed Chrome or Edge to screenshot the dashboard (dev only)."),
    ("Docker", None, "Packaging", "built",
     "The API image: slim, non-root, health-checked, 4 workers; secrets via --env-file."),
    ("Docker Compose", None, "Packaging", "planned",
     "Starts the API, dashboard and a Postgres container with one command (Phase 6)."),
]
# fmt: on

TEST_FILES = {
    "test_api.py": "API: invocations, responses, idempotency, per-request policy, errors, "
    "state and metrics",
    "test_domain.py": "Configuration, offer validation, calendar and context helpers",
    "test_population.py": "Synthetic population: reproducibility, realism, CSV round trip",
    "test_ground_truth.py": "Hidden truth, segment stories, payday, revenue, response "
    "model and the architecture guards",
    "test_docs.py": "These docs: generated blocks are current and every link resolves",
    "test_store.py": "Options store: arm statistics, atomic learning under concurrency, "
    "and identical behaviour in memory, SQLite and Postgres",
    "test_policies.py": "Bandit policies: ranking, learning, context keys, forgetting, convergence",
    "test_rewards.py": "Reward functions and the policy factory",
    "test_simulation.py": "Simulator: oracle and regret, common random numbers, learning, the "
    "comparison and simulation through the API",
    "test_dashboard.py": "Dashboard on a throwaway database and results folder: every page, "
    "content checks and a Try it click-through",
    "test_events.py": "Event log: round trips, simultaneous duplicates, averages and safe "
    "concurrent start-up, on SQLite and Postgres",
}

CHIP_LABELS = {"built": "Built", "planned": "Planned", "proposed": "Proposed"}


def chip(status: str) -> str:
    return f'<span class="chip {status}">{CHIP_LABELS[status]}</span>'


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def md_inline(text: str) -> str:
    """The small subset of Markdown used in DECISIONS.md cells, as HTML."""
    text = escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)


def table(headers: list[str], rows: list[list[str]], cls: str = "", numeric=()) -> str:
    """An HTML table. Cells are already-escaped HTML; ``numeric`` are column indexes."""
    head = "".join(
        f'<th scope="col"{" class=num" if i in numeric else ""}>{h}</th>'
        for i, h in enumerate(headers)
    )
    body = "\n".join(
        "<tr>"
        + "".join(
            f"<td{' class=num' if i in numeric else ''}>{cell}</td>" for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<div class="table-wrap"><table{class_attr}>\n<thead><tr>{head}</tr></thead>\n'
        f"<tbody>\n{body}\n</tbody></table></div>"
    )


class Sources:
    """Everything the docs are generated from, loaded once."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.simulation = load_simulation_settings()
        self.offers: list[Offer] = load_offers()
        self.truth = GroundTruth.from_yaml(self.simulation.ground_truth, self.offers)
        population = self.simulation.population
        frame = generate_customers(
            population.n_customers, population.segment_mix, self.settings.seed
        )
        self.population = frame
        self.customers = [Customer(**record) for record in frame.to_dict("records")]
        self.days = self.simulation.calendar.days()
        self.margins = pd.Series({offer.offer_id: offer.margin_zar for offer in self.offers})

    def _table(self, truth: GroundTruth, by: tuple[str, ...] = ("segment",)) -> pd.DataFrame:
        table = take_up_table(
            truth, self.customers, self.offers, self.days, by=by, seed=self.settings.seed
        )
        return table.reindex(list(SEGMENTS), level=0) if len(by) > 1 else table.reindex(SEGMENTS)

    @cached_property
    def before(self) -> pd.DataFrame:
        return self._table(self.truth)

    @cached_property
    def after(self) -> pd.DataFrame:
        return self._table(self.truth.after_shift())

    @cached_property
    def by_payday(self) -> pd.DataFrame:
        return self._table(self.truth, ("segment", "is_payday_week"))

    @cached_property
    def test_inventory(self) -> list[tuple[str, str, int]]:
        """(file, test name, number of cases) for every test function."""
        inventory = []
        for path in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    inventory.append((path.name, node.name, _parametrized_cases(node)))
        return inventory


def _parametrized_cases(node: ast.FunctionDef) -> int:
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and getattr(decorator.func, "attr", "") == "parametrize"
            and len(decorator.args) > 1
            and isinstance(decorator.args[1], ast.List | ast.Tuple)
        ):
            return len(decorator.args[1].elts)
    return 1


# --- blocks ---------------------------------------------------------------------------


def block_sidenav(src: Sources, page: str) -> str:
    links = "\n".join(
        f'<li><a href="{href}"{" aria-current=page" if href == page else ""}>'
        f"{escape(label)}</a></li>"
        for href, label in PAGES
    )
    phases = "\n".join(
        f"<li><span>{n} · {escape(name)}</span>{chip(status)}</li>" for n, name, status, _ in PHASES
    )
    return (
        '<a class="brand" href="index.html"><b>Telco Dynamic Recommender</b>'
        "<span>Project documentation</span></a>\n"
        f'<div><p class="nav-label">Pages</p>\n<ul class="pages">\n{links}\n</ul></div>\n'
        f'<div class="phase-block"><p class="nav-label">Build status</p>\n'
        f'<ul class="phases">\n{phases}\n</ul></div>\n'
        '<button class="theme-toggle" type="button">Theme: system</button>'
    )


def block_phases(src: Sources, page: str) -> str:
    items = "\n".join(
        f'<li><span class="n">Phase {n}</span><b>{escape(name)}</b>{chip(status)}'
        f"<p>{escape(summary)}</p></li>"
        for n, name, status, summary in PHASES
    )
    return f'<ol class="phase-list">\n{items}\n</ol>'


def block_facts(src: Sources, page: str) -> str:
    cases = sum(n for _, _, n in src.test_inventory)
    facts = [
        (len(src.offers), "offers in the catalogue"),
        (len(SEGMENTS), "customer segments"),
        (f"{len(src.customers):,}", "synthetic customers"),
        (len(src.days), "simulated days"),
        (len(src.truth.effects), "context effects"),
        (cases, "automated tests"),
    ]
    items = "\n".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for value, label in facts)
    return f'<dl class="facts">\n{items}\n</dl>'


def block_offers(src: Sources, page: str) -> str:
    rows = [
        [
            f"<code>{o.offer_id}</code>",
            escape(o.name),
            escape(o.category),
            f"R{o.price_zar:,.0f}",
            f"R{o.margin_zar:,.0f}",
            "all five"
            if len(o.eligible_segments) == len(SEGMENTS)
            else ", ".join(s.replace("_", " ") for s in o.eligible_segments),
        ]
        for o in src.offers
    ]
    headers = ["Offer id", "Offer", "Category", "Price", "Margin", "Shown to"]
    return table(headers, rows, numeric=(3, 4))


def block_segments(src: Sources, page: str) -> str:
    frame = src.population
    mix = src.simulation.population.segment_mix
    rows = []
    for segment in SEGMENTS:
        members = frame[frame["segment"] == segment]
        profile = SEGMENT_PROFILES[segment]
        usual_time = max(profile.time_of_day_mix, key=profile.time_of_day_mix.get)
        devices = ", ".join(
            f"{tier} {share:.0%}" for tier, share in profile.device_tier_mix.items()
        )
        rows.append(
            [
                f"<code>{segment}</code>",
                f"{mix[segment]:.0%}",
                f"{len(members):,}",
                f"R{members['avg_monthly_spend'].median():,.0f}",
                f"{(members['prepaid_vs_contract'] == 'contract').mean():.0%}",
                devices,
                usual_time,
            ]
        )
    headers = ["Segment", "Target share", "Customers", "Median spend / month",
               "On contract", "Handsets", "Most common time"]  # fmt: skip
    return table(headers, rows, numeric=(1, 2, 3, 4))


def _condition(attribute: str, expected: object) -> str:
    if isinstance(expected, list):
        return f"<code>{attribute}</code> in {', '.join(map(str, expected))}"
    value = str(expected).lower() if isinstance(expected, bool) else expected
    return f"<code>{attribute}</code> = {escape(str(value))}"


def block_effects(src: Sources, page: str) -> str:
    rows = [
        [
            escape(effect.name[0].upper() + effect.name[1:]),
            "<br>".join(_condition(k, v) for k, v in effect.when.items()),
            ", ".join(f"{escape(key)}&nbsp;×{m:g}" for key, m in effect.multiply.items()),
        ]
        for effect in src.truth.effects
    ]
    return table(["Effect", "Applies when", "Multiplies take-up of"], rows)


def block_heatmap(src: Sources, page: str) -> str:
    table_ = src.before
    vmax = float(table_.max().max())
    favourites = table_.idxmax(axis=1)
    head = "".join(f'<th scope="col">{s.replace("_", " ")}</th>' for s in SEGMENTS)
    rows = []
    for offer in src.offers:
        cells = []
        for segment in SEGMENTS:
            p = table_.loc[segment, offer.offer_id]
            if pd.isna(p):
                cells.append(
                    f'<td class="na" title="{offer.offer_id} is not shown to {segment}">–</td>'
                )
                continue
            v = p / vmax
            classes = ["cell"]
            if v > 0.55:
                classes.append("hot")
            if favourites[segment] == offer.offer_id:
                classes.append("best")
            cells.append(
                f'<td class="{" ".join(classes)}" style="--v:{v:.3f}" '
                f'title="{segment} · {offer.offer_id}: {pct(p)}">{p * 100:.1f}</td>'
            )
        rows.append(f'<tr><th scope="row">{offer.offer_id}</th>{"".join(cells)}</tr>')
    ticks = "".join(
        f'<span style="left:{tick / vmax * 100:.1f}%">{tick:.0%}</span>'
        for tick in (0.0, 0.05, 0.10, 0.15, 0.20)
        if tick <= vmax
    )
    legend = (
        '<div class="heat-legend">'
        f'<div class="ramp"><div class="ramp-bar"></div><div class="ramp-ticks">{ticks}</div></div>'
        '<span><i class="swatch best"></i>segment favourite</span>'
        '<span><i class="swatch na"></i>not shown to this segment</span></div>'
    )
    return (
        '<div class="table-wrap">\n<table class="heat">\n'
        f'<thead><tr><th scope="col">Take-up %</th>{head}</tr></thead>\n'
        f"<tbody>\n{chr(10).join(rows)}\n</tbody></table>\n{legend}</div>"
    )


def _favourite_cell(row: pd.Series, money: bool = False, changed: bool = False) -> str:
    value = f"R{row.value:.2f}" if money else pct(row.value)
    cls = ' class="changed"' if changed else ""
    return f"<span{cls}><code>{row.best}</code></span><br>{value} <small>×{row.lead:.2f}</small>"


def block_favourites(src: Sources, page: str) -> str:
    take_up = winners(src.before)
    revenue = winners(src.before.mul(src.margins, axis=1))
    payday = winners(src.by_payday)
    after = winners(src.after)
    rows = []
    for segment in SEGMENTS:
        rest, paid = payday.loc[(segment, False)], payday.loc[(segment, True)]
        rows.append(
            [
                escape(segment.replace("_", " ")),
                _favourite_cell(take_up.loc[segment]),
                _favourite_cell(revenue.loc[segment], money=True,
                                changed=revenue.loc[segment].best != take_up.loc[segment].best),
                _favourite_cell(rest),
                _favourite_cell(paid, changed=paid.best != rest.best),
                _favourite_cell(after.loc[segment],
                                changed=after.loc[segment].best != take_up.loc[segment].best),
            ]
        )  # fmt: skip
    headers = ["Segment", "By take-up", "By revenue", "Rest of month", "Payday week",
               "After the shift"]  # fmt: skip
    return table(headers, rows, cls="favourites")


def block_shift(src: Sources, page: str) -> str:
    rows = []
    for segment, overrides in src.truth.shift_base_take_up.items():
        for offer_id, new_base in overrides.items():
            old_base = src.truth.base_take_up[segment][offer_id]
            rows.append(
                [
                    f"<code>{segment}</code>",
                    f"<code>{offer_id}</code>",
                    pct(old_base),
                    pct(new_base),
                    pct(src.before.loc[segment, offer_id]),
                    pct(src.after.loc[segment, offer_id]),
                ]
            )
    headers = ["Segment", "Offer", "Base before", "Base after",
               "Segment take-up before", "Segment take-up after"]  # fmt: skip
    return table(headers, rows, numeric=(2, 3, 4, 5))


EXAMPLE_CONTEXT = Context(
    customer_id="example",
    segment="family",
    device_tier="mid",
    prepaid_vs_contract="contract",
    avg_monthly_spend=420.0,
    province="Gauteng",
    channel="app",
    time_of_day="evening",
    day_of_week="saturday",
    is_payday_week=True,
)
EXAMPLE_OFFER = "funeral_cover"


def block_example(src: Sources, page: str) -> str:
    offer = next(o for o in src.offers if o.offer_id == EXAMPLE_OFFER)
    context = EXAMPLE_CONTEXT
    p = src.truth.base_take_up[context.segment][offer.offer_id]
    rows = [["Base rate for <code>family</code> × <code>funeral_cover</code>", "", pct(p)]]
    idle = []
    for effect in src.truth.effects:
        if not effect.applies_to(context):
            continue
        multiplier = effect.multiplier_for(offer)
        if multiplier == 1.0:
            idle.append(escape(effect.name))
            continue
        p *= multiplier
        rows.append([escape(effect.name[0].upper() + effect.name[1:]), f"×{multiplier:g}", pct(p)])
    final = src.truth.take_up_probability(context, offer)
    rows.append(
        [
            f"Cap at {src.truth.max_probability:.0%}, final probability",
            "",
            f"<strong>{pct(final)}</strong>",
        ]
    )
    note = (
        f'<p class="source">Also matched this context but do not touch {offer.category}: '
        f"{', '.join(idle)}.</p>"
        if idle
        else ""
    )
    return table(["Step", "Factor", "Probability"], rows, numeric=(1, 2)) + "\n" + note


def block_attention(src: Sources, page: str) -> str:
    headers = ["Position in the list"] + [
        f"#{k}" for k in range(1, src.truth.response.max_offers + 1)
    ]
    row = ["Chance the customer notices it"] + [f"{a:.0%}" for a in src.truth.response.attention]
    return table(headers, [row], numeric=tuple(range(1, len(headers))))


def block_decision_log(src: Sources, page: str) -> str:
    sections: list[tuple[str, list[list[str]]]] = []
    for line in (PROJECT_ROOT / "DECISIONS.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif line.startswith("|") and sections:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0] in ("#", "") or set(cells[0]) <= set("-: "):
                continue
            sections[-1][1].append([md_inline(cell) for cell in cells])
    parts = []
    for title, rows in sections:
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        parts.append(f'<h3 id="log-{anchor}">{escape(title)}</h3>')
        parts.append(table(["#", "Decision", "Why"], rows, cls="decision-log"))
    return "\n".join(parts)


def _requirement_versions() -> dict[str, str]:
    versions = {}
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            versions[re.sub(r"\[.*\]", "", name).strip().lower()] = version.strip()
    return versions


def block_tools(src: Sources, page: str) -> str:
    versions = _requirement_versions()
    rows = [
        [
            f"<strong>{escape(name)}</strong>",
            escape(role),
            versions.get(package, "3.11+") if package or name == "Python" else "–",
            md_inline(why),
            chip(status),
        ]
        for name, package, role, status, why in TOOLS
    ]
    return table(["Tool", "Role", "Version", "Why this tool", "Status"], rows)


def block_tests(src: Sources, page: str) -> str:
    rows = []
    current = None
    for file, name, cases in src.test_inventory:
        if file != current:
            current = file
            note = escape(TEST_FILES.get(file, ""))
            rows.append(
                f'<tr><th colspan="2" scope="rowgroup"><code>tests/{file}</code> · {note}</th></tr>'
            )
        label = name.removeprefix("test_").replace("_", " ")
        rows.append(
            f'<tr><td>{escape(label[0].upper() + label[1:])}</td><td class="num">{cases}</td></tr>'
        )
    total = sum(n for _, _, n in src.test_inventory)
    return (
        '<div class="table-wrap"><table class="tests">\n'
        '<thead><tr><th scope="col">What the test guarantees</th>'
        '<th scope="col" class="num">Cases</th></tr></thead>\n'
        f"<tbody>\n{chr(10).join(rows)}\n</tbody>\n"
        f"<caption>{total} test cases in {len(src.test_inventory)} test functions.</caption>"
        "</table></div>"
    )


def _code(*objects: object) -> str:
    """Source of classes or functions, as a code block captioned with the file it is in."""
    path = Path(inspect.getsourcefile(objects[0])).resolve().relative_to(PROJECT_ROOT)
    code = "\n\n".join(textwrap.dedent(inspect.getsource(obj)).rstrip() for obj in objects)
    caption = f'<p class="source">{path.as_posix()}</p>'
    return f"<pre><code>{escape(code)}</code></pre>\n{caption}"


def block_code_armstats(src: Sources, page: str) -> str:
    return _code(ArmStats)


def block_code_thompson(src: Sources, page: str) -> str:
    return _code(ThompsonSamplingPolicy)


def block_code_egreedy(src: Sources, page: str) -> str:
    return _code(
        EpsilonGreedyPolicy.current_epsilon,
        EpsilonGreedyPolicy.estimate,
        EpsilonGreedyPolicy.rank,
    )


def block_code_update(src: Sources, page: str) -> str:
    return "\n".join(
        _code(method)
        for method in (BanditPolicy.update, InMemoryOptionsStore.learn, SqlOptionsStore.learn)
    )


def block_code_static(src: Sources, page: str) -> str:
    return _code(StaticPolicy.fit, StaticPolicy.rank)


def block_policies(src: Sources, page: str) -> str:
    active = src.settings.recommendation.active_policy
    algorithm = {"random": "Random", "static": "Static (batch)",
                 "epsilon_greedy": "Epsilon-greedy", "thompson": "Thompson sampling"}  # fmt: skip
    rows = []
    for name, spec in src.settings.policies.items():
        learns = spec.type in ("epsilon_greedy", "thompson")
        exploration = {
            "random": "always random",
            "static": "none",
            "epsilon_greedy": f"ε = {spec.epsilon:g}",
            "thompson": "by uncertainty",
        }[spec.type]
        label = f"<code>{name}</code>"
        if name == active:
            label += ' <span class="chip built">Active</span>'
        rows.append(
            [
                label,
                algorithm[spec.type],
                src.settings.reward_for(name) if learns else "–",
                spec.context.replace("_", " × ") if learns else "–",
                f"γ = {spec.decay:g}" if learns and spec.decay < 1 else ("off" if learns else "–"),
                exploration,
            ]
        )
    return table(
        ["Policy", "Algorithm", "Reward", "Context key", "Forgetting", "Exploration"], rows
    )


def block_rewards(src: Sources, page: str) -> str:
    by_id = {offer.offer_id: offer for offer in src.offers}
    examples = ["data_20gb", "funeral_cover", "airtime_r50_bonus"]
    binary = make_reward("binary", src.offers)
    margin = make_reward("margin", src.offers)
    lenient = make_reward("binary", src.offers, learn_from_ignored=True)

    def show(value: float | None) -> str:
        return "no update" if value is None else f"{value:.3g}"

    rows = [
        [f"<code>{outcome}</code>", show(binary(by_id["data_1gb"], outcome))]
        + [show(margin(by_id[offer_id], outcome)) for offer_id in examples]
        + [show(lenient(by_id["data_1gb"], outcome))]
        for outcome in OUTCOMES
    ]
    revenue_headers = [f"Revenue: {o} (R{by_id[o].margin_zar:.0f})" for o in examples]
    headers = ["Outcome", "Binary", *revenue_headers, "Binary, learning from ignored"]
    top_margin = max(o.margin_zar for o in src.offers)
    note = (
        f'<p class="source">Revenue reward = margin ÷ top margin (R{top_margin:.0f}). '
        f"<code>{IGNORED}</code> makes no update unless learn_from_ignored is set; "
        f"<code>{TAKEN_UP}</code> is the only outcome that earns.</p>"
    )
    return table(headers, rows, numeric=(1, 2, 3, 4, 5)) + "\n" + note


def block_endpoints(src: Sources, page: str) -> str:
    from recommender.api.main import create_app  # routes only; no database connection

    rows = []
    for route in create_app().routes:
        methods = getattr(route, "methods", None)
        if not methods or route.path.startswith(("/docs", "/openapi", "/redoc")):
            continue
        method = sorted(methods - {"HEAD"})[0]
        summary = (inspect.getdoc(route.endpoint) or "").split("\n\n")[0].replace("\n", " ")
        rows.append(
            [f"<code>{method}</code>", f"<code>{escape(route.path)}</code>", escape(summary)]
        )
    return table(["Method", "Path", "What it does"], rows)


def block_event_schema(src: Sources, page: str) -> str:
    from recommender.events.log import EVENT_TABLES

    rows = []
    for event_table in EVENT_TABLES:
        for column in event_table.columns:
            keys = ["PK"] if column.primary_key else []
            keys += ["FK"] if column.foreign_keys else []
            rows.append(
                [
                    f"<code>{event_table.name}</code>",
                    f"<code>{column.name}</code>",
                    escape(str(column.type)),
                    " ".join(keys),
                    "yes" if column.nullable and not column.primary_key else "no",
                ]
            )
    return table(["Table", "Column", "Type", "Key", "Nullable"], rows)


# --- results (from docs/data/comparison.json, written by make compare) ----------------

RESULTS_PATH = DOCS / "data" / "comparison.json"
# The policies drawn in the regret chart, each with its fixed categorical colour slot.
CHART_SERIES = [
    ("thompson_segment_decay", 1),
    ("static", 2),
    ("random", 3),
    ("thompson", 4),
    ("epsilon_greedy_segment", 5),
]
NO_RESULTS = '<p class="note">No results yet: run <code>make compare</code>, then <code>make docs</code>.</p>'


def _results() -> dict | None:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else None


def block_results_table(src: Sources, page: str) -> str:
    data = _results()
    if data is None:
        return NO_RESULTS
    best = data["table"][0]["policy"]
    rows = []
    for row in data["table"]:
        cells = [
            f"<code>{escape(row['policy'])}</code>",
            f"{row['take_up_rate']:.3f}",
            f"{row['take_up_before_shift']:.3f}",
            f"{row['take_up_after_shift']:.3f}",
            f"R{row['revenue_per_open']:.2f}",
            f"{row['regret']:,.0f} ± {row['regret_sd']:,.0f}",
            f"{row['regret_after_shift']:,.0f}",
            f"{row['share_of_oracle']:.1%}",
            f"{row['regret_vs_random']:.0%}",
            f"{row['regret_vs_static']:.0%}",
        ]
        css = ' class="best"' if row["policy"] == best else ""
        rows.append(
            f"<tr{css}>"
            + "".join(f"<td{' class=num' if i else ''}>{cell}</td>" for i, cell in enumerate(cells))
            + "</tr>"
        )
    head = "".join(
        f'<th scope="col"{" class=num" if i else ""}>{h}</th>'
        for i, h in enumerate(["Policy", "Take-up per open", "Before shift", "After shift",
                               "Revenue per open", "Regret", "Regret after shift",
                               "Share of oracle", "Regret cut vs random", "Regret cut vs static"])
    )  # fmt: skip
    caption = (
        f"{data['rounds']:,} app opens per run, behaviour shift at {data['shift_round']:,}, "
        f"mean of {data['seeds']} seeds; the static baseline was fitted on "
        f"{data['warm_up_rounds']:,} warm-up app opens."
    )
    return (
        f'<div class="table-wrap"><table class="results-table">\n<thead><tr>{head}</tr></thead>\n'
        f"<tbody>\n{chr(10).join(rows)}\n</tbody><caption>{caption}</caption></table></div>"
    )


def _paired(by_seed: list[dict], a: str, b: str, metric: str) -> list[float]:
    """Per seed, ``metric`` of policy ``a`` minus that of ``b``. Seeds are paired: both
    policies faced the identical app opens, so the difference is theirs alone."""
    values = {(row["policy"], row["seed"]): row[metric] for row in by_seed}
    seeds = sorted({row["seed"] for row in by_seed})
    return [values[(a, s)] - values[(b, s)] for s in seeds if (a, s) in values and (b, s) in values]


def _findings(data: dict) -> list[tuple[str, str]]:
    """Headline findings as (headline, text), in plain text for HTML and Markdown alike.
    Every comparison is paired by seed and says how many seeds agree, so nothing is
    claimed that the runs do not support."""
    by_seed = data.get("by_seed", [])
    for row in by_seed:
        row["regret_before_shift"] = row["regret"] - row["regret_after_shift"]
    table = {row["policy"]: row for row in data["table"]}
    n = data["seeds"]
    items = []

    def lower_in(diffs: list[float]) -> int:
        return sum(d < 0 for d in diffs)

    learning = [r["policy"] for r in data["table"] if r["policy"] not in ("random", "static")]
    best = data["table"][0]
    static, random_ = table.get("static"), table.get("random")
    if static and random_:
        vs_static = lower_in(_paired(by_seed, best["policy"], "static", "regret"))
        vs_random = lower_in(_paired(by_seed, best["policy"], "random", "regret"))
        uplift = best["take_up_rate"] / static["take_up_rate"] - 1
        items.append(
            (
                "Learning per segment beats both baselines, in every run.",
                f"{best['policy']} reached {best['share_of_oracle']:.1%} of the oracle, with "
                f"{uplift:.0%} more take-ups per app open than the static batch model "
                f"({best['take_up_rate']:.3f} against {static['take_up_rate']:.3f}). It cut regret by "
                f"{best['regret_vs_random']:.0%} against random and {best['regret_vs_static']:.0%} "
                f"against static: below random's regret in {vs_random} of {n} seeds and below "
                f"static's in {vs_static} of {n}.",
            )
        )
        items.append(
            (
                "It adapts when behaviour changes; the batch model does not.",
                f"After the shift its take-up rose from {best['take_up_before_shift']:.3f} to "
                f"{best['take_up_after_shift']:.3f} per app open; the static model went from "
                f"{static['take_up_before_shift']:.3f} to {static['take_up_after_shift']:.3f}.",
            )
        )

    tied = [
        policy
        for policy in learning[1:]
        if lower_in(_paired(by_seed, best["policy"], policy, "regret")) < 0.8 * n
    ]
    if tied:
        items.append(
            (
                "The top of the table is a tie.",
                f"{best['policy']} has the lowest mean regret, but {', '.join(tied)} had lower "
                f"regret in at least a fifth of the seeds, so the order among them is within the "
                f"noise.",
            )
        )

    personal, global_ = table.get("thompson_segment"), table.get("thompson")
    if personal and global_:
        wins = lower_in(_paired(by_seed, "thompson_segment", "thompson", "regret"))
        items.append(
            (
                "Personalisation is the biggest single lever.",
                f"Thompson sampling with statistics per segment reached "
                f"{personal['share_of_oracle']:.1%} of the oracle; one global model reached "
                f"{global_['share_of_oracle']:.1%} (lower regret per segment in {wins} of {n} seeds).",
            )
        )

    decay = table.get("thompson_segment_decay")
    if personal and decay:
        after = _paired(by_seed, "thompson_segment_decay", "thompson_segment", "regret_after_shift")
        before = _paired(
            by_seed, "thompson_segment_decay", "thompson_segment", "regret_before_shift"
        )
        items.append(
            (
                "Forgetting trades a little before the shift for more after it.",
                f"With forgetting, regret after the shift was lower in {lower_in(after)} of {n} seeds, "
                f"by {-sum(after) / len(after):,.0f} take-ups on average; before the shift it cost "
                f"{sum(before) / len(before):,.0f} on average (higher in "
                f"{sum(d > 0 for d in before)} of {n} seeds). Its value grows with how often, and "
                f"how sharply, behaviour changes.",
            )
        )

    revenue = table.get("thompson_segment_revenue")
    if revenue and decay:
        more = _paired(
            by_seed, "thompson_segment_revenue", "thompson_segment_decay", "revenue_per_open"
        )
        uplift = revenue["revenue_per_open"] / decay["revenue_per_open"] - 1
        drop = 1 - revenue["take_up_rate"] / decay["take_up_rate"]
        items.append(
            (
                "The reward decides the business outcome.",
                f"The same policy learning from margin instead of take-up earned {uplift:.0%} more "
                f"revenue per app open (R{revenue['revenue_per_open']:.2f} against "
                f"R{decay['revenue_per_open']:.2f}, more in {sum(d > 0 for d in more)} of {n} "
                f"seeds), at the cost of {drop:.0%} fewer take-ups. Which one is right is a "
                f"business choice, not a modelling one.",
            )
        )

    greedy = table.get("epsilon_greedy_segment")
    if personal and greedy:
        wins = lower_in(_paired(by_seed, "thompson_segment", "epsilon_greedy_segment", "regret"))
        items.append(
            (
                "Thompson sampling explores better than epsilon-greedy.",
                f"With the same per-segment statistics, regret was {personal['regret']:,.0f} against "
                f"{greedy['regret']:,.0f}, lower in {wins} of {n} seeds: epsilon-greedy keeps "
                f"spending a fixed share of slots on offers it already knows are poor.",
            )
        )
    convergence = data.get("convergence", [])
    after = {
        (row["policy"], row["segment"]): row["app_opens"]
        for row in convergence
        if row["phase"] == "after the shift"
    }

    def opens(policy: str, segment: str) -> str:
        value = after.get((policy, segment))
        return (
            "not within the run" if value is None or value != value else f"{value:,.0f} app opens"
        )

    if ("thompson_segment_decay", "young_professional") in after:
        items.append(
            (
                "Forgetting adapts sooner.",
                "After the shift, with forgetting the young professionals' new favourite led within "
                f"{opens('thompson_segment_decay', 'young_professional')}, against "
                f"{opens('thompson_segment', 'young_professional')} without; for families, "
                f"{opens('thompson_segment_decay', 'family')} against "
                f"{opens('thompson_segment', 'family')}. The static model adapted for neither segment "
                "(first seed; see the adaptation table).",
            )
        )
    return items


def block_findings(src: Sources, page: str) -> str:
    data = _results()
    if data is None:
        return NO_RESULTS
    items = [
        f"<strong>{escape(head, quote=False)}</strong> {escape(text, quote=False)}"
        for head, text in _findings(data)
    ]
    return '<ol class="findings">\n' + "\n".join(f"<li>{item}</li>" for item in items) + "\n</ol>"


# Markdown blocks, for README.md and WRITEUP.md.
MD_NO_RESULTS = "_No results yet: run `make compare`, then `make docs`._"


def block_md_results(src: Sources, page: str) -> str:
    data = _results()
    if data is None:
        return MD_NO_RESULTS
    best = data["table"][0]["policy"]
    lines = [
        "| Policy | Take-up per app open | Revenue per app open | Regret (take-ups lost) "
        "| Share of oracle | Regret cut vs random | Regret cut vs static |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for row in data["table"]:
        name = f"`{row['policy']}`"
        lines.append(
            f"| {f'**{name}**' if row['policy'] == best else name} "
            f"| {row['take_up_rate']:.3f} | R{row['revenue_per_open']:.2f} "
            f"| {row['regret']:,.0f} ± {row['regret_sd']:,.0f} | {row['share_of_oracle']:.1%} "
            f"| {row['regret_vs_random']:.0%} | {row['regret_vs_static']:.0%} |"
        )
    lines.append(
        f"\n_{data['rounds']:,} app opens per run, behaviour shift at {data['shift_round']:,}, "
        f"mean ± standard deviation over {data['seeds']} seeds. Regret is take-ups lost against "
        f"an oracle that knows the hidden take-up probabilities._"
    )
    return "\n".join(lines)


def block_md_findings(src: Sources, page: str) -> str:
    """All findings as a numbered list (``md-headline``: the first two only)."""
    data = _results()
    if data is None:
        return MD_NO_RESULTS
    findings = _findings(data)
    return "\n".join(f"{i}. **{head}** {text}" for i, (head, text) in enumerate(findings, 1))


def block_md_headline(src: Sources, page: str) -> str:
    data = _results()
    if data is None:
        return MD_NO_RESULTS
    return "\n".join(f"- **{head}** {text}" for head, text in _findings(data)[:2])


def block_regret_chart(src: Sources, page: str) -> str:
    data = _results()
    if data is None:
        return NO_RESULTS
    curves = data["regret_curves"]
    series = [(name, slot) for name, slot in CHART_SERIES if name in curves]
    rounds, shift = data["rounds"], data["shift_round"]
    x0, x1, y0, y1 = 70, 730, 30, 340
    peak = max(curves[name]["regret"][-1] for name, _ in series)
    step = 1000 if peak <= 8000 else 2000
    top = math.ceil(peak / step) * step

    def sx(r: float) -> float:
        return x0 + (x1 - x0) * r / rounds

    def sy(v: float) -> float:
        return y1 - (y1 - y0) * v / top

    parts = []
    for value in range(0, top + 1, step):
        parts.append(
            f'<line class="grid" x1="{x0}" x2="{x1}" y1="{sy(value):.1f}" y2="{sy(value):.1f}"/>'
        )
        parts.append(
            f'<text class="axis-label" x="{x0 - 8}" y="{sy(value) + 4:.1f}" text-anchor="end">'
            f"{value:,}</text>"
        )
    for r in range(0, rounds + 1, rounds // 5):
        parts.append(
            f'<text class="axis-label" x="{sx(r):.1f}" y="{y1 + 18}" text-anchor="middle">'
            f"{r // 1000}k</text>"
        )
    parts.append(
        f'<text class="axis-label" x="{(x0 + x1) / 2:.0f}" y="{y1 + 38}" text-anchor="middle">App opens</text>'
    )
    parts.append(
        f'<text class="axis-label" x="{x0}" y="{y0 - 14}">Take-ups lost to the oracle (cumulative)</text>'
    )
    if shift:
        parts.append(
            f'<line class="shift" x1="{sx(shift):.1f}" x2="{sx(shift):.1f}" y1="{y0}" y2="{y1}"/>'
        )
        parts.append(
            f'<text class="shift-label" x="{sx(shift) + 6:.1f}" y="{y0 + 12}">behaviour shift</text>'
        )

    ends = []
    for name, slot in series:
        xs, ys = [0, *curves[name]["round"]], [0.0, *curves[name]["regret"]]
        points = " ".join(f"{sx(r):.1f},{sy(v):.1f}" for r, v in zip(xs, ys, strict=True))
        colour = f"var(--series-{slot})"
        parts.append(
            f'<polyline class="series" points="{points}" style="stroke: {colour}">'
            f"<title>{name}: {ys[-1]:,.0f} take-ups lost by {rounds:,} app opens</title></polyline>"
        )
        parts.append(
            f'<circle class="end-dot" cx="{sx(xs[-1]):.1f}" cy="{sy(ys[-1]):.1f}" r="4" style="fill: {colour}"/>'
        )
        ends.append([sy(ys[-1]), name, ys[-1]])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least 16px apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
    for label_y, name, value in ends:
        parts.append(
            f'<text class="end-label" x="{x1 + 12}" y="{label_y + 4:.1f}">{name} '
            f'<tspan class="end-value">{value:,.0f}</tspan></text>'
        )

    legend = "".join(
        f'<span><i class="swatch-line" style="background: var(--series-{slot})"></i>{name}</span>'
        for name, slot in series
    )
    label = "Cumulative regret per policy over the run; lower is better. " + ", ".join(
        f"{name} ends at {curves[name]['regret'][-1]:,.0f}" for name, _ in series
    )
    return (
        '<figure class="fig">\n<div class="fig-scroll">\n'
        f'<svg class="dg chart" viewBox="0 0 980 400" style="min-width: 760px" role="img" '
        f'aria-label="{escape(label)}">\n' + "\n".join(parts) + "\n</svg>\n</div>\n"
        f'<figcaption><span class="legend">{legend}</span><span>Expected take-ups lost '
        "against an oracle that knows the true probabilities, first seed. The other policies "
        "are in the table: the per-segment Thompson variants overlap the blue line.</span>"
        "</figcaption>\n</figure>"
    )


ADAPTATION_POLICIES = ["thompson_segment_decay", "thompson_segment", "thompson_segment_payday",
                       "epsilon_greedy_segment", "static"]  # fmt: skip


def block_adaptation(src: Sources, page: str) -> str:
    data = _results()
    if data is None or "convergence" not in data:
        return NO_RESULTS
    rows = {}
    for row in data["convergence"]:
        if row["policy"] in ADAPTATION_POLICIES:
            rows.setdefault((row["phase"], row["segment"]), {})[row["policy"]] = row["app_opens"]
    policies = [p for p in ADAPTATION_POLICIES if any(p in r for r in rows.values())]

    def cell(value) -> str:
        return "not within the run" if value is None or value != value else f"{value:,.0f}"

    body = []
    for phase in ("from the start", "after the shift"):
        for (row_phase, segment), values in sorted(rows.items()):
            if row_phase == phase:
                body.append([escape(phase), f"<code>{segment}</code>"]
                            + [cell(values.get(p)) for p in policies])  # fmt: skip
    head = ["When", "Segment", *[f"<code>{p}</code>" for p in policies]]
    return table(head, body, cls="adaptation", numeric=tuple(range(2, 2 + len(policies))))


BLOCKS: dict[str, Callable[[Sources, str], str]] = {
    "sidenav": block_sidenav,
    "phases": block_phases,
    "facts": block_facts,
    "offers": block_offers,
    "segments": block_segments,
    "effects": block_effects,
    "heatmap": block_heatmap,
    "favourites": block_favourites,
    "shift": block_shift,
    "example": block_example,
    "attention": block_attention,
    "decision-log": block_decision_log,
    "tools": block_tools,
    "tests": block_tests,
    "policies": block_policies,
    "rewards": block_rewards,
    "code-armstats": block_code_armstats,
    "code-thompson": block_code_thompson,
    "code-egreedy": block_code_egreedy,
    "code-update": block_code_update,
    "code-static": block_code_static,
    "endpoints": block_endpoints,
    "event-schema": block_event_schema,
    "results-table": block_results_table,
    "findings": block_findings,
    "regret-chart": block_regret_chart,
    "adaptation": block_adaptation,
    "md-results": block_md_results,
    "md-findings": block_md_findings,
    "md-headline": block_md_headline,
}
# Markdown pages at the repository root with generated blocks of their own.
MARKDOWN_PAGES = ["README.md", "WRITEUP.md"]


def render(text: str, page: str, src: Sources) -> str:
    """Replace every generated block in one page."""

    def fill(match: re.Match) -> str:
        name = match["name"]
        if name not in BLOCKS:
            raise KeyError(f"{page}: unknown generated block '{name}'")
        return f"{match['begin']}\n{BLOCKS[name](src, page)}\n{match['end']}"

    return BLOCK.sub(fill, text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the data-driven parts of docs/.")
    parser.add_argument("--check", action="store_true", help="fail if any page is stale")
    args = parser.parse_args()

    src = Sources()
    stale = []
    paths = [DOCS / page for page, _ in PAGES]
    paths += [PROJECT_ROOT / name for name in MARKDOWN_PAGES if (PROJECT_ROOT / name).exists()]
    for path in paths:
        page = path.name
        current = path.read_text(encoding="utf-8")
        updated = render(current, page, src)
        if updated != current:
            stale.append(page)
            if not args.check:
                path.write_text(updated, encoding="utf-8", newline="\n")

    if args.check and stale:
        print(f"Docs are out of date: {', '.join(stale)}. Run `make docs`.", file=sys.stderr)
        return 1
    print("Docs are up to date." if not stale else f"Updated {', '.join(stale)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
