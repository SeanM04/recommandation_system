"""Regenerate the data-driven parts of the HTML docs in ``docs/`` from code and config.

Prose and diagrams in ``docs/*.html`` are written by hand. Everything between
``<!-- BEGIN generated:NAME -->`` and ``<!-- END generated:NAME -->`` is rebuilt here
from the source of truth (YAML config, the ground-truth model, DECISIONS.md,
requirements.txt and the test files), so the docs cannot drift from the code.

Usage:  python scripts/build_docs.py           rewrite the docs in place
        python scripts/build_docs.py --check   exit 1 if any page is out of date
"""

from __future__ import annotations

import argparse
import ast
import inspect
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
        "planned",
        "/invocations and /response with validation, idempotency, an event log, policy "
        "switching and latency timing.",
    ),
    (
        4,
        "Simulator",
        "planned",
        "50,000 app opens through the API or in fast mode, with the behaviour shift halfway.",
    ),
    (
        5,
        "Evaluation & dashboard",
        "planned",
        "Take-up, revenue, regret against an oracle and convergence; a five-page dashboard.",
    ),
    (
        6,
        "Quality & delivery",
        "planned",
        "Docker, JSON logging, README and WRITEUP, and an optional nightly aggregation job.",
    ),
]

# fmt: off
TOOLS = [  # name, package in requirements.txt, role, status, why it was chosen
    ("Python", None, "Language", "built",
     "Readable, typed code; the bandit maths is plain Python and NumPy."),
    ("FastAPI", "fastapi", "Real-time API", "built",
     "Pydantic validation and OpenAPI docs out of the box. Serves /health today."),
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
    ("Requests", "requests", "HTTP client", "planned",
     "The simulator and dashboard call the API the way an app would."),
    ("Streamlit", "streamlit", "Dashboard", "planned",
     "Python-only dashboards; quick to build five pages."),
    ("Plotly", "plotly", "Charts", "planned",
     "Interactive regret, reward and Beta-distribution charts."),
    ("Docker Compose", None, "Packaging", "planned",
     "Starts the API and dashboard with one command."),
]
# fmt: on

TEST_FILES = {
    "test_api.py": "API endpoints",
    "test_domain.py": "Configuration, offer validation, calendar and context helpers",
    "test_population.py": "Synthetic population: reproducibility, realism, CSV round trip",
    "test_ground_truth.py": "Hidden truth, segment stories, payday, revenue, response "
    "model and the architecture guards",
    "test_docs.py": "These docs: generated blocks are current and every link resolves",
    "test_store.py": "Options store: arm statistics, atomic learning under concurrency, "
    "and identical behaviour in memory, SQLite and Postgres",
    "test_policies.py": "Bandit policies: ranking, learning, context keys, forgetting, convergence",
    "test_rewards.py": "Reward functions and the policy factory",
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
                spec.context.replace("_", " × ") if learns else "–",
                f"γ = {spec.decay:g}" if learns and spec.decay < 1 else ("off" if learns else "–"),
                exploration,
            ]
        )
    return table(["Policy", "Algorithm", "Context key", "Forgetting", "Exploration"], rows)


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
}


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
    for page, _ in PAGES:
        path = DOCS / page
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
