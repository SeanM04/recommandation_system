# Decisions log

Ambiguities in the brief, the default I picked, and why. Newest at the bottom of each section.

## Setup

| # | Decision | Why |
|---|----------|-----|
| S1 | The repository root is this folder; there is no nested `telco-dynamic-recommender/` directory. | The folder already existed with a `.gitignore`. Nesting would add a level with no benefit. |
| S2 | Exact-pinned dependencies in `requirements.txt` (`==`). | "Works on a clean machine" and reproducible results matter more than picking up new minor versions. |
| S3 | `httpx2` instead of `httpx` for FastAPI's `TestClient`. | Starlette deprecates `httpx` for its test client and emits a warning; `httpx2` is the recommended replacement. |
| S4 | On Windows/OneDrive, create the local virtualenv **outside** OneDrive (for example `%USERPROFILE%\.venvs\telco-recommender`). | OneDrive sync locks files inside `.venv` and `pip install` fails with `WinError 32`. This does not affect Docker. |
| S5 | `make` is the documented interface; every target is a single plain command, so it can be run by hand where `make` is not installed (for example Windows without Git-Bash make). | Keeps one source of truth for commands. |
| S6 | On Windows, keep the virtualenv path short (for example `C:\Users\<you>\.venvs\telco-recommender`). | Streamlit ships deeply nested files; in a long path they exceed Windows' 260-character limit and `pip install` fails with `WinError 206`. |

## Phase 1: Domain and data

| # | Decision | Why |
|---|----------|-----|
| D1 | A customer has a **static profile** (segment, device tier, prepaid/contract, monthly spend, province, *usual* time of day). `time_of_day`, `day_of_week` and `is_payday_week` are **per-session context**, derived when the app is opened (from the simulated calendar, or the server clock in the API). | These change between app opens. Storing them on the customer would freeze "the moment" the recommender is meant to react to. |
| D2 | Payday week: salaried segments from the 25th to month end; pensioners on SASSA grant days 1–7. | Matches South African pay patterns and makes payday a real contextual signal that differs by segment. |
| D3 | The hidden truth lives in `config/ground_truth.yaml` and is used only by `recommender.simulator` and offline evaluation. A test fails if any other `recommender` module imports the simulator. | Enforces "the recommender never reads the true probabilities" with a test rather than relying on convention. |
| D4 | Truth model: `base[segment][offer]` × multiplicative context effects (payday, time of day, weekend, device, contract, spend band, province), capped at 0.9. | One readable table plus a short list of named effects that a reviewer can reason about. |
| D5 | Take-up rates of roughly 2–20%, higher than the 1–5% typical in practice. | 50,000 simulated rounds are then enough to show convergence. The relative patterns matter, not the absolute levels. |
| D6 | Segment eligibility (`offers.yaml`) is a business rule the recommender **may** use. The ground truth defines probabilities for eligible pairs only, and this is validated on load. | Eligibility is public business logic; the probabilities are not. |
| D7 | Behaviour shift: young professionals' `streaming_bundle` base rate goes from 5% to 20% and their `data_5gb` drops from 13% to 8% (cannibalisation); students' streaming rises from 8% to 12%. | Flipping the favourite in the largest segment is the clearest adaptation test. Because the old favourite also gets *worse*, models that never forget are penalised. |
| D8 | The customer table is a CSV (`data/customers.csv`), regenerated deterministically from the seed and git-ignored. | Small, inspectable and reproducible. It can be loaded into SQLite or Postgres later. |
| D9 | Prices for cover products are monthly premiums; the savings pocket "price" is its minimum deposit. All prices and margins are illustrative. | Keeps `margin_zar <= price_zar` meaningful for every offer. |
| D10 | Generic names for third-party products ("Music + Video Streaming Bundle", "Food Delivery Voucher"). | Avoids implying real partnerships. |
| D11 | Population, the ground-truth path and the calendar live in `config/simulation.yaml`, loaded only by `recommender.simulator`. The recommender's `settings.yaml` has no path to the truth, and a test fails if recommender code mentions `ground_truth` or `simulation.yaml`. | Hiding the truth is enforced for file reads as well as imports. |
| D12 | The simulated calendar is 60 days from 2026-09-01: two salary paydays and two SASSA grant weeks. | Payday effects get exercised twice, and the shift can sit at the halfway point. |
| D13 | Payday **changes the favourite** in two segments. Pensioners: airtime for the rest of the month, electricity in grant week. Small businesses: pay-later in the mid-month cash squeeze, 20GB data in payday week. Other segments keep one favourite. | Without this the segment × payday context mode has nothing to learn and would lose to plain per-segment mode, because it splits the data for no benefit. |
| D14 | Funeral cover margin raised from R40 to R50. Every favourite, by take-up or by revenue, leads its runner-up by at least ×1.2 (tested). | Removes the family revenue near-tie (×1.02), which would have stopped any policy from settling. Revenue changes the favourite for students (5GB), young professionals (20GB), small businesses (20GB) and pensioners (funeral cover), which makes the revenue-aware reward worth demonstrating. |
| D15 | **Response model for a ranked list** (in the ground truth): the customer scans from the top and notices position k with probability `attention = [1.0, 0.6, 0.4, 0.25, 0.15]`. A noticed offer is taken up with its true probability, and the first take-up ends the visit. Outcomes: noticed and taken = `taken_up`; noticed, not taken = `viewed`; not noticed (or below a take-up) = `ignored`. | Gives the API's three outcomes a precise meaning and makes position matter, so ranking order affects reward. It is simple enough to compute expected reward exactly, which the regret oracle needs. *Implication for Phase 2:* `ignored` says little about whether an offer appeals, so how policies learn from it has to be a deliberate choice. |

## Phase 2: Algorithms

| # | Decision | Why |
|---|----------|-----|
| P1 | Policies hold no state. The **options store is the model**: one row per (policy, context key, offer). There is one interface with two implementations, in-memory for simulations and SQLAlchemy Core (SQLite now, Postgres by URL) for the API, and both pass the same tests. | A restart, a second API worker or a dashboard all see the same model. Simulations stay fast because they skip I/O. |
| P2 | The store keeps `pulls` and `total_reward`. `alpha = 1 + total_reward` and `beta = 1 + pulls - total_reward` (a Beta(1, 1) prior) are derived, and are still written to the table for anyone querying it. `pulls` is a float. | The Beta belief can never disagree with the counts. With forgetting on, `pulls` becomes an effective count. |
| P3 | Contextual personalisation is a `context` option on each learning policy (`global`, `segment`, `segment_payday`) rather than a separate class. The option becomes the store's `context_key`. | One implementation of each algorithm. Every variant is a line of config, so variants can be compared like for like. |
| P4 | Forgetting: before each update, **every** offer's evidence in that context shrinks by γ towards the prior, and then the new reward is added (discounted Thompson sampling). The same rule applies to epsilon-greedy averages. | Offers that are not being shown must also lose stale confidence, otherwise a once-good offer would keep a high, certain estimate forever. |
| P5 | A reward function maps an outcome to a value in [0, 1], or `None` for no update. `taken_up` scores 1 (binary) or margin ÷ the catalogue's top margin (revenue-aware); `viewed` scores 0; `ignored` makes no update by default (`learn_from_ignored` turns it into 0). | Rewards stay on the scale a Beta belief needs. Skipping `ignored` avoids punishing offers for a low rank (decision D15). |
| P6 | Epsilon-greedy fills the slate one slot at a time: each slot explores with probability ε, otherwise takes the best average. Untried offers score 1.0 so each is tried early, ties break randomly, and ε can decay with the evidence in the context (`ε × decay^pulls`, floored at `min_epsilon`). | The textbook algorithm extended to a ranked list, with every behaviour visible in about 20 lines. |
| P7 | The static baseline is one global popularity table (average reward per offer), fitted from a historical log by `StaticPolicy.fit` and then frozen. With no history it uses catalogue order. Phase 4 fits it on a warm-up batch. | A fair stand-in for a batch-trained model: it learned from real data once, then stopped. |
| P8 | Each policy gets its own random stream, seeded from the global seed and a hash of its name. | Results are reproducible, and adding or removing one policy never changes another's results. |
| P9 | A quick experiment (50,000 app opens, 3 offers each, shift at the halfway point) showed that Thompson sampling **without** forgetting adapts to the streaming hit on its own, because the offer was rarely shown and its belief stayed wide. Forgetting at γ = 0.999 only cost take-up. So the shift gained a second event: families' well-known runner-up, electricity cashback, becomes their favourite (base 10% to 20%). With that, no forgetting ranked electricity first in 24% of family app opens after the shift, against 77% at γ = 0.9998. Take-up per app open after the shift rose from 0.278 to 0.299, at a cost of 0.233 to 0.228 before it. The default decay is **0.9998**. | Forgetting pays off only when the model is *confident* about something that changes. The shift now tests both kinds of change, and the default rate is the one that won on evidence rather than a guess. |
| P10 | Default slate size is 3 offers per app open (`number_of_offers`). | Matches a phone screen's offer carousel and the attention model's first three positions. |

## Documentation

| # | Decision | Why |
|---|----------|-----|
| DOC1 | The docs are a static HTML site in `docs/`: six pages, one stylesheet, no build step and no JavaScript libraries. Diagrams are hand-drawn inline SVG. | Opens offline straight from the file system, deploys to GitHub Pages unchanged, and follows the reader's light or dark theme. SVG diagrams stay crisp and change in the same diff as the code they describe. |
| DOC2 | Every data table in the docs (offers, segments, effects, take-up rates, favourites, this log, tool versions, tests, build status) is generated from the code by `scripts/build_docs.py` between marker comments. `tests/test_docs.py` fails if a page is stale, a link or anchor is broken, or an id repeats. | Docs as code: the numbers a reviewer reads are the numbers the simulator uses, and stale docs fail the build like broken code. |
| DOC3 | The README will still carry its own Mermaid architecture diagram (Phase 6). | GitHub renders Mermaid inside Markdown, and the README is the first thing a reviewer sees. |
