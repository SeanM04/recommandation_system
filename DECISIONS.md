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
