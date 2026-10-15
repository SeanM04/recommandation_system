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
