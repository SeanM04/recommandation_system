PYTHON ?= python

.PHONY: install db data truth experiment run benchmark test lint format docs

install:  ## Install dependencies and the package in editable mode
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

db:  ## Create the tables (Postgres from .env, else SQLite) and migrate a local SQLite model
	$(PYTHON) scripts/init_db.py

data:  ## Generate the synthetic customer population (data/customers.csv)
	$(PYTHON) scripts/generate_customers.py

truth:  ## Print the hidden take-up table per segment (simulator only)
	$(PYTHON) scripts/inspect_ground_truth.py

experiment:  ## Rerun the forgetting experiment behind decision P9 (about a minute)
	$(PYTHON) scripts/experiment_forgetting.py

run:  ## Start the recommender API on http://localhost:8000 (docs at /docs)
	$(PYTHON) -m uvicorn recommender.api.main:app --host 0.0.0.0 --port 8000 --no-access-log

benchmark:  ## Measure API latency end to end (real server, database and HTTP)
	$(PYTHON) scripts/benchmark_api.py

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Check code style
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Auto-format code
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

docs:  ## Rebuild the data-driven tables in docs/*.html from code and config
	$(PYTHON) scripts/build_docs.py
