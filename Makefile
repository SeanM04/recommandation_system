PYTHON ?= python

.PHONY: help install db data truth simulate compare dashboard screenshots experiment nightly run benchmark docker-build docker-api benchmark-docker up down benchmark-compose test lint format docs

help:  ## List the targets
	@grep -E '^[a-z-]+:.*## ' Makefile | sed -E 's/:.*## /	/' | expand -t 20

install:  ## Install dependencies and the package in editable mode
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

db:  ## Create the tables (Postgres from .env, else SQLite) and migrate a local SQLite model
	$(PYTHON) scripts/init_db.py

data:  ## Generate the synthetic customer population (data/customers.csv)
	$(PYTHON) scripts/generate_customers.py

truth:  ## Print the hidden take-up table per segment (simulator only)
	$(PYTHON) scripts/inspect_ground_truth.py

simulate:  ## Everything the dashboard shows: customers, a 3-seed comparison, 1,000 app opens through the API
	$(PYTHON) scripts/generate_customers.py
	$(PYTHON) scripts/compare_algorithms.py --seeds 3 --no-docs
	$(PYTHON) scripts/run_simulation.py --mode api --in-process --rounds 1000

compare:  ## Compare every policy on the same world, 10 seeds (~7 min); results to results/ and the docs
	$(PYTHON) scripts/compare_algorithms.py

dashboard:  ## The Streamlit dashboard on http://localhost:8501 (run make simulate first)
	$(PYTHON) -m streamlit run dashboard/app.py

screenshots:  ## Screenshot every dashboard page into docs/assets/screenshots (dashboard running)
	$(PYTHON) scripts/screenshots.py

experiment:  ## Rerun the forgetting experiment behind decision P9 (about a minute)
	$(PYTHON) scripts/experiment_forgetting.py

nightly:  ## Roll yesterday's event log up into daily_offer_stats (cron: 5 0 * * *); ARGS=--all backfills
	$(PYTHON) scripts/nightly_aggregate.py $(ARGS)

run:  ## Start the recommender API on http://localhost:8000 (docs at /docs)
	$(PYTHON) -m uvicorn recommender.api.main:app --host 0.0.0.0 --port 8000 --no-access-log

benchmark:  ## Measure API latency end to end (real server, database and HTTP)
	$(PYTHON) scripts/benchmark_api.py --clients 1,4,8

docker-build:  ## Build the image (API, dashboard and scripts in one)
	docker build -t telco-recommender:latest .

docker-api:  ## Run the API in Docker on port 8000 (4 workers), using the database in .env
	docker run --rm -p 8000:8000 --env-file .env -e DB_HOST=host.docker.internal telco-recommender:latest

benchmark-docker:  ## Latency at 1, 4, 8 and 16 concurrent clients against the API in Docker
	$(PYTHON) scripts/benchmark_api.py --docker --workers 4 --clients 1,4,8,16

up:  ## The whole system in Docker: Postgres, API (:8000), dashboard (:8501), a first simulation
	docker compose up --build -d
	@echo "API http://localhost:8000/docs  dashboard http://localhost:8501  (simulation: docker compose logs -f simulate)"

down:  ## Stop the Docker system and delete its data
	docker compose down --volumes

benchmark-compose:  ## Latency against the compose API, which shares a Docker network with Postgres
	$(PYTHON) scripts/benchmark_api.py --url http://localhost:8000 --clients 1,4,8,16

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
