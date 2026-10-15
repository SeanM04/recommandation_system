# Build Prompt: Real-Time Dynamic Offer Recommender (Telco / Fintech App)

## Who I am and why this exists
I'm Sean Mukariri, a Data Engineer (Python, SQL, Spark, Airflow, AWS/GCP). I'm building this portfolio project for the technical team at **ecosystem.Ai**, whose platform runs *dynamic recommenders*. Those recommenders learn in real time from customer responses, using algorithms such as Epsilon Greedy and Thompson sampling, rather than relying on a static, batch-trained model. Their philosophy is "life happens in the moment, not in batches."

The project should look and work like a small, production-minded version of that idea, set inside a telco super-app similar to VodaPay. **Deadline: Friday 16 October.** Prioritise a working, well-documented end-to-end system over extra features.

You are my coding partner. Follow the phases below in order. After each phase, run the code and tests, fix anything broken, and give me a short summary before moving on.

---

## What the system does (one paragraph)
A simulated population of app users opens a VodaPay-style app. Each time a user opens it, the app calls a **recommender API** for the best offers for that user. The API ranks offers with a multi-armed bandit algorithm. A simulator decides whether the user takes up an offer, based on hidden "true" probabilities that the recommender cannot see, and reports the outcome back to a **response API**. The model updates immediately. A dashboard shows the system learning: cumulative reward rising, regret falling, and the algorithm converging on the best offers per segment, compared with random and static baselines.

---

## Tech stack (keep it simple and runnable locally)
- **Python 3.11+**
- **FastAPI** + Uvicorn for the API
- **SQLite** (via SQLAlchemy or plain `sqlite3`) for the event log and model state. Design the code so it could be swapped for Postgres.
- **NumPy / pandas** for the algorithms and analysis
- **Streamlit** + Plotly for the dashboard
- **pytest** for tests
- **Docker + docker-compose** so the project runs with one command
- **Makefile** with `make install`, `make run`, `make simulate`, `make dashboard`, `make test`

Don't use heavy ML frameworks. Write the bandit algorithms by hand so the logic is visible.

---

## Project structure
```
telco-dynamic-recommender/
├── README.md
├── WRITEUP.md
├── Makefile
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config/
│   ├── offers.yaml            # offer catalogue
│   └── settings.yaml          # algorithm + simulation settings
├── src/recommender/
│   ├── api/                   # FastAPI app, routes, schemas
│   ├── algorithms/            # base class + epsilon_greedy, thompson, random, static
│   ├── store/                 # options store / model state persistence
│   ├── events/                # event logging (invocations + responses)
│   ├── simulator/             # synthetic customers + behaviour engine
│   └── evaluation/            # metrics: CTR, cumulative reward, regret
├── dashboard/app.py           # Streamlit dashboard
├── scripts/                   # run_simulation.py, compare_algorithms.py
├── notebooks/                 # optional analysis notebook
└── tests/
```

---

## Phase 1: Domain and data
1. **Offer catalogue** (`config/offers.yaml`) with about 12 realistic South African telco/fintech offers. Each offer has `offer_id`, `name`, `category`, `price_zar`, `margin_zar`, and `eligible_segments`. Examples:
   - 1GB / 5GB / 20GB data bundles
   - R50 airtime with 10% bonus
   - Electricity token cashback
   - Pay-later "Buy now" voucher
   - Funeral cover or device insurance
   - Savings pocket (earn interest)
   - Spotify / Showmax bundle
   - Uber or food-delivery voucher
   - International roaming pack
2. **Synthetic customers.** Generate about 5,000 users with contextual attributes: `segment` (student, young_professional, family, small_business, pensioner), `device_tier`, `prepaid_vs_contract`, `avg_monthly_spend`, `province`, `time_of_day`, `day_of_week`, and `is_payday_week`.
3. **Hidden ground truth.** For each (segment × offer), define a true take-up probability, with realistic effects such as payday boosts, students favouring data, and families favouring insurance. The recommender must never read these values; only the simulator uses them.

## Phase 2: Algorithms (the core)
Create a common interface:
```python
class Policy:
    def rank(self, context, candidate_offers, n) -> list[ScoredOffer]: ...
    def update(self, context, offer_id, reward) -> None: ...
```
Implement:
- **RandomPolicy**: baseline.
- **StaticPolicy**: always ranks by a fixed popularity order (the "batch model" baseline).
- **EpsilonGreedyPolicy**: configurable epsilon, optional epsilon decay.
- **ThompsonSamplingPolicy**: Beta(alpha, beta) per arm, sampled at ranking time.
- **Contextual variant**: keep separate arm statistics per segment (and optionally per segment × payday). This is the "personalised" mode.
- **Forgetting / decay**: an optional decay factor that discounts old evidence so the model adapts when behaviour changes. ecosystem.Ai calls this "sophisticated forgetfulness", so make it configurable and demonstrate it.
- **Reward options**: binary take-up, or margin-weighted reward (revenue-aware). Make the reward function pluggable.

State lives in an **options store** table: `(policy, context_key, offer_id, alpha, beta, pulls, total_reward, updated_at)`.

## Phase 3: Real-time API
FastAPI endpoints:
- `POST /invocations`: input `customer_id`, `channel` ("app", "ussd", "sms"), optional context overrides, and `number_of_offers`. It returns ranked offers with scores, plus a unique `invocation_id`. Log every invocation.
- `POST /response`: input `invocation_id`, `offer_id`, `outcome` ("taken_up", "viewed", "ignored"), optional `value`. It updates the model immediately and logs the event.
- `GET /offers`, `GET /policies/{name}/state`, `GET /health`, `GET /metrics`.
- Use Pydantic schemas and validation. Return clear errors for unknown customers or offers, and for duplicate responses (idempotency on `invocation_id` + `offer_id`).
- Make the active policy switchable through config or a query parameter so policies can be A/B compared.
- Target p95 latency under 50 ms locally. Add a simple timing middleware and report the number.

## Phase 4: Simulator
- `scripts/run_simulation.py` runs N rounds (default 50,000). Each round, it samples a customer and context, calls `/invocations` (or the policy directly in fast mode), draws an outcome from the hidden probabilities, and calls `/response`.
- **Behaviour shift scenario.** Partway through the run (for example at round 25,000), change the ground truth. For instance, a new streaming bundle becomes hot with young professionals. This shows that decay or forgetting lets the model adapt where the static model fails.
- `scripts/compare_algorithms.py` runs every policy on the same random seed and saves results to `results/`.

## Phase 5: Evaluation and dashboard
Metrics per policy:
- take-up rate (CTR)
- cumulative reward and revenue
- **cumulative regret** against an oracle that knows the true probabilities
- arm selection share over time
- convergence: rounds needed to find the best arm per segment

Streamlit dashboard pages:
1. **Live view**: recent invocations and responses streaming in, current top offers per segment.
2. **Algorithm comparison**: regret and reward curves for Random vs Static vs Epsilon Greedy vs Thompson (contextual and non-contextual).
3. **Explore vs exploit**: Beta distributions per offer for a selected segment, shown changing over time.
4. **Behaviour shift**: before/after view showing adaptation with and without decay.
5. **Try it**: pick a customer, call the API, and click "take up" or "ignore" to watch the model update.

## Phase 6: Quality and delivery
- **Tests**: algorithm unit tests (Thompson converges to the best arm on a known bandit; epsilon=0 is greedy), API tests with `TestClient`, and a simulator smoke test.
- Type hints, docstrings on public functions, `ruff` formatting.
- Logging with structured JSON lines.
- `docker-compose up` starts the API and dashboard.
- Optional stretch goal if time allows: an Airflow DAG (or a simple scheduled script) that runs a nightly batch job aggregating the event log into daily performance tables, to show the data-engineering side.

## README.md must include
- A one-paragraph pitch and an architecture diagram (Mermaid)
- Quickstart in three commands or fewer
- API examples with `curl`
- Screenshots of the dashboard
- The results table from `compare_algorithms.py`

## WRITEUP.md (1–2 pages, for Rob and the technical team)
- Problem: why static, batch recommenders fall short in a telco app
- Approach: the bandit formulation, contextual segments, rewards, and decay
- Results: the key numbers (regret reduction against static and random, take-up uplift, adaptation speed after the behaviour shift)
- How this maps to ecosystem.Ai concepts: dynamic recommender, options store, real-time feedback, Epsilon Greedy and Thompson sampling, forgetting
- What I'd do next in production: Postgres or Kafka event stream, feature store, monitoring and drift alerts, guardrails such as offer eligibility, fatigue caps and frequency limits

---

## Working rules for you (the AI assistant)
1. Start by creating the folder structure, `requirements.txt`, and a minimal FastAPI `/health` endpoint, and confirm it runs.
2. Build in the phase order above. Don't skip ahead.
3. After each phase, run the tests and show me what passed. Keep the code runnable at every step.
4. Prefer clear, readable code over clever code. A reviewer should understand the bandit logic in 2 minutes.
5. Use a fixed random seed so results are reproducible.
6. Don't invent ecosystem.Ai APIs or claim the project uses their product. It is an independent build *inspired by* their approach.
7. If something is ambiguous, pick the sensible default, note it in `DECISIONS.md`, and keep going.
8. Commit in small logical steps with clear messages (for example `feat: thompson sampling policy`).

**Definition of done:** `make install && make simulate && make dashboard` works on a clean machine. All tests pass. The README and WRITEUP are complete. The comparison results clearly show Thompson sampling (contextual) beating the static and random baselines and adapting after the behaviour shift.
