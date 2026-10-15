# Real-time dynamic offer recommender: write-up

For Rob and the ecosystem.Ai technical team. The code, the docs and every decision are in this repository; this is the two-page version.

## The problem

A telco or fintech app has room for three offers each time a customer opens it. Which three? Students want the cheapest data bundle. Pensioners want airtime, except in the week their grant is paid, when they buy electricity. Small businesses reach for a pay-later voucher mid-month but buy a large data bundle at payday. And preferences move: when a streaming bundle goes viral, young professionals switch to it, and an electricity tariff hike turns families from funeral cover to prepaid electricity.

The usual answer is a batch model, retrained monthly. It is out of date when it ships, never learns about offers it doesn't show, and can't notice a change in behaviour until someone retrains it. The brief asked for a recommender that learns from every response in real time, explores enough to find what works and adapts when behaviour shifts, with evidence that it beats a static model and random offers.

## Approach

**A world with a known answer.** Real take-up data doesn't come with the truth attached, so I simulated a world that does: 5,000 South African customers in five segments and 12 offers, each with a margin and eligibility rules. A hidden take-up probability per segment and offer is adjusted by 14 context effects (payday week, time of day, device, contract type and others). Customers read offers top-down with falling attention and take up at most one per app open. Halfway through each run, behaviour shifts in the two ways above. The policies never see any of this; the evaluation uses it to compute regret, the take-ups lost against an oracle that always shows the truly best three.

**Nine policies, one interface.** The baselines are random offers and a static popularity ranking, fitted on a month of history and then frozen. Against them run epsilon-greedy and Thompson sampling, with one global belief or one per segment, and three Thompson variants that add payday week to the context, forgetting, or a revenue reward. Thompson sampling keeps a Beta belief per context and offer, draws once from each eligible offer's belief and ranks by the draws. Offers it is unsure about sometimes draw high and get shown: that is the exploration, and it shrinks as evidence builds.

**Learning in the request path.** `POST /invocations` builds the context, ranks, logs what was shown and returns three offers. `POST /response` logs the outcome once (a repeat gets 409), turns it into a reward (1 for a take-up, 0 for a view; an ignored offer may never have been seen, so it teaches nothing) and updates the belief in one atomic transaction. Forgetting discounts the evidence in a context by γ = 0.9998 on every update, a memory of about 5,000 updates per context.

**A fair comparison.** Every policy faces the identical 50,000 app opens and outcome draws, so differences are the policies' alone. Each claim below is paired seed by seed over 10 seeds and says how many agree.

## Results

<!-- BEGIN generated:md-results -->
| Policy | Take-up per app open | Revenue per app open | Regret (take-ups lost) | Share of oracle | Regret cut vs random | Regret cut vs static |
|---|--:|--:|--:|--:|--:|--:|
| **`thompson_segment_decay`** | 0.263 | R9.68 | 1,041 ± 39 | 92.7% | 85% | 81% |
| `thompson_segment` | 0.261 | R10.01 | 1,078 ± 166 | 92.4% | 84% | 81% |
| `thompson_segment_payday` | 0.261 | R10.16 | 1,099 ± 127 | 92.3% | 84% | 80% |
| `epsilon_greedy_segment` | 0.250 | R9.94 | 1,690 ± 381 | 88.1% | 75% | 70% |
| `thompson_segment_revenue` | 0.205 | R13.37 | 3,916 ± 92 | 72.4% | 42% | 30% |
| `thompson` | 0.193 | R5.09 | 4,582 ± 274 | 67.7% | 32% | 18% |
| `epsilon_greedy` | 0.186 | R4.97 | 4,836 ± 359 | 65.9% | 28% | 13% |
| `static` | 0.172 | R4.42 | 5,563 ± 703 | 60.8% | 17% | 0% |
| `random` | 0.149 | R5.80 | 6,728 ± 17 | 52.6% | 0% | -21% |

_50,000 app opens per run, behaviour shift at 25,000, mean ± standard deviation over 10 seeds. Regret is take-ups lost against an oracle that knows the hidden take-up probabilities._
<!-- END generated:md-results -->

<!-- BEGIN generated:md-findings -->
1. **Learning per segment beats both baselines, in every run.** thompson_segment_decay reached 92.7% of the oracle, with 53% more take-ups per app open than the static batch model (0.263 against 0.172). It cut regret by 85% against random and 81% against static: below random's regret in 10 of 10 seeds and below static's in 10 of 10.
2. **It adapts when behaviour changes; the batch model does not.** After the shift its take-up rose from 0.231 to 0.295 per app open; the static model went from 0.168 to 0.175.
3. **The top of the table is a tie.** thompson_segment_decay has the lowest mean regret, but thompson_segment, thompson_segment_payday had lower regret in at least a fifth of the seeds, so the order among them is within the noise.
4. **Personalisation is the biggest single lever.** Thompson sampling with statistics per segment reached 92.4% of the oracle; one global model reached 67.7% (lower regret per segment in 10 of 10 seeds).
5. **Forgetting trades a little before the shift for more after it.** With forgetting, regret after the shift was lower in 10 of 10 seeds, by 130 take-ups on average; before the shift it cost 93 on average (higher in 8 of 10 seeds). Its value grows with how often, and how sharply, behaviour changes.
6. **The reward decides the business outcome.** The same policy learning from margin instead of take-up earned 38% more revenue per app open (R13.37 against R9.68, more in 10 of 10 seeds), at the cost of 22% fewer take-ups. Which one is right is a business choice, not a modelling one.
7. **Thompson sampling explores better than epsilon-greedy.** With the same per-segment statistics, regret was 1,078 against 1,690, lower in 8 of 10 seeds: epsilon-greedy keeps spending a fixed share of slots on offers it already knows are poor.
8. **Forgetting adapts sooner.** After the shift, with forgetting the young professionals' new favourite led within 1,460 app opens, against 5,601 app opens without; for families, 7,980 app opens against not within the run. The static model adapted for neither segment (first seed; see the adaptation table).
<!-- END generated:md-findings -->

The second finding matters most for the brief. Before the shift the static model looks respectable, because it was fitted on the right month; after it, the model is wrong and stays wrong, while the learning policies find the new favourites.

## Where the brief's concepts live here

The brief names the concepts below. This project implements each one from scratch; it does not use ecosystem.Ai's product.

| Concept | In this project |
|---|---|
| Dynamic recommender, real-time scoring | `POST /invocations` ranks from the current beliefs on every request; no batch step |
| Options store | The `options` table: one row per policy, context and offer, holding the evidence; the database computes α and β, so they can't disagree with the counts |
| Real-time feedback | `POST /response` updates the belief before it returns; the next app open uses it |
| Epsilon-greedy and Thompson sampling | Both, with global, per-segment and per-segment-and-payday contexts |
| Forgetting | Discounted evidence per context, so old behaviour fades; measured against no forgetting, before and after the shift |
| Presentation and response logging | The event log: each app open with its full context, each offer shown and its position, each response; the model can be rebuilt from it |

## What production would need

- **Learning off the request path.** Today a response waits for its own update. I would publish responses to a stream (Kafka, or a Postgres outbox) and let a consumer apply them: seconds of freshness traded for independence from database latency. It also removes a contention point: forgetting touches every row in a context, so updates to one segment serialize. Under load in Docker, four workers deadlocked on it until I added a per-context lock.
- **Throughput.** One worker answers an app open in 11 ms (p95, one client) and saturates near 190 requests a second. Four workers sharing a Docker network with Postgres kept p95 under 25 ms up to four concurrent clients (224 requests a second); beyond that, commit flushes to Docker Desktop's virtual disk dominated. Next: more workers behind PgBouncer, beliefs cached in memory, and a load test on Linux servers rather than a laptop.
- **Richer context, where it pays.** Segments carry most of the value here (92% of the oracle against 68% for one global model); payday context only tied. With real data I would test a contextual bandit on customer features from a feature store (linear or logistic Thompson sampling) against this as the baseline.
- **Guardrails.** Eligibility is enforced already. Production also needs fatigue caps, frequency limits, business overrides and affordability checks before any credit offer: the pay-later voucher falls under the National Credit Act.
- **Measurement that never stops.** A small holdout on the static policy measures uplift continuously. Alerts fire when take-up drops, exploration collapses or beliefs move sharply. Logged ranking propensities would let new policies be evaluated offline before they serve anyone.
- **Delayed rewards, security, privacy.** Real take-ups can arrive hours later, so responses need an attribution window. The dashboard has accounts with roles, and the API takes optional API keys; production would use the company's single sign-on and a secrets manager instead, and customer data needs POPIA-compliant handling.

## Limitations

The world is simulated and I wrote its ground truth, so the results show that the mechanism works and how its parts compare, not the uplift in a real market. Simulated customers differ only through their segment and a few attributes; real customers differ in many more ways. The adaptation-speed numbers come from one seed, and the convergence threshold (best offer leads in 80% of the last 100 app opens) is a choice whose sensitivity I didn't test. These, and every other judgement call, are recorded in DECISIONS.md.
