"""Live view: events arriving in the log, and what the serving policy believes right now."""

import pandas as pd
import streamlit as st
from sqlalchemy import desc, func, select

from recommender.events.log import impressions, invocations, responses
from shared import engine, settings, store

st.title("Live view")
st.caption(
    "App opens and responses as they reach the event log in the database, and the serving "
    "policy's current beliefs. Refreshes every 3 seconds: start a simulation through the API "
    "(`python scripts/run_simulation.py --mode api`) to watch it learn."
)

OUTCOME_MARK = {"taken_up": "taken up", "viewed": "viewed", "ignored": "ignored"}


@st.fragment(run_every="3s")
def live() -> None:
    """Counts, the latest app opens and the serving policy's beliefs, every 3 seconds."""
    with engine().connect() as connection:
        total = connection.execute(select(func.count()).select_from(invocations)).scalar()
        outcomes = dict(
            connection.execute(
                select(responses.c.outcome, func.count()).group_by(responses.c.outcome)
            ).all()
        )
        recent = connection.execute(
            select(invocations).order_by(desc(invocations.c.created_at)).limit(15)
        ).all()
        ids = [row.invocation_id for row in recent]
        shown = connection.execute(
            select(impressions).where(impressions.c.invocation_id.in_(ids))
        ).all()
        answered = connection.execute(
            select(responses).where(responses.c.invocation_id.in_(ids))
        ).all()

    if not total:
        st.info("The event log is empty. Run `make simulate`, or send requests to the API.")
        return

    active = settings().recommendation.active_policy
    taken = outcomes.get("taken_up", 0)
    columns = st.columns(3)
    columns[0].metric("App opens logged", f"{total:,}")
    columns[1].metric("Offers taken up", f"{taken:,}")
    columns[2].metric("Take-up per app open", f"{taken / total:.3f}")
    st.caption(f"Serving policy: `{active}`")

    outcome_of = {(r.invocation_id, r.offer_id): r.outcome for r in answered}
    lists: dict[str, list[str]] = {}
    for row in sorted(shown, key=lambda r: r.position):
        mark = OUTCOME_MARK.get(outcome_of.get((row.invocation_id, row.offer_id)), "waiting")
        lists.setdefault(row.invocation_id, []).append(f"#{row.position} {row.offer_id} ({mark})")
    st.subheader("Latest app opens")
    st.table(
        pd.DataFrame(
            {
                "time (UTC)": [row.created_at.strftime("%H:%M:%S") for row in recent],
                "customer": [row.customer_id for row in recent],
                "segment": [row.segment for row in recent],
                "policy": [row.policy for row in recent],
                "offers shown, and what happened": [
                    "  ·  ".join(lists.get(row.invocation_id, [])) for row in recent
                ],
            }
        ).style.hide(axis="index")
    )

    st.subheader(f"What {active} believes now")
    rows = store().rows(active)
    if not rows:
        st.caption("No learning yet for this policy.")
        return
    beliefs = pd.DataFrame(
        {
            "context": row.context_key,
            "offer": row.offer_id,
            "mean": row.stats.mean,
            "evidence": row.stats.pulls,
        }
        for row in rows
    )
    top = (
        beliefs.sort_values("mean", ascending=False)
        .groupby("context")
        .head(3)
        .assign(rank=lambda frame: frame.groupby("context").cumcount() + 1)
    )
    top["offer (mean take-up)"] = top["offer"] + "  (" + top["mean"].map("{:.1%}".format) + ")"
    st.table(
        top.pivot(index="context", columns="rank", values="offer (mean take-up)").rename(
            columns={1: "1st", 2: "2nd", 3: "3rd"}
        )
    )
    st.caption("Ranked by posterior mean; Thompson sampling still explores around these.")


live()
