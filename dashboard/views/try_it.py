"""Try it: be the customer. Open the app, respond to an offer, watch the model update."""

import datetime as dt

import streamlit as st

from auth import require_admin
from shared import api

require_admin()  # every click changes the model
st.title("Try it")
client = api()
st.caption(f"Calling the API: {client.where}. Every click below is real learning in the database.")

policies = {p["name"]: p for p in client.get("/policies")}
active = next(name for name, p in policies.items() if p["active"])

left, middle, right = st.columns(3)
segment = left.selectbox(
    "Segment",
    ["student", "young_professional", "family", "small_business", "pensioner"],
    format_func=lambda s: s.replace("_", " "),
)
customers = [c["customer_id"] for c in client.get("/customers", segment=segment, limit=50)]
customer_id = middle.selectbox("Customer", customers)
policy = right.selectbox("Policy", list(policies), index=list(policies).index(active))

with st.expander("The moment (defaults to now)"):
    day = st.date_input("Day", value=dt.date.today())
    time_of_day = st.selectbox("Time of day", ["morning", "afternoon", "evening", "night"])
    payday = st.radio("Payday week", ["from the calendar", "yes", "no"], horizontal=True)


def context_key(mode: str, context: dict) -> str:
    """The key a policy pools this context's evidence under (mirrors the API's)."""
    if mode == "global":
        return "all"
    if mode == "segment_payday":
        return f"{context['segment']}|{'payday' if context['is_payday_week'] else 'rest'}"
    return context["segment"]


def belief(name: str, key: str, offer_id: str) -> dict | None:
    """One offer's current belief under a policy and context key, or None."""
    arms = client.get(f"/policies/{name}/state", context_key=key)["arms"]
    return next((arm for arm in arms if arm["offer_id"] == offer_id), None)


if st.button("Open the app", type="primary"):
    overrides = {"day": day.isoformat(), "time_of_day": time_of_day}
    if payday != "from the calendar":
        overrides["is_payday_week"] = payday == "yes"
    reply = client.post(
        "/invocations",
        {"customer_id": customer_id, "context": overrides},
        policy=policy,
    )
    if reply.status_code != 200:
        st.error(reply.json().get("detail", reply.text))
    else:
        st.session_state.invocation = reply.json()
        st.session_state.answers = {}

invocation = st.session_state.get("invocation")
if invocation:
    ctx = invocation["context"]
    mode = policies[invocation["policy"]]["context"]
    key = context_key(mode, ctx)
    st.markdown(
        f"**{invocation['customer_id']}** · {ctx['segment'].replace('_', ' ')} · "
        f"{ctx['day_of_week']} {ctx['time_of_day']} · "
        f"{'payday week' if ctx['is_payday_week'] else 'rest of month'} · "
        f"served by `{invocation['policy']}` (learns into context `{key}`)"
    )
    answers = st.session_state.answers
    for offer in invocation["offers"]:
        with st.container(border=True):
            info, actions = st.columns([3, 2])
            info.markdown(
                f"**{offer['rank']}. {offer['name']}**  \n"
                f"R{offer['price_zar']:,.0f} · {offer['category']} · score {offer['score']:.3f}"
            )
            done = answers.get(offer["offer_id"])
            if done:
                actions.success(done)
                continue
            buttons = actions.columns(3)
            for button, (label, outcome) in zip(
                buttons,
                [("Take up", "taken_up"), ("Viewed", "viewed"), ("Ignore", "ignored")],
                strict=True,
            ):
                if button.button(
                    label, key=f"{invocation['invocation_id']}-{offer['offer_id']}-{outcome}"
                ):
                    before = belief(invocation["policy"], key, offer["offer_id"])
                    result = client.post(
                        "/response",
                        {
                            "invocation_id": invocation["invocation_id"],
                            "offer_id": offer["offer_id"],
                            "outcome": outcome,
                        },
                    ).json()
                    after = belief(invocation["policy"], key, offer["offer_id"])
                    if result.get("model_updated"):
                        old = f"{before['mean']:.1%}" if before else "the prior (50.0%)"
                        summary = (
                            f"{outcome.replace('_', ' ')}: reward {result['reward']:.2f}; "
                            f"mean take-up {old} → {after['mean']:.1%} "
                            f"(evidence {after['pulls']:.1f})"
                        )
                    else:
                        summary = f"{outcome.replace('_', ' ')}: logged, no model update"
                    answers[offer["offer_id"]] = summary
                    st.rerun()
    st.caption(
        "Taking an offer up adds 1 to α; viewing without taking adds 1 to β; ignoring changes "
        "nothing, because the customer probably never saw it."
    )
