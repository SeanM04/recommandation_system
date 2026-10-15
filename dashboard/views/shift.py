"""Behaviour shift: how fast each policy notices that the right answer changed."""

import plotly.graph_objects as go
import streamlit as st

from recommender.evaluation.compare import moment_favourites
from recommender.evaluation.metrics import selection_share
from recommender.simulator.engine import World
from shared import colour, mark_shift, need_results, provenance, results_file, style

st.title("Behaviour shift")
st.caption(
    "Halfway through, a streaming bundle goes viral with young professionals, and an "
    "electricity tariff hike makes the cashback offer families' first choice. A batch model "
    "keeps serving yesterday's answer; a policy that forgets old evidence notices sooner."
)

rounds = results_file("comparison_rounds.parquet")
convergence = results_file("convergence.csv")
if not need_results(rounds, convergence):
    st.stop()
provenance()
shift_round = int(rounds.loc[rounds["shifted"], "round"].min())


@st.cache_data
def new_favourites() -> dict[str, str]:
    """Each segment's right first offer for most of the month, after the shift."""
    world = World.load()
    favourites = moment_favourites(world.truth.after_shift(), world)
    return {segment: offer for (segment, payday), offer in favourites.items() if not payday}


favourites = new_favourites()
left, right = st.columns([2, 3])
segment = left.selectbox(
    "Segment", ["young_professional", "family"], format_func=lambda s: s.replace("_", " ")
)
policies = sorted(rounds["policy"].unique())
chosen = right.multiselect(
    "Policies",
    policies,
    default=[p for p in ["thompson_segment_decay", "thompson_segment", "static"] if p in policies],
)
favourite = favourites[segment]
window = st.slider("Rolling window (app opens of this segment)", 100, 1000, 400, step=100)

share, take_up = go.Figure(), go.Figure()
for policy in chosen:
    opens = rounds[(rounds["policy"] == policy) & (rounds["segment"] == segment)]
    first = (opens["top_offer"] == favourite).rolling(window).mean()
    line = {"color": colour(policy), "width": 2}
    share.add_scatter(x=opens["round"], y=first, name=policy, mode="lines", line=line)
    take_up.add_scatter(
        x=opens["round"],
        y=opens["take_ups"].rolling(window).mean(),
        name=policy,
        mode="lines",
        line=line,
    )

st.subheader(f"Share of {segment.replace('_', ' ')} app opens with {favourite} ranked first")
share.update_layout(xaxis_title="App opens", yaxis_tickformat=".0%", yaxis_range=[0, 1])
st.plotly_chart(mark_shift(style(share, 380), shift_round), use_container_width=True)

after = convergence[
    (convergence["phase"] == "after the shift") & (convergence["segment"] == segment)
]
columns = st.columns(max(len(chosen), 1))
for column, policy in zip(columns, chosen, strict=False):
    value = after.loc[after["policy"] == policy, "app_opens"]
    opens = value.iloc[0] if len(value) else float("nan")
    column.metric(
        f"{policy}: app opens to adapt",
        "not within the run" if opens != opens else f"{opens:,.0f}",  # NaN check
    )

st.subheader("Take-up per app open for this segment")
take_up.update_layout(xaxis_title="App opens")
st.plotly_chart(mark_shift(style(take_up, 340), shift_round), use_container_width=True)

st.subheader("Which offer led, over time")
pick = st.selectbox("Policy", chosen or policies, key="share_policy")
blocks = selection_share(
    rounds[rounds["policy"] == pick].reset_index(drop=True), segment, every=1000
)
area = go.Figure()
for offer in blocks.columns:
    area.add_scatter(x=blocks.index, y=blocks[offer], name=offer, stackgroup="share", mode="lines",
                     line={"width": 0.5})  # fmt: skip
area.update_layout(
    xaxis_title="App opens", yaxis_tickformat=".0%", yaxis_title="Share ranked first"
)
st.plotly_chart(mark_shift(style(area, 360), shift_round), use_container_width=True)
