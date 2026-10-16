"""Behaviour shift, told as a data story (Brent Dykes' storytelling arc).

Setting and hook: halfway through, a segment's favourite offer changes overnight, and a
model fitted on last month can't see it. Rising insights: the learner switches to the
new favourite (sooner with forgetting); take-ups follow; forgetting is why, at a small
price before the change. Then the answer, what to do next, and the interactive charts.
Every number comes from the saved run (recommender.evaluation.adaptation).
"""

import plotly.graph_objects as go
import streamlit as st

from recommender.config import load_offers
from recommender.evaluation.adaptation import (
    BASELINE,
    LEARNER,
    NO_FORGETTING,
    build_shift_story,
    first_share,
)
from recommender.evaluation.compare import moment_favourites
from recommender.evaluation.metrics import selection_share
from recommender.evaluation.story import label
from recommender.simulator.engine import World
from shared import colour, mark_shift, need_results, provenance, results_file, settings, style

MUTED = "#a3acb5"  # everything the current message is not about
SEGMENTS = {
    "young_professional": "young professionals",
    "family": "families",
    "student": "students",
    "pensioner": "pensioners",
    "small_business": "small businesses",
}
OFFER_NAMES = {offer.offer_id: offer.name for offer in load_offers()}
SWITCHED = 0.8  # the new favourite led 80% of the segment's last 100 app opens

rounds = results_file("comparison_rounds.parquet")
convergence = results_file("convergence.csv")
table = results_file("comparison.csv")
by_seed = results_file("comparison_by_seed.csv")
if not need_results(rounds, convergence, table, by_seed) or not rounds["shifted"].any():
    st.title("Behaviour shift")
    if rounds is not None and not rounds["shifted"].any():
        st.info("This run had no behaviour shift.")
    st.stop()


@st.cache_data
def favourites() -> tuple[dict[str, str], dict[str, str]]:
    """Each segment's right first offer for most of the month, before and after the shift."""
    world = World.load()
    return tuple(
        {
            segment: offer
            for (segment, payday), offer in moment_favourites(truth, world).items()
            if not payday
        }
        for truth in (world.truth, world.truth.after_shift())
    )


def name(offer: str) -> str:
    """An offer's name as customers see it."""
    return OFFER_NAMES.get(offer, offer)


def opens(count: float | None) -> str:
    """'1,460 app opens', or that it never happened within the run."""
    return f"{count:,.0f} app opens" if count is not None else "it hadn't by the end of the run"


old, new = favourites()
changed = [segment for segment in SEGMENTS if old.get(segment) != new.get(segment)]
stories = {
    segment: build_shift_story(table, by_seed, rounds, convergence, (old, new), segment)
    for segment in changed
}
overall = stories[changed[0]]  # the overall numbers are the same in every segment's story

# --- headline and setting ------------------------------------------------------------------
st.title(
    f"When customers changed, the learner followed: {overall.fewer_than_baseline:.0%} fewer "
    "take-ups lost than the fixed ranking"
)
st.caption(
    "Behaviour shift · read top to bottom: the change, three findings, the answer, and what "
    "to do next. The interactive charts are at the end."
)
provenance()
changes = "; ".join(
    f"{SEGMENTS[s]} went from the {name(old[s])} to the {name(new[s])}" for s in changed
)
st.markdown(
    f"Halfway through the run, after {overall.shift_round:,} app opens, two favourites "
    f"changed overnight: {changes}. Nothing told the policies; they could only notice it in "
    f"what customers took up."
)

# --- 1. the hook -----------------------------------------------------------------------------
st.header("1 · The change: last month's favourite stopped being the best offer", divider="gray")
segment = st.selectbox(
    "Follow one segment",
    changed,
    format_func=lambda key: SEGMENTS[key].capitalize(),
)
story = stories[segment]
with st.container(border=True):
    st.markdown(f"#### {name(story.old_favourite)} → {name(story.new_favourite)}")
    st.markdown(
        f"That was the change for {SEGMENTS[segment]}. A model fitted on last month can't see "
        f"it: after the change, the {label(BASELINE).lower()} put the {name(story.new_favourite)} "
        f"first for {SEGMENTS[segment]} in **{story.baseline_new_first:.0%}** of their app "
        f"opens."
    )

# --- 2. rising insights ----------------------------------------------------------------------
st.header("2 · What we found", divider="gray")

# 2a: the learner switched; forgetting made it sooner.
learner_at, slow_at = story.adapted_after[LEARNER], story.adapted_after[NO_FORGETTING]
st.subheader(
    f"The learner switched to the new favourite within {opens(learner_at)}; without "
    f"forgetting, {opens(slow_at)}"
    if learner_at is not None
    else "In this run the learner didn't switch to the new favourite"
)
figure = go.Figure()
figure.add_hline(
    y=SWITCHED,
    line_dash="dot",
    line_color=MUTED,
    annotation_text="switched",
    annotation_position="bottom right",
)
for policy in (BASELINE, NO_FORGETTING, LEARNER):
    share = first_share(rounds, policy, segment, story.new_favourite)
    figure.add_scatter(
        x=share.index,
        y=share.to_numpy(),
        mode="lines",
        name=label(policy),
        line={"color": colour(policy), "width": 3 if policy == LEARNER else 2},
        hovertemplate="%{x:,} app opens: %{y:.0%} first<extra></extra>",
    )
for at, text, ax in ((learner_at, "with forgetting", -120), (slow_at, "without", 90)):
    if at is not None:
        figure.add_annotation(
            x=story.shift_round + at, y=SWITCHED, ax=ax, ay=70, text=f"{text}: {at:,.0f}"
        )
figure.update_layout(
    xaxis_title="App opens",
    yaxis_title="Share with the new favourite first",
    yaxis_tickformat=".0%",
    yaxis_range=[0, 1.05],
    hovermode="closest",
    legend={"orientation": "h", "y": 1.12, "x": 0},
)
st.plotly_chart(mark_shift(style(figure, height=400), story.shift_round), width="stretch")
st.caption(
    f"Share of the last 400 app opens by {SEGMENTS[segment]} with the "
    f"{name(story.new_favourite)} ranked first, on the first run. A policy has switched once "
    f"the new favourite led 80% of the segment's last 100 app opens; the counts are app opens "
    f"by all customers since the change. The {label(BASELINE).lower()} never switched: it was "
    f"fitted before the change."
)

# 2b: take-ups followed the switch.
(lb, la), (bb, ba) = story.take_ups[LEARNER], story.take_ups[BASELINE]
st.subheader(
    f"Take-ups from {SEGMENTS[segment]} followed: {lb:,.0f} → {la:,.0f} per 1,000 app opens "
    f"for the learner; the fixed ranking went from {bb:,.0f} to {ba:,.0f}"
)
figure = go.Figure()
for policy in (BASELINE, NO_FORGETTING, LEARNER):
    before, after = story.take_ups[policy]
    figure.add_scatter(
        x=["Before the change", "After the change"],
        y=[before, after],
        mode="lines+markers+text",
        name=label(policy),
        line={"color": colour(policy), "width": 3 if policy == LEARNER else 2},
        marker={"size": 10, "color": colour(policy)},
        text=[f"{before:,.0f}", f"{after:,.0f}"],
        # The two learners start close together: one label above, one below.
        textposition={
            LEARNER: ["top left", "top right"],
            NO_FORGETTING: ["bottom left", "bottom right"],
        }.get(policy, ["middle left", "middle right"]),
        hovertemplate=f"{label(policy)}: %{{y:,.0f}} per 1,000 app opens<extra></extra>",
    )
figure.update_layout(
    yaxis_title=f"Take-ups per 1,000 app opens ({SEGMENTS[segment]})",
    hovermode="closest",
    xaxis={"range": [-0.4, 1.4]},
    legend={"orientation": "h", "y": 1.12, "x": 0},
)
st.plotly_chart(style(figure, height=360), width="stretch")
st.caption(
    "Expected take-ups (what each choice was worth on average) per 1,000 of the segment's app "
    "opens, before and after the change, on the first run. The new favourite is taken up more "
    "often than the old one was, so a policy that finds it gains."
)

# 2c: forgetting is why, at a small price.
st.subheader("Forgetting is why: it costs a little before a change and saves more after it")
cost, saving = story.forgetting_cost_before, story.forgetting_saving_after
figure = go.Figure(
    go.Bar(
        x=["Before the change", "After the change"],
        y=[-cost, saving],
        marker={"color": [MUTED, colour(LEARNER)]},
        text=[f"{-cost:+,.0f}", f"{saving:+,.0f}"],
        textposition="outside",
        hovertemplate="%{x}: %{y:+,.0f} take-ups<extra></extra>",
        width=0.45,
    )
)
figure.add_hline(y=0, line_color=MUTED, line_width=1)
span = max(cost, saving) * 1.35
figure.update_layout(
    yaxis_title="Take-ups saved by forgetting",
    yaxis_range=[-span, span],
    showlegend=False,
    hovermode="closest",
)
st.plotly_chart(style(figure, height=320), width="stretch")
decay = settings().policies[LEARNER].decay
st.caption(
    f"The same learner with and without forgetting, across all segments, mean over "
    f"{overall.seeds} runs. Before the change, forgetting old evidence (each update keeps "
    f"{decay} of it) cost {cost:,.0f} take-ups: it lost more in "
    f"{overall.seeds_cost_before} of {overall.seeds} runs. After the change it saved "
    f"{saving:,.0f}: it lost fewer in {overall.seeds_saved_after} of {overall.seeds} runs."
)

# --- 3. the aha moment -------------------------------------------------------------------------
st.header("3 · The answer", divider="gray")
with st.container(border=True):
    st.markdown("### Adapting is where learning online pays")
    st.markdown(
        f"After the change, the learner lost **{overall.regret_after[LEARNER]:,.0f}** take-ups "
        f"to the oracle; the {label(BASELINE).lower()} lost "
        f"**{overall.regret_after[BASELINE]:,.0f}**. That is "
        f"**{overall.fewer_than_baseline:.0%} fewer**, in **{overall.beats_baseline_after} of "
        f"{overall.seeds} runs**. Forgetting is what makes it quick: it switched "
        f"{SEGMENTS[segment]} to the new favourite within {opens(learner_at)}, against "
        f"{opens(slow_at)} without."
    )

# --- 4. what to do next ------------------------------------------------------------------------
st.header("4 · What to do next", divider="gray")
memory = 1 / (1 - decay) if decay < 1 else None
steps = [
    "**Keep forgetting on wherever demand moves:** launches, price changes, load-shedding, "
    "seasons, a competitor's offer. The small cost in quiet months buys a fast recovery.",
    "**Match the memory to how fast your market changes.** "
    + (
        f"Today it is about {memory:,.0f} responses per segment (`decay: {decay}`). "
        if memory
        else ""
    )
    + "Faster-moving products want a shorter memory; stable ones a longer one.",
    "**Catch changes early.** A segment's take-up dropping, or exploration jumping (the "
    "Explore vs exploit page shows it right after this change), is the first sign; alert on "
    "them instead of waiting for a monthly report.",
]
st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))

# --- the detail, for analysts ------------------------------------------------------------------
st.divider()
with st.expander("Explore the change: any policy, any window, which offer led"):
    policies = sorted(rounds["policy"].unique())
    chosen = st.multiselect(
        "Policies",
        policies,
        default=[p for p in (LEARNER, NO_FORGETTING, BASELINE) if p in policies],
    )
    window = st.slider("Rolling window (app opens of this segment)", 100, 1000, 400, step=100)
    share, take_up = go.Figure(), go.Figure()
    for policy in chosen:
        segment_opens = rounds[(rounds["policy"] == policy) & (rounds["segment"] == segment)]
        line = {"color": colour(policy), "width": 2}
        share.add_scatter(
            x=segment_opens["round"],
            y=(segment_opens["top_offer"] == story.new_favourite).rolling(window).mean(),
            name=policy,
            mode="lines",
            line=line,
        )
        take_up.add_scatter(
            x=segment_opens["round"],
            y=segment_opens["take_ups"].rolling(window).mean(),
            name=policy,
            mode="lines",
            line=line,
        )
    st.markdown(f"**Share of app opens with the {name(story.new_favourite)} first**")
    share.update_layout(xaxis_title="App opens", yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(mark_shift(style(share, 360), story.shift_round), width="stretch")
    st.markdown("**Take-up per app open for this segment**")
    take_up.update_layout(xaxis_title="App opens")
    st.plotly_chart(mark_shift(style(take_up, 320), story.shift_round), width="stretch")

    st.markdown("**Which offer led, over time**")
    pick = st.selectbox("Policy", chosen or policies, key="share_policy")
    blocks = selection_share(
        rounds[rounds["policy"] == pick].reset_index(drop=True), segment, every=1000
    )
    area = go.Figure()
    for offer in blocks.columns:
        area.add_scatter(
            x=blocks.index,
            y=blocks[offer],
            name=name(offer),
            stackgroup="share",
            mode="lines",
            line={"width": 0.5},
        )
    area.update_layout(
        xaxis_title="App opens", yaxis_tickformat=".0%", yaxis_title="Share ranked first"
    )
    st.plotly_chart(mark_shift(style(area, 360), story.shift_round), width="stretch")
