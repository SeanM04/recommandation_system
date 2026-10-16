"""Explore vs exploit, told as a data story (Brent Dykes' storytelling arc).

Setting and hook: the dilemma, and how much the learner explored at first. Rising
insights: its doubt decides how much it explores; exploration fades as it grows confident
while epsilon-greedy never stops; forgetting keeps enough doubt to notice a change. Then
the answer, what to do next, and the belief explorer for anyone who wants to dig.
Every number comes from the saved run (recommender.evaluation.exploration).
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from recommender.config import load_offers
from recommender.evaluation.exploration import (
    FIXED_RATE,
    LEARNER,
    NO_FORGETTING,
    build_explore_story,
    chance_of_best,
    evidence,
    exploration_share,
)
from recommender.evaluation.story import label
from shared import (
    colour,
    mark_shift,
    need_results,
    provenance,
    results_dir,
    results_file,
    settings,
    style,
)

MUTED = "#a3acb5"  # everything the current message is not about
SEGMENTS = {
    "young_professional": "young professionals",
    "family": "families",
    "student": "students",
    "pensioner": "pensioners",
    "small_business": "small businesses",
}

OFFER_NAMES = {offer.offer_id: offer.name for offer in load_offers()}

snapshots = results_file("comparison_snapshots.parquet")
rounds = results_file("comparison_rounds.parquet")
table = results_file("comparison.csv")
by_seed = results_file("comparison_by_seed.csv")
if not need_results(snapshots, rounds, table, by_seed):
    st.title("Explore vs exploit")
    st.stop()


@st.cache_data(max_entries=4)
def explore_numbers(saved_at: float):
    """The story and both exploration lines, computed once per saved run (keyed by its time)."""
    return (
        build_explore_story(rounds, table, by_seed),
        exploration_share(rounds, LEARNER).set_index("round")["share"],
        exploration_share(rounds, FIXED_RATE).set_index("round")["share"],
    )


story, learner_line, fixed_line = explore_numbers(
    (results_dir() / "comparison_rounds.parquet").stat().st_mtime
)
shifted = rounds.loc[rounds["shifted"], "round"]
shift_round = int(shifted.min()) if len(shifted) else None


@st.cache_data
def best_chances(policy: str, context: str, at: int):
    """Chance of being best per offer, cached: the Monte Carlo draws are the slow part."""
    return chance_of_best(snapshots, policy, context, at)


# --- headline and setting ------------------------------------------------------------------
st.title(
    f"It explores while it's unsure, then commits: {story.early:.0%} of app opens at first, "
    f"about {story.settled:.0%} once confident"
)
st.caption(
    "Explore vs exploit · read top to bottom: the dilemma, three findings, the answer, and "
    "what to do next. The belief explorer is at the end."
)
provenance()
st.markdown(
    "Every app open is a choice. **Exploit**: put first the offer the learner believes is "
    "best. **Explore**: try one it's less sure about, in case it's better. Explore too "
    "little and it never finds the better offer; explore too much and it wastes app opens "
    "on offers it already knows are poor. Thompson sampling decides by drawing once from its "
    "belief about each offer, so an offer leads as often as it might be the best."
)

# --- 1. the hook -----------------------------------------------------------------------------
st.header(
    "1 · The dilemma: exploring costs take-ups now, not exploring costs them forever",
    divider="gray",
)
left, right = st.columns([1, 2])
left.metric("Explored in the first 2,500 app opens", f"{story.early:.0%}")
right.markdown(
    f"At the start the learner knows nothing: every offer could be the best. In its first "
    f"2,500 app opens, **{story.early:.0%}** of first slots went to an offer other than "
    f"the one it showed most. Was that waste, or is that how it learns which offer each "
    f"segment wants?"
)

# --- 2. rising insights ----------------------------------------------------------------------
st.header("2 · What we found", divider="gray")

# 2a: doubt decides how much it explores.
keys = sorted(
    k
    for k in snapshots.loc[snapshots["policy"] == LEARNER, "context_key"].unique()
    if k in SEGMENTS
)
segment = st.selectbox(
    "Follow one segment",
    keys,
    index=keys.index("young_professional") if "young_professional" in keys else 0,
    format_func=lambda key: SEGMENTS[key].capitalize(),
)
saved = sorted(snapshots.loc[snapshots["policy"] == LEARNER, "round"].unique())
early_at = int(saved[0])
late_at = int(max(r for r in saved if r <= shift_round) if shift_round else saved[-1])
early, late = best_chances(LEARNER, segment, early_at), best_chances(LEARNER, segment, late_at)
favourite = late.index[0]
favourite_name = OFFER_NAMES.get(favourite, favourite)
st.subheader(
    f"Its doubt decides how much it explores: for {SEGMENTS[segment]}, the {favourite_name} "
    f"went from a {early.get(favourite, 0):.0%} to a {late[favourite]:.0%} chance of being best"
)
panels = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    horizontal_spacing=0.08,
    subplot_titles=[
        f"After {early_at:,} app opens",
        f"After {late_at:,}" + (" (just before the shift)" if shift_round else ""),
    ],
)
offers = list(dict.fromkeys([*late.index[:5], *early.index[:5]]))[:6]
for col, chances in ((1, early), (2, late)):
    shown = [chances.get(offer, 0.0) for offer in offers]
    others = max(0.0, 1 - sum(shown))
    panels.add_bar(
        x=[*shown, others],
        y=[OFFER_NAMES.get(o, o) for o in offers] + ["All other offers"],
        orientation="h",
        marker={"color": [colour(LEARNER) if o == favourite else MUTED for o in offers] + [MUTED]},
        text=[f"{v:.0%}" for v in [*shown, others]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1%} chance of being best<extra></extra>",
        row=1,
        col=col,
    )
panels.update_xaxes(range=[0, 1.18], tickformat=".0%", showgrid=False)
panels.update_yaxes(autorange="reversed")
panels.update_layout(showlegend=False, hovermode="closest", bargap=0.3)
st.plotly_chart(style(panels, height=340), width="stretch")
st.caption(
    f"Chance of being best, from the learner's beliefs ({label(LEARNER).lower()}). Thompson "
    f"sampling puts an offer first as often as it might be the best, so these bars are also "
    f"how often each offer led. Early on, the other offers still had a "
    f"{1 - early.get(favourite, 0):.0%} chance between them, so they kept being tried; by "
    f"{late_at:,} app opens, {1 - late[favourite]:.0%}."
)

# 2b: exploration fades as confidence grows; epsilon-greedy never stops.
st.subheader("Exploration fades as confidence grows; epsilon-greedy never stops")
figure = go.Figure()
for policy, shares, width in ((FIXED_RATE, fixed_line, 2), (LEARNER, learner_line, 3)):
    figure.add_scatter(
        x=shares.index,
        y=shares.to_numpy(),
        mode="lines+markers",
        name=label(policy),
        line={"color": colour(policy), "width": width},
        marker={"size": 8},
        hovertemplate="%{x:,} app opens: %{y:.0%} explored<extra></extra>",
    )
figure.add_annotation(
    x=2500, y=story.early, ax=40, ay=-10, xanchor="left", text=f"<b>{story.early:.0%}</b> at first"
)
if shift_round and shift_round in learner_line.index:
    figure.add_annotation(
        x=shift_round,
        y=learner_line[shift_round],
        ax=-70,
        ay=35,
        text=f"about {story.settled:.0%} once confident",
    )
    after = learner_line[learner_line.index > shift_round]
    if len(after):
        figure.add_annotation(
            x=after.index[0], y=after.iloc[0], ax=55, ay=-30, text="trying and switching"
        )
figure.add_annotation(
    x=fixed_line.index[-1],
    y=fixed_line.iloc[-1],
    ax=-40,
    ay=-35,
    text=f"epsilon-greedy: about {story.fixed_rate_settled:.0%}, always",
)
figure.update_layout(
    xaxis_title="App opens",
    yaxis_title="App opens spent exploring",
    yaxis_tickformat=".0%",
    hovermode="closest",
    legend={"orientation": "h", "y": 1.12, "x": 0},
)
st.plotly_chart(mark_shift(style(figure, height=380), shift_round), width="stretch")
st.caption(
    "Per 2,500 app opens, the share whose first offer wasn't the one the policy showed first "
    "most often in that period, across all segments, on the first run. Epsilon-greedy "
    f"explores a fixed share of the time: it never fell below {story.fixed_rate_floor:.0%}, "
    "even for offers it had long known were poor. Right after the shift, the share also counts "
    "switching: the old and new favourites share that period."
)

# 2c: forgetting keeps enough doubt to notice a change.
st.subheader("Forgetting keeps just enough doubt to notice a change")
remembered, piled_up = (
    evidence(snapshots, LEARNER, segment),
    evidence(snapshots, NO_FORGETTING, segment),
)
figure = go.Figure()
for policy, series in ((NO_FORGETTING, piled_up), (LEARNER, remembered)):
    figure.add_scatter(
        x=series.index,
        y=series.to_numpy(),
        mode="lines",
        name=label(policy),
        line={"color": colour(policy), "width": 3 if policy == LEARNER else 2},
        hovertemplate="%{x:,} app opens: %{y:,.0f} responses<extra></extra>",
    )
    figure.add_annotation(
        x=series.index[-1], y=series.iloc[-1], ax=-50, ay=-25, text=f"{series.iloc[-1]:,.0f}"
    )
figure.update_layout(
    xaxis_title="App opens",
    yaxis_title="Responses behind the beliefs",
    hovermode="closest",
    legend={"orientation": "h", "y": 1.12, "x": 0},
)
st.plotly_chart(mark_shift(style(figure, height=360), shift_round), width="stretch")
decay = settings().policies[LEARNER].decay
memory = 1 / (1 - decay) if decay < 1 else None
caption = (
    f"Responses the beliefs about {SEGMENTS[segment]} rest on. Without forgetting they pile "
    f"up and the beliefs harden, so a change takes longer to outweigh the past."
)
if memory:
    caption += (
        f" With forgetting (each update keeps {decay} of the old evidence), they level off "
        f"near {memory:,.0f}, so the beliefs stay open to new information."
    )
if story.after_shift is not None and story.after_shift_no_forgetting is not None:
    caption += (
        f" In the 2,500 app opens after the shift, {story.after_shift:.0%} of the learner's first "
        f"offers differed from its most-shown one as it tried and switched to new favourites; "
        f"without forgetting, {story.after_shift_no_forgetting:.0%}. The Behaviour shift page "
        f"shows how much sooner that found the new favourites."
    )
st.caption(caption)

# --- 3. the aha moment -------------------------------------------------------------------------
st.header("3 · The answer", divider="gray")
with st.container(border=True):
    st.markdown("### Explore while unsure, commit when sure")
    answer = (
        f"Thompson sampling explored **{story.early:.0%}** of app opens at the start and about "
        f"**{story.settled:.0%}** once confident; epsilon-greedy explored about "
        f"**{story.fixed_rate_settled:.0%}** of the time, whatever it knew."
    )
    if story.regret and story.seeds_won:
        (thompson, greedy), (won, runs) = story.regret, story.seeds_won
        answer += (
            f" With the same customer segments and the same app opens, Thompson sampling lost "
            f"**{thompson:,.0f}** take-ups to the oracle against epsilon-greedy's "
            f"**{greedy:,.0f}**: fewer in **{won} of {runs} runs**."
        )
    st.markdown(answer)

# --- 4. what to do next ------------------------------------------------------------------------
st.header("4 · What to do next", divider="gray")
steps = [
    "**Let the beliefs set the exploration.** Thompson sampling has no exploration rate to "
    "tune: it explores in proportion to its doubt, offer by offer and segment by segment.",
    "**Match forgetting to how fast customers change.** "
    + (
        f"The learner keeps about {memory:,.0f} responses per segment (`decay: {decay}`). "
        if memory
        else ""
    )
    + "A shorter memory adapts faster but keeps exploring more; a longer one exploits more "
    "but notices change later.",
    "**Watch exploration as a health signal.** A share stuck near zero means the model has "
    "stopped questioning itself; a sudden rise means customers may be changing. Both can be "
    "read from the event log.",
]
st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))

# --- the detail, for analysts ------------------------------------------------------------------
st.divider()
with st.expander("Explore the beliefs: any policy, any context, any point in the run"):
    st.caption(
        "A wide curve is an offer the policy is unsure about: its draws sometimes land high, "
        "so it still gets shown (exploration). A narrow curve far to the right is an offer it "
        "is confident in (exploitation)."
    )
    highlight = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # top four offers; the rest grey
    thompson = sorted(p for p in snapshots["policy"].unique() if p.startswith("thompson"))
    left, middle, right = st.columns([2, 2, 3])
    policy = left.selectbox(
        "Policy", thompson, index=thompson.index(LEARNER) if LEARNER in thompson else 0
    )
    contexts = sorted(snapshots.loc[snapshots["policy"] == policy, "context_key"].unique())
    context = middle.selectbox(
        "Context", contexts, index=contexts.index("family") if "family" in contexts else 0
    )
    available = sorted(snapshots.loc[snapshots["policy"] == policy, "round"].unique())
    at = right.select_slider("After this many app opens", options=available, value=available[-1])

    beliefs = snapshots[(snapshots["policy"] == policy) & (snapshots["context_key"] == context)]
    now = beliefs[beliefs["round"] == at].copy()
    now["mean"] = now["alpha"] / (now["alpha"] + now["beta"])
    now = now.sort_values("mean", ascending=False)
    palette = {o: highlight[i] if i < 4 else MUTED for i, o in enumerate(now["offer_id"])}

    def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """The Beta(a, b) density at ``x``, computed in log space so large counts stay finite."""
        log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        return np.exp(log_norm + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x))

    x = np.linspace(0.001, 0.5, 400)
    figure = go.Figure()
    for row in now.itertuples():
        figure.add_scatter(
            x=x,
            y=beta_pdf(x, row.alpha, row.beta),
            name=row.offer_id,
            mode="lines",
            line={
                "color": palette[row.offer_id],
                "width": 2 if palette[row.offer_id] != MUTED else 1,
            },
        )
    figure.update_layout(
        xaxis_title="Take-up rate",
        yaxis_title="Belief (density)",
        xaxis_tickformat=".0%",
        hovermode="closest",
    )
    st.plotly_chart(style(figure, 380), width="stretch")

    draws = np.random.default_rng(0)
    now["90% interval"] = [
        "{:.1%} – {:.1%}".format(*np.quantile(draws.beta(r.alpha, r.beta, 4000), [0.05, 0.95]))
        for r in now.itertuples()
    ]
    st.dataframe(
        now[["offer_id", "mean", "90% interval", "alpha", "beta"]].rename(
            columns={"offer_id": "offer"}
        ),
        column_config={
            "mean": st.column_config.NumberColumn("Mean take-up", format="percent"),
            "alpha": st.column_config.NumberColumn("α", format="%.1f"),
            "beta": st.column_config.NumberColumn("β", format="%.1f"),
        },
        hide_index=True,
        width="stretch",
    )

    st.markdown("**How the beliefs moved**")
    history = beliefs.assign(mean=beliefs["alpha"] / (beliefs["alpha"] + beliefs["beta"]))
    trend = go.Figure()
    for offer, series in history.groupby("offer_id"):
        trend.add_scatter(
            x=series["round"],
            y=series["mean"],
            name=offer,
            mode="lines+markers",
            line={
                "color": palette.get(offer, MUTED),
                "width": 2 if palette.get(offer) != MUTED else 1,
            },
        )
    trend.update_layout(xaxis_title="App opens", yaxis_title="Mean take-up", yaxis_tickformat=".0%")
    st.plotly_chart(mark_shift(style(trend, 360), shift_round), width="stretch")
