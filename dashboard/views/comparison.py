"""Algorithm comparison, told as a data story (Brent Dykes' storytelling arc).

Setting and hook: what was possible, and what the usual fixed ranking leaves on the table.
Rising insights: three findings, one chart and one message each, with colour only on the
policies the message is about. Aha moment: the answer in one sentence. Then the
recommendation and next steps, and the full results for anyone who wants to dig.
Every number comes from the saved run (recommender.evaluation.story).
"""

import plotly.graph_objects as go
import streamlit as st

from recommender.algorithms.factory import servable_policies
from recommender.evaluation.metrics import cumulative
from recommender.evaluation.story import BASELINE, LADDER, build_story, gap_curve, label
from shared import (
    KEY_POLICIES,
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

table = results_file("comparison.csv")
by_seed = results_file("comparison_by_seed.csv")
rounds = results_file("comparison_rounds.parquet")
convergence = results_file("convergence.csv")
if not need_results(table, by_seed, rounds, convergence):
    st.title("Algorithm comparison")
    st.stop()


@st.cache_data(max_entries=4)
def comparison_story(saved_at: float):
    """The story's numbers, computed once per saved run (keyed by when it was saved)."""
    return build_story(table, by_seed, rounds, convergence, servable_policies(settings()))


story = comparison_story((results_dir() / "comparison_rounds.parquet").stat().st_mtime)
best, n = story.best, story.seeds
best_name, baseline_name = label(best), label(BASELINE).lower()
extra = story.per_1000[best] - story.per_1000[BASELINE]


def runs(count: int) -> str:
    """'7 of 10 runs': how many seeds back a claim."""
    return f"{count} of {n} run{'s' if n != 1 else ''}"


# --- headline and setting ------------------------------------------------------------------
st.title(f"{best_name}: {story.uplift:.0%} more take-ups than a fixed ranking")
st.caption(
    "Algorithm comparison · read top to bottom: the problem, three findings, the answer, "
    "and what to do next. The full results are at the end."
)
provenance()
st.markdown(
    f"Each time a customer opens the app, it shows three of 12 offers. The usual way to pick "
    f"them is a **{baseline_name}**: last month's most popular offers, the same for everyone. "
    f"We tested {len(table)} ways of choosing on the same {story.app_opens:,} simulated app "
    f"opens, against an oracle that knows what each customer would take up. Halfway through, "
    f"customers' tastes changed: the behaviour shift."
)

# --- 1. the hook -----------------------------------------------------------------------------
st.header("1 · The problem: a fixed ranking leaves take-ups on the table", divider="gray")
possible, fixed = story.oracle_per_1000, story.per_1000[BASELINE]
left, middle, right = st.columns(3)
left.metric(
    "What was possible",
    f"{possible:,.0f}",
    help="Take-ups per 1,000 app opens "
    "for an oracle that always shows the three offers each customer is likeliest "
    "to take up.",
)
middle.metric(
    f"The {baseline_name}",
    f"{fixed:,.0f}",
    f"{story.share_of_oracle[BASELINE]:.0%} of what was possible",
    delta_color="off",
    delta_arrow="off",
)
right.metric("Left on the table", f"{possible - fixed:,.0f}")
st.caption("Take-ups per 1,000 app opens, averaged over every run.")
st.markdown(
    f"For every 1,000 app opens, the {baseline_name} misses **{possible - fixed:,.0f}** "
    f"take-ups that were there to be had: it shows everyone the same offers, and it can't "
    f"learn from what happens next."
)

# --- 2. rising insights ----------------------------------------------------------------------
st.header("2 · What we found", divider="gray")

# 2a: the learner pulls ahead, and keeps pulling.
if story.overtakes_baseline_at is not None:
    st.subheader(
        f"It overtakes the fixed ranking after {story.overtakes_baseline_at:,} app opens, "
        "and the lead keeps growing"
    )
else:
    st.subheader("In this run it never overtook the fixed ranking for good")
curve = gap_curve(rounds, best)
figure = go.Figure()
figure.add_hline(y=0, line_color=MUTED, line_width=1)
figure.add_annotation(
    x=0,
    y=0,
    xanchor="left",
    yanchor="top",
    showarrow=False,
    text=f"the {baseline_name}",
    font={"color": MUTED},
)
figure.add_scatter(
    x=curve["round"],
    y=curve["extra"],
    mode="lines",
    name=best_name,
    line={"color": colour(best), "width": 3},
    hovertemplate="%{x:,} app opens: %{y:+,.0f} take-ups<extra></extra>",
)
if story.shift_round:
    figure.add_annotation(
        x=story.shift_round,
        y=story.extra_at_shift,
        ax=-60,
        ay=-40,
        text=f"{story.extra_at_shift:+,.0f} by the shift",
    )
figure.add_annotation(
    x=curve["round"].iloc[-1],
    y=story.extra_at_end,
    ax=-40,
    ay=-30,
    text=f"<b>{story.extra_at_end:+,.0f} by the end</b>",
)
figure.update_layout(
    xaxis_title="App opens",
    yaxis_title="Extra take-ups (cumulative)",
    showlegend=False,
    hovermode="closest",
)
st.plotly_chart(mark_shift(style(figure, height=380), story.shift_round), width="stretch")
growth = story.extra_at_end - story.extra_at_shift
st.caption(
    f"Take-ups won over the {baseline_name}, in expected take-ups (what each choice was worth "
    f"on average), on the first run. The lead grew faster after the shift "
    f"({growth:+,.0f} in the second half against {story.extra_at_shift:+,.0f} in the first)."
    if story.shift_round and growth > story.extra_at_shift
    else f"Take-ups won over the {baseline_name}, in expected take-ups, on the first run."
)

# 2b: personalisation is the lever.
ladder = [p for p in LADDER if p in story.share_of_oracle]
if best not in ladder:
    ladder.append(best)
segment_jump = (
    story.share_of_oracle["thompson_segment"] - story.share_of_oracle["thompson"]
    if {"thompson", "thompson_segment"} <= set(ladder)
    else None
)
st.subheader(
    "Learning per customer segment is the biggest single step"
    if segment_jump
    else "Each idea adds take-ups"
)
names = [label(p) for p in ladder]
focus = "thompson_segment" if segment_jump else best  # the bar this chart's message is about
figure = go.Figure(
    go.Bar(
        x=[story.share_of_oracle[p] for p in ladder],
        y=names,
        orientation="h",
        marker={"color": [colour(p) if p == focus else MUTED for p in ladder]},
        text=[f"{story.share_of_oracle[p]:.0%}" for p in ladder],
        textposition="outside",
        customdata=ladder,
        hovertemplate="%{y} (%{customdata}): %{x:.1%} of what was possible<extra></extra>",
    )
)
figure.add_vline(
    x=1,
    line_dash="dot",
    line_color=MUTED,
    annotation_text="what was possible",
    annotation_position="top",
)
if segment_jump:
    figure.add_annotation(
        x=story.share_of_oracle["thompson_segment"] - 0.01,
        y=label("thompson_segment"),
        xanchor="right",
        showarrow=False,
        text=f"<b>+{segment_jump * 100:.0f} points over one model for everyone</b>",
        font={"color": "white"},
    )
figure.update_layout(
    xaxis={"tickformat": ".0%", "range": [0, 1.12], "title": "Share of what was possible"},
    yaxis={"autorange": "reversed"},
    showlegend=False,
    hovermode="closest",
    bargap=0.35,
)
st.plotly_chart(style(figure, height=320), width="stretch")
st.caption(
    "Each bar adds one idea to the one above: learning instead of a fixed ranking, then one "
    "set of beliefs per customer segment instead of one for everyone, then forgetting old "
    "evidence. Students, pensioners and small businesses want different things, so a model "
    "that learns one answer for everyone can't serve any of them well."
)

# 2c: only the learners followed the change.
if story.shift_round:
    st.subheader("When customers changed, the learner followed; the fixed ranking didn't")
    figure = go.Figure()
    for policy in [p for p in ("random", BASELINE, best) if p in story.before_after]:
        before, after = story.before_after[policy]
        focus = policy in (best, BASELINE)
        line_colour = colour(policy) if focus else MUTED
        figure.add_scatter(
            x=["Before the shift", "After the shift"],
            y=[before * 1000, after * 1000],
            mode="lines+markers+text",
            name=label(policy),
            line={"color": line_colour, "width": 3 if focus else 1.5},
            marker={"size": 10, "color": line_colour},
            text=[f"{before * 1000:,.0f}", f"{after * 1000:,.0f}"],
            textposition=["middle left", "middle right"],
            hovertemplate=f"{label(policy)}: %{{y:,.0f}} per 1,000 app opens<extra></extra>",
        )
    figure.update_layout(
        yaxis_title="Take-ups per 1,000 app opens",
        hovermode="closest",
        xaxis={"range": [-0.4, 1.4]},
    )
    st.plotly_chart(style(figure, height=360), width="stretch")
    before, after = story.before_after[best]
    fixed_before, fixed_after = story.before_after[BASELINE]
    sentence = (
        f"When young professionals switched to streaming and families to electricity, the "
        f"learner's take-ups rose from {before * 1000:,.0f} to {after * 1000:,.0f} per 1,000 app "
        f"opens: it found the new favourites. The {baseline_name} went from "
        f"{fixed_before * 1000:,.0f} to {fixed_after * 1000:,.0f}, still showing last month's."
    )
    if story.adaptation and story.adaptation[0]:
        with_, without = story.adaptation
        sentence += (
            f" With forgetting, the young professionals' new favourite led within "
            f"{with_:,.0f} app opens; without it, "
            + (f"{without:,.0f}." if without else "not within the run.")
        )
    st.caption(sentence)

# --- 3. the aha moment -------------------------------------------------------------------------
st.header("3 · The answer", divider="gray")
with st.container(border=True):
    st.markdown(f"### {extra:,.0f} more take-ups for every 1,000 app opens")
    st.markdown(
        f"**{best_name}** won {story.per_1000[best]:,.0f} take-ups per 1,000 app opens, against "
        f"{fixed:,.0f} for the {baseline_name}: **{story.uplift:+.0%}**. It beat the "
        f"{baseline_name} in **{runs(story.best_beats_baseline)}** and random offers in "
        f"**{runs(story.best_beats_random)}**, reaching **{story.share_of_oracle[best]:.0%}** "
        f"of what was possible."
    )
    if story.tied_with_best:
        tied = ", ".join(label(p).lower() for p in story.tied_with_best)
        note = f"Its closest variants ({tied}) are a statistical tie on take-ups."
        if story.forgetting_after_shift and best == "thompson_segment_decay":
            won, saved = story.forgetting_after_shift
            note += (
                f" Forgetting gets the nod because it adapts: after the shift, it "
                f"lost {saved:,.0f} fewer take-ups than the same policy without forgetting "
                f"(fewer in {runs(won)})."
            )
        st.caption(note)

# --- 4. what to do next ------------------------------------------------------------------------
st.header("4 · What to do next", divider="gray")
active = settings().recommendation.active_policy
serve = (
    f"**Serve {best_name.lower()}.** It is already what the API serves (`{best}`)."
    if active == best
    else f"**Serve {best_name.lower()}:** set `active_policy: {best}` in "
    f"`config/settings.yaml` (the API serves `{active}` today)."
)
steps = [serve]
if story.revenue_trade:
    more_revenue, fewer = story.revenue_trade
    steps.append(
        f"**Decide what to optimise.** Learning from revenue instead of take-ups earns "
        f"{more_revenue:.0%} more revenue per app open, for {fewer:.0%} fewer take-ups. That is "
        f"a business choice; the system makes it one setting (`reward`)."
    )
steps.append(
    "**Prove it on real traffic.** Send a share of real app opens to the learner and keep a "
    f"holdout on the {baseline_name} (the API takes `?policy=` per request), then compare "
    "take-ups per app open. These results come from a simulation."
)
steps.append(
    "**Keep watching.** Alert when take-up per app open drops or a segment's favourite "
    "changes; the nightly roll-up (`daily_offer_stats`) has the numbers."
)
st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)))

# --- the detail, for analysts ------------------------------------------------------------------
st.divider()
with st.expander("Explore the full results: every policy, every measure, convergence"):
    st.dataframe(
        table.set_index("policy")[
            [
                "take_up_rate",
                "take_up_before_shift",
                "take_up_after_shift",
                "revenue_per_open",
                "regret",
                "regret_sd",
                "share_of_oracle",
                "regret_vs_random",
                "regret_vs_static",
            ]
        ],
        column_config={
            "take_up_rate": st.column_config.NumberColumn("Take-up per open", format="%.3f"),
            "take_up_before_shift": st.column_config.NumberColumn("Before shift", format="%.3f"),
            "take_up_after_shift": st.column_config.NumberColumn("After shift", format="%.3f"),
            "revenue_per_open": st.column_config.NumberColumn("Revenue per open", format="R%.2f"),
            "regret": st.column_config.NumberColumn("Regret", format="%.0f"),
            "regret_sd": st.column_config.NumberColumn("± sd", format="%.0f"),
            "share_of_oracle": st.column_config.NumberColumn("Share of oracle", format="percent"),
            "regret_vs_random": st.column_config.NumberColumn(
                "Regret cut vs random", format="percent"
            ),
            "regret_vs_static": st.column_config.NumberColumn(
                "Regret cut vs static", format="percent"
            ),
        },
        width="stretch",
    )
    st.caption(
        "Mean over seeds. Regret is the take-ups lost to the oracle: lower is better, and a "
        "flattening curve means learning."
    )
    available = list(table["policy"])
    chosen = st.multiselect(
        "Policies on the charts",
        available,
        default=[p for p in KEY_POLICIES if p in available],
    )
    tabs = st.tabs(["Regret", "Take-ups", "Revenue"])
    measures = [
        ("regret", "Take-ups lost to the oracle"),
        ("take_ups", "Take-ups"),
        ("revenue", "Revenue (R)"),
    ]
    curves = {
        policy: cumulative(log.reset_index(drop=True), every=250)
        for policy, log in rounds.groupby("policy")
        if policy in chosen
    }
    for tab, (column, axis_label) in zip(tabs, measures, strict=True):
        figure = go.Figure()
        for policy, policy_curve in curves.items():
            figure.add_scatter(
                x=policy_curve["round"],
                y=policy_curve[column],
                name=policy,
                mode="lines",
                line={"color": colour(policy), "width": 2},
            )
        figure.update_layout(xaxis_title="App opens", yaxis_title=f"{axis_label} (cumulative)")
        tab.plotly_chart(mark_shift(style(figure), story.shift_round), width="stretch")
    st.caption("Curves are from the first seed; the table averages every seed.")

    st.markdown("**Convergence**")
    st.caption(
        "App opens until each segment's best offer for the moment (including payday week) was "
        "ranked first in 80% of that segment's last 100 app opens: from the start, and again "
        "after the shift. Empty means it did not happen within the run."
    )
    shown = convergence[convergence["policy"].isin(chosen)]
    st.dataframe(
        shown.pivot_table(
            index=["phase", "segment"], columns="policy", values="app_opens", dropna=False
        )
        .reindex(columns=[p for p in chosen if p in set(shown["policy"])])
        .sort_index(ascending=[False, True]),
        column_config={p: st.column_config.NumberColumn(p, format="%.0f") for p in chosen},
        width="stretch",
    )
