"""Algorithm comparison: regret, take-ups and revenue for every policy on the same world."""

import plotly.graph_objects as go
import streamlit as st

from recommender.evaluation.metrics import cumulative
from shared import KEY_POLICIES, colour, mark_shift, need_results, results_file, style

st.title("Algorithm comparison")
table = results_file("comparison.csv")
rounds = results_file("comparison_rounds.parquet")
convergence = results_file("convergence.csv")
if not need_results(table, rounds, convergence):
    st.stop()

shift_round = int(rounds.loc[rounds["shifted"], "round"].min()) if rounds["shifted"].any() else None
st.caption(
    f"Every policy served the same {rounds['round'].max() + 1:,} simulated app opens and was "
    "scored against an oracle that knows the true take-up probabilities. Regret is the "
    "take-ups lost to the oracle: lower is better, and a flattening curve means learning."
)

st.subheader("Results (mean over seeds)")
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
        "regret_vs_random": st.column_config.NumberColumn("Regret cut vs random", format="percent"),
        "regret_vs_static": st.column_config.NumberColumn("Regret cut vs static", format="percent"),
    },
    use_container_width=True,
)

available = list(table["policy"])
chosen = st.multiselect(
    "Policies on the charts",
    available,
    default=[p for p in KEY_POLICIES if p in available],
)
tabs = st.tabs(["Regret", "Take-ups", "Revenue"])
measures = [("regret", "Take-ups lost to the oracle"), ("take_ups", "Take-ups"),
            ("revenue", "Revenue (R)")]  # fmt: skip
curves = {
    policy: cumulative(log.reset_index(drop=True), every=250)
    for policy, log in rounds.groupby("policy")
    if policy in chosen
}
for tab, (column, label) in zip(tabs, measures, strict=True):
    figure = go.Figure()
    for policy, curve in curves.items():
        figure.add_scatter(
            x=curve["round"],
            y=curve[column],
            name=policy,
            mode="lines",
            line={"color": colour(policy), "width": 2},
        )
    figure.update_layout(xaxis_title="App opens", yaxis_title=f"{label} (cumulative)")
    tab.plotly_chart(mark_shift(style(figure), shift_round), use_container_width=True)
st.caption("Curves are from the first seed; the table averages every seed.")

st.subheader("Convergence")
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
    use_container_width=True,
)
