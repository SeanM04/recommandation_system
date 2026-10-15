"""Explore vs exploit: each offer's Beta belief for one segment, and how it changes."""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from shared import mark_shift, need_results, provenance, results_file, style

st.title("Explore vs exploit")
st.caption(
    "Thompson sampling keeps a Beta belief about each offer's take-up rate and ranks by one "
    "random draw from each. A wide curve is an offer the policy is unsure about: its draws "
    "sometimes land high, so it still gets shown (exploration). A narrow curve far to the "
    "right is an offer it is confident in (exploitation)."
)

snapshots = results_file("comparison_snapshots.parquet")
rounds = results_file("comparison_rounds.parquet")
if not need_results(snapshots, rounds):
    st.stop()
provenance()

HIGHLIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # top four offers; the rest grey
OTHER = "#9aa5ad"

thompson = sorted(p for p in snapshots["policy"].unique() if p.startswith("thompson"))
left, middle, right = st.columns([2, 2, 3])
policy = left.selectbox(
    "Policy", thompson, index=thompson.index("thompson_segment_decay")
    if "thompson_segment_decay" in thompson else 0
)  # fmt: skip
keys = sorted(snapshots.loc[snapshots["policy"] == policy, "context_key"].unique())
key = middle.selectbox("Context", keys, index=keys.index("family") if "family" in keys else 0)
available = sorted(snapshots.loc[snapshots["policy"] == policy, "round"].unique())
at = right.select_slider("After this many app opens", options=available, value=available[-1])

beliefs = snapshots[(snapshots["policy"] == policy) & (snapshots["context_key"] == key)]
now = beliefs[beliefs["round"] == at].copy()
now["mean"] = now["alpha"] / (now["alpha"] + now["beta"])
now = now.sort_values("mean", ascending=False)
palette = {offer: HIGHLIGHT[i] if i < 4 else OTHER for i, offer in enumerate(now["offer_id"])}


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
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
        line={"color": palette[row.offer_id], "width": 2 if palette[row.offer_id] != OTHER else 1},
    )
figure.update_layout(xaxis_title="Take-up rate", yaxis_title="Belief (density)",
                     xaxis_tickformat=".0%", hovermode="closest")  # fmt: skip
st.plotly_chart(style(figure, 380), use_container_width=True)

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
    use_container_width=True,
)

st.subheader("How the beliefs moved")
history = beliefs.assign(mean=beliefs["alpha"] / (beliefs["alpha"] + beliefs["beta"]))
trend = go.Figure()
for offer, series in history.groupby("offer_id"):
    trend.add_scatter(
        x=series["round"],
        y=series["mean"],
        name=offer,
        mode="lines+markers",
        line={"color": palette.get(offer, OTHER), "width": 2 if palette.get(offer) != OTHER else 1},
    )
trend.update_layout(xaxis_title="App opens", yaxis_title="Mean take-up", yaxis_tickformat=".0%")
shift = int(rounds.loc[rounds["shifted"], "round"].min()) if rounds["shifted"].any() else None
st.plotly_chart(mark_shift(style(trend, 360), shift), use_container_width=True)
if policy.endswith("decay") or policy.endswith("revenue"):
    st.caption(
        "With forgetting, old evidence fades, so beliefs keep a little width and can move "
        "after the shift."
    )
else:
    st.caption(
        "Without forgetting, evidence only accumulates: beliefs narrow and move slowly "
        "after the shift."
    )
