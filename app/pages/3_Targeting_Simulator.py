"""Change the assumptions and the budget, see who should get the next booklet."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

ui.page("Targeting Simulator", "🎯")
ui.bundle_ready()
meta = ui.results("uplift_meta")
causal = ui.results("causal")
s = ui.query("SELECT household_id, uplift_raw, lapse_prob, spend_16w FROM household_scores")
n = len(s)

st.title("Who should get the next booklet?")
st.markdown(
    "Expected profit of mailing a household = **expected extra spend per week x weeks x margin - cost of the booklet**. "
    "Mail the households with positive expected profit, best first. Margins and costs aren't in the data, so set your own. "
    "All money here is simulated."
)

with st.sidebar:
    st.header("Assumptions")
    margin = st.slider("Gross margin", 0.15, 0.40, 0.28, 0.01, format="%.2f")
    cost = st.slider("Cost per booklet ($)", 0.25, 5.0, 1.50, 0.25)
    weeks = st.slider("Campaign length (weeks)", 4, 10, int(round(causal["pooled"]["post_weeks_avg"])))
    st.header("How much to trust the uplift model")
    lam = st.slider(
        "Weight on household-level predictions",
        0.0,
        1.0,
        float(meta.get("shrinkage", 0.5)),
        0.05,
        help="0 = every household gets the average effect (the model has no say). 1 = use raw predictions. "
        "The default is the slope of real outcomes on predictions in waves the model never saw.",
    )
    budget = st.slider("Booklets to send", 0, n, int(0.45 * n), step=25)

ate = meta["ate"]
s["uplift"] = ate + lam * (s.uplift_raw - meta["center"])
s["profit"] = s.uplift * weeks * margin - cost
rng = np.random.default_rng(42)
rankers = {
    "uplift model": s.uplift,
    "heaviest shoppers (current rule)": s.spend_16w,
    "highest lapse risk": s.lapse_prob.fillna(-1),
    "random": pd.Series(rng.random(n), index=s.index),
}
colors = {
    "uplift model": ui.BLUE,
    "heaviest shoppers (current rule)": ui.ORANGE,
    "highest lapse risk": ui.AQUA,
    "random": ui.NEUTRAL,
}
curves = {}
for name, score in rankers.items():
    order = np.argsort(-score.to_numpy(), kind="stable")
    curves[name] = (order, np.cumsum(s.profit.to_numpy()[order]))

best_k = int((s.profit > 0).sum())
at_budget = {name: (c[budget - 1] if budget else 0.0) for name, (_, c) in curves.items()}

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Break-even response",
    f"${cost / (weeks * margin):.2f}/wk",
    "extra spend needed per household",
    delta_color="off",
)
c2.metric("Best list size", f"{best_k:,}", f"{best_k / n:.0%} of active households", delta_color="off")
c3.metric("Profit at your budget, uplift ranking", ui.money(at_budget["uplift model"]))
c4.metric(
    "Same budget, current rule",
    ui.money(at_budget["heaviest shoppers (current rule)"]),
    ui.money(at_budget["heaviest shoppers (current rule)"] - at_budget["uplift model"]) + " vs uplift",
    delta_color="off",
)

fig = go.Figure()
x = np.arange(1, n + 1)
for name, (_, cum) in curves.items():
    fig.add_trace(
        go.Scatter(
            x=x,
            y=cum,
            name=name,
            line={"color": colors[name], "width": 3 if name == "uplift model" else 2},
            hovertemplate=f"{name}<br>%{{x}} booklets: $%{{y:,.0f}}<extra></extra>",
        )
    )
fig.add_hline(y=0, line_color=ui.INK_2, line_width=1)
fig.add_vline(
    x=budget, line_color=ui.MUTED, line_width=1, annotation_text="your budget", annotation_position="top"
)
if best_k:
    fig.add_trace(
        go.Scatter(
            x=[best_k],
            y=[curves["uplift model"][1][best_k - 1]],
            mode="markers",
            name="best list size",
            marker={"color": ui.BLUE, "size": 12, "line": {"color": "white", "width": 2}},
        )
    )
fig.update_layout(
    title="Cumulative expected profit by number of booklets sent (simulated)",
    height=460,
    xaxis_title="booklets sent, best-ranked household first",
    yaxis_title="expected profit ($)",
)
st.plotly_chart(fig, width="stretch")

st.caption(
    "Every ranking is scored with the same household-level estimates, so the uplift ranking wins here by construction. "
    "The honest test is out of time: on waves the model never saw, its top 30% responded with "
    f"+${[r for r in ui.results('uplift')['out_of_time'] if r['learner'] == meta['learner']][0]['top30_effect']:.2f}/wk "
    "vs the average, with an interval that still touches zero. Slide the trust weight to 0 to see the decision "
    "if you don't believe the model at all."
)

policy = st.selectbox("Download a mailing list ranked by", list(rankers), index=0)
order, _ = curves[policy]
pick = s.iloc[order[:budget]][["household_id", "uplift", "profit", "lapse_prob", "spend_16w"]]
st.download_button(
    f"Download {len(pick):,} households (CSV)",
    pick.rename(columns={"uplift": "expected_uplift_per_week", "profit": "expected_profit_simulated"}).to_csv(
        index=False
    ),
    file_name=f"next_wave_{policy.split()[0]}_{budget}.csv",
    mime="text/csv",
)
ui.footer()
