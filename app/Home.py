"""Coupon ROI & Targeting Engine: overview page."""

import plotly.graph_objects as go
import streamlit as st
import ui

ui.page("Overview")
ui.bundle_ready()

causal, churn, uplift, policy = (
    ui.results("causal"),
    ui.results("churn"),
    ui.results("uplift"),
    ui.results("policy"),
)
pooled = causal["pooled"]
roi = causal["roi"]
lgbm = next(r for r in churn["summary"] if r["model"] == "LightGBM")
oot = {r["learner"]: r for r in uplift["out_of_time"]}[uplift["chosen_learner"]]

ratios = [w["recipients"] / w["non-recipients"] for w in ui.results("eda")["targeting_bias"]["pre_spend"]]
alt = [r for r in causal["robustness"] if not r["spec"].startswith(("main", "placebo", "unmatched"))]
placebo = causal["placebo"]["pooled"]
hold30 = policy["holdout_profit_per_1000_at_30pct"]
per_booklet_top30 = hold30["uplift model"] / 300  # 30% of 1,000 households get a booklet
per_booklet_all = policy["holdout_profit_per_1000_all"]["uplift model"] / 1000
power = causal["power"]

st.title("Coupon ROI & Targeting Engine")
st.markdown(
    "A grocery retailer mails personalised coupon booklets to about 1,100 households at a time. "
    "**Did they work, who is about to drift away, and who should get the next one?** "
    "Built on a year of real transactions from 2,469 households."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Naive 'lift' from booklets",
    f"+${pooled['naive_post']['est']:.0f}/wk",
    help="Recipients minus non-recipients during the waves. Mostly targeting bias.",
)
c2.metric(
    "Actual effect (matched DiD)",
    f"{pooled['did_matched']['est']:+.2f} $/wk",
    f"{pooled['rel_lift']:+.1%} (95% CI {pooled['rel_lift_lo']:+.1%} to {pooled['rel_lift_hi']:+.1%})",
    delta_color="off",
)
c3.metric(
    "Lapse model, out of time",
    f"AUC {lgbm['roc_auc']:.2f}",
    f"top 10% catches {lgbm['recall_at_k']:.0%} of lapsers",
    delta_color="off",
)
c4.metric(
    "Uplift model, top 30% realised",
    f"+${oot['top30_effect']:.2f}/wk",
    f"vs +${oot['all_effect']:.2f} average",
    delta_color="off",
)

left, right = st.columns([1.15, 1])
with left:
    st.subheader("What I found")
    st.markdown(
        f"""
1. **Most of the apparent lift is targeting bias.** Before any booklet arrived, recipients already spent
   {min(ratios):.1f}x to {max(ratios):.1f}x more per week than non-recipients. Matching each recipient to a look-alike
   and comparing *changes* in spend shrinks the effect from ${pooled["naive_post"]["est"]:.0f}/week to
   **{pooled["did_matched"]["est"]:+.2f} $/week** ({pooled["n_pairs"]:,} matched pairs, 4 waves). Pre-trends are flat,
   a fake start date finds nothing ({placebo["est"]:+.2f} $/week), and {len(alt)} alternative specifications land
   between {min(r["est"] for r in alt):+.2f} and {max(r["est"] for r in alt):+.2f} $/week.
2. **Past waves roughly broke even** (simulated: {roi["assumptions"]["gross_margin"]:.0%} margin,
   ${roi["assumptions"]["cost_per_booklet"]:.2f} per booklet). Break-even is
   ${roi["breakeven_lift_per_week"]:.2f}/week of extra spend, and the 95% interval on profit runs from
   {ui.money(roi["simulated_profit_lo"], 2)} to {ui.money(roi["simulated_profit_hi"], 2)} per booklet.
3. **Who you send to may matter more than whether you send.** On two waves the uplift model never saw, mailing
   everyone lost about {ui.money(-per_booklet_all, 2)} per booklet, while its top 30% responded with
   +${oot["top30_effect"]:.2f}/week (about {ui.money(per_booklet_top30, 2)} profit per booklet, simulated).
   Promising, but the interval on that gain still touches zero.
4. **Next step:** mail the next wave by uplift with a randomised holdout. Even with CUPED, detecting an effect this
   small needs about {power["n_per_arm_cuped"]:,.0f} households per arm, so pool holdouts across several waves.
"""
    )
with right:
    waves = causal["waves"]
    labels = [f"wave {w['wave']}" for w in waves] + ["all"]
    naive = [w["naive_post"]["est"] for w in waves] + [pooled["naive_post"]["est"]]
    did = [w["did_matched"]["est"] for w in waves] + [pooled["did_matched"]["est"]]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=naive,
        name="naive gap",
        marker_color=ui.NEUTRAL,
        hovertemplate="%{x}: $%{y:.2f}/wk<extra>naive</extra>",
    )
    fig.add_bar(
        x=labels,
        y=did,
        name="matched DiD",
        marker_color=ui.BLUE,
        hovertemplate="%{x}: $%{y:.2f}/wk<extra>matched DiD</extra>",
    )
    fig.update_layout(
        title="Naive gap vs real effect ($ per household per week)",
        barmode="group",
        height=380,
        bargap=0.35,
        bargroupgap=0.08,
    )
    st.plotly_chart(fig, width="stretch")

st.subheader("How it's built")
st.graphviz_chart(
    """digraph {
    rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#f0efec" color="#c3c2b7" fontname="Helvetica" fontsize=11];
    edge [color="#898781"];
    raw [label="8 raw tables\\n1.47M transactions\\n20.9M promo rows"];
    sql [label="DuckDB star schema\\n+ household-week panel\\n(9 data checks)"];
    causal [label="Matched DiD\\nevent study, placebo\\nCUPED, power"];
    churn [label="Lapse model\\nrolling backtest\\nSHAP, calibration"];
    uplift [label="Uplift learners\\nbenchmark + out-of-time\\nshrinkage"];
    policy [label="Targeting policy\\nbudget, profit curves"];
    serve [label="Streamlit app\\nFastAPI + Docker\\nask-the-data"];
    raw -> sql -> causal -> uplift -> policy -> serve; sql -> churn -> policy;
    }""",
    width="stretch",
)
st.markdown(
    "Use the pages on the left: **Campaign Impact** for the causal analysis, **Household Risk** to look up any "
    "household, **Targeting Simulator** to change the budget and assumptions, and **Ask the Data** to query the "
    "tables in plain English."
)
ui.footer()
