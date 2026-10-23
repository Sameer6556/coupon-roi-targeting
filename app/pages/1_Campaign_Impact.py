"""Did the coupon booklets change spending? Matched difference-in-differences, interactively."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

ui.page("Campaign Impact", "📈")
ui.bundle_ready()
r = ui.results("causal")
pooled, waves = r["pooled"], r["waves"]

st.title("Did the booklets work?")
st.markdown(
    "Type A booklets went to households picked from their purchase history, so recipients and non-recipients were "
    "different before anything was mailed. Each recipient is matched to a look-alike non-recipient (same spend, "
    "trips, trend, coupon habits, earlier booklets and other campaigns), then the **change** in weekly spend is "
    "compared. Shared shocks like the holidays cancel out."
)

choice = st.radio("Wave", ["all waves"] + [f"wave {w['wave']}" for w in waves], horizontal=True)
if choice == "all waves":
    did, naive, base = pooled["did_matched"], pooled["naive_post"], pooled["baseline_weekly_spend"]
    n_pairs, n_t = pooled["n_pairs"], pooled["n_treated"]
else:
    w = waves[int(choice.split()[-1]) - 1]
    did, naive, base = w["did_matched"], w["naive_post"], w["baseline_weekly_spend"]
    n_pairs, n_t = w["n_pairs"], w["n_treated"]
    st.caption(
        f"Campaign {w['campaign_id']}: baseline weeks {w['pre_window'][0]}-{w['pre_window'][1]}, "
        f"campaign weeks {w['post_window'][0]}-{w['post_window'][1]}"
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Recipients", f"{n_t:,}")
c2.metric("Matched to a look-alike", f"{n_pairs:,}", f"{n_pairs / n_t:.0%} of recipients", delta_color="off")
c3.metric("Naive gap", f"+${naive['est']:.2f}/wk")
c4.metric(
    "Effect (matched DiD)",
    f"{did['est']:+.2f} $/wk",
    f"95% CI {did['lo']:+.2f} to {did['hi']:+.2f}",
    delta_color="off",
)
st.caption(
    f"Matched recipients spent ${base:.0f}/week at baseline, so the effect is {did['est'] / base:+.1%} of spend. "
    "Recipients without a close look-alike (mostly the very heaviest shoppers) are left out; that's the price of a fair comparison."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Event study", "Balance", "Robustness", "Who responds?", "Was it worth it?"]
)

with tab1:
    es = pd.DataFrame(r["event_study"])
    fig = go.Figure()
    fig.add_vrect(x0=-0.5, x1=es.rel_week.max() + 0.5, fillcolor=ui.BLUE, opacity=0.05, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=np.r_[es.rel_week, es.rel_week[::-1]],
            y=np.r_[es.hi, es.lo[::-1]],
            fill="toself",
            fillcolor="rgba(42,120,214,0.14)",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=es.rel_week,
            y=es.est,
            mode="lines+markers",
            line={"color": ui.BLUE, "width": 2},
            marker={"size": 8, "line": {"color": "white", "width": 2}},
            name="gap",
            customdata=np.c_[es.lo, es.hi],
            hovertemplate="week %{x}: %{y:+.2f} $/wk<br>95% CI %{customdata[0]:+.2f} to %{customdata[1]:+.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color=ui.INK_2, line_width=1)
    fig.update_layout(
        title="Weekly spend gap, recipient minus look-alike (all waves)",
        height=420,
        showlegend=False,
        xaxis_title="weeks relative to the start of the wave (shaded = booklet in hand)",
        yaxis_title="$ per week vs own baseline",
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "Before week 0 the two groups move together (flat line around zero), which is the check that "
        "parallel trends is believable. During the wave the gap is small and noisy."
    )

with tab2:
    wsel = waves if choice == "all waves" else [waves[int(choice.split()[-1]) - 1]]
    b = pd.concat([pd.DataFrame(w["balance"]).assign(wave=w["wave"]) for w in wsel])
    b = b.groupby("covariate", sort=False)[["before", "after"]].mean().reset_index()
    fig = go.Figure()
    fig.add_vrect(x0=-0.1, x1=0.1, fillcolor="#f0efec", opacity=1, line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=b.before,
            y=b.covariate,
            mode="markers",
            name="before matching",
            marker={"color": ui.NEUTRAL, "size": 10, "line": {"color": "white", "width": 2}},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=b.after,
            y=b.covariate,
            mode="markers",
            name="after matching",
            marker={"color": ui.BLUE, "size": 10, "line": {"color": "white", "width": 2}},
        )
    )
    fig.update_layout(
        title="Standardised mean difference (|SMD| < 0.1 = balanced)",
        height=460,
        yaxis={"autorange": "reversed"},
        xaxis_title="SMD",
    )
    st.plotly_chart(fig, width="stretch")

with tab3:
    rob = pd.DataFrame(r["robustness"])
    fig = go.Figure()
    for _, row in rob.iterrows():
        check = row.spec.startswith(("placebo", "unmatched"))
        color = ui.NEUTRAL if check else ui.BLUE
        fig.add_trace(
            go.Scatter(
                x=[row.lo, row.hi],
                y=[row.spec, row.spec],
                mode="lines",
                line={"color": color, "width": 3},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row.est],
                y=[row.spec],
                mode="markers",
                marker={"color": color, "size": 11, "line": {"color": "white", "width": 2}},
                showlegend=False,
                hovertemplate=f"{row.spec}<br>{row.est:+.2f} $/wk [{row.lo:+.2f}, {row.hi:+.2f}]<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_color=ui.INK_2, line_width=1)
    fig.update_layout(
        title="Same question, different reasonable choices (all waves, 95% CI)",
        height=480,
        yaxis={"autorange": "reversed"},
        xaxis_title="effect on weekly spend, $",
    )
    st.plotly_chart(fig, width="stretch")
    alt = rob[~rob.spec.str.startswith(("main", "placebo", "unmatched"))]
    plc = rob[rob.spec.str.startswith("placebo")].iloc[0]
    st.markdown(
        f"The {len(alt)} alternative specifications land between {alt.est.min():+.2f} and {alt.est.max():+.2f} $/week, "
        f"and the placebo (a fake start date inside the baseline) gives {plc.est:+.2f}. The effect is small and "
        "positive, but most intervals touch zero."
    )

with tab4:
    seg = pd.DataFrame(r["segments"])
    names = {"spend_tier": "spend tier", "booklet_seg": "booklet history", "coupon_seg": "coupon habit"}
    seg["label"] = [f"{names[d]}: {s}" for d, s in zip(seg.dimension, seg.segment, strict=True)]
    fig = go.Figure()
    for _, row in seg.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row.lo, row.hi],
                y=[row.label] * 2,
                mode="lines",
                line={"color": ui.BLUE, "width": 3},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row.est],
                y=[row.label],
                mode="markers",
                showlegend=False,
                marker={"color": ui.BLUE, "size": 11, "line": {"color": "white", "width": 2}},
                hovertemplate=f"{row.label}<br>{row.est:+.2f} $/wk, n={row.n_pairs}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_color=ui.INK_2, line_width=1)
    fig.update_layout(
        title="Effect by segment (segments defined BEFORE the baseline window)",
        height=420,
        yaxis={"autorange": "reversed"},
        xaxis_title="effect on weekly spend, $",
    )
    st.plotly_chart(fig, width="stretch")
    with st.expander("The regression-to-the-mean trap (a mistake worth showing)"):
        trap = pd.DataFrame(r["segments_regression_to_mean_trap"])
        st.markdown(
            "My first version cut segments on spend in the **same** baseline window used for the DiD. It 'found' "
            "that light shoppers respond strongly and heavy shoppers react negatively:"
        )
        st.dataframe(
            trap[["segment", "n_pairs", "est", "lo", "hi"]].round(2), hide_index=True, width="stretch"
        )
        st.markdown(
            "That's an artefact. A household that happened to have a quiet baseline lands in 'light' and then "
            "bounces back, which looks like a big positive effect. Cutting segments on spend from *before* the "
            "baseline removes the selection, and the differences disappear."
        )

with tab5:
    roi = r["roi"]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Break-even lift",
        f"${roi['breakeven_lift_per_week']:.2f}/wk",
        f"{roi['breakeven_rel']:.1%} of spend",
        delta_color="off",
    )
    c2.metric(
        "Simulated profit per booklet",
        ui.money(roi["simulated_profit_per_booklet"], 2),
        f"95% CI {ui.money(roi['simulated_profit_lo'], 2)} to {ui.money(roi['simulated_profit_hi'], 2)}",
        delta_color="off",
    )
    c3.metric("Chance a booklet paid off", f"{roi['prob_profitable']:.0%}")
    st.markdown(
        "**Plan the next test.** How big would a randomised holdout need to be to see an effect this size?"
    )
    pw = r["power"]
    a, b, c = st.columns(3)
    households = a.slider("Households in the wave", 500, 20000, 2200, step=100)
    holdout = b.slider("Share held out (no booklet)", 0.05, 0.5, 0.2, step=0.05)
    use_cuped = c.toggle("Use CUPED (baseline spend as covariate)", value=True)
    z = 1.959964 + 0.841621  # z for 5% two-sided significance + z for 80% power
    sigma = pw["sigma_weekly_spend"] * (np.sqrt(1 - pw["rho"] ** 2) if use_cuped else 1)
    mde = z * sigma * np.sqrt(1 / (households * (1 - holdout)) + 1 / (households * holdout))
    m1, m2 = st.columns(2)
    m1.metric(
        "Smallest detectable lift (80% power)",
        f"${mde:.2f}/wk",
        f"{mde / pw['baseline_weekly_spend']:.1%} of spend",
        delta_color="off",
    )
    m2.metric(
        "Effect we estimated",
        f"${pooled['did_matched']['est']:.2f}/wk",
        f"{pooled['rel_lift']:.1%} of spend",
        delta_color="off",
    )
    st.caption(
        f"CUPED cut the variance by {r['cuped']['variance_reduction_vs_post']:.0%} on the matched sample "
        f"(baseline-to-campaign correlation {pw['rho']:.2f})."
    )

ui.footer()
