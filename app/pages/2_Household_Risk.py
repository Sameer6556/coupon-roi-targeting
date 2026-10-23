"""Look up any household: lapse risk, why, expected response to a booklet, and its spending history."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ui

ui.page("Household Risk", "🧭")
ui.bundle_ready()
churn = ui.results("churn")
scores = ui.query("SELECT * FROM household_scores")
reg = scores[scores.is_regular.fillna(False).astype(bool)].copy()

st.title("Which regular shoppers are about to go quiet?")
st.markdown(
    f"Scored as of the last week of data. **Regular** = shopped in at least 8 of the last 16 weeks "
    f"({len(reg):,} households). **Going quiet** = shopping in 2 or fewer of the next 8 weeks, which happens to about "
    f"{churn['base_rate']:.0%} of regulars in any given 8 weeks."
)

c1, c2, c3 = st.columns(3)
top_decile = reg[reg.risk_decile == 10]
c1.metric("Regular shoppers scored", f"{len(reg):,}")
c2.metric(
    "Average risk, top decile",
    f"{top_decile.lapse_prob.mean():.0%}",
    f"vs {reg.lapse_prob.mean():.0%} overall",
    delta_color="off",
)
c3.metric("Spend at stake in the top decile", ui.money(top_decile.spend_16w.sum()) + "/wk")

left, right = st.columns([1, 1.25])
with left:
    fig = go.Figure(
        go.Histogram(
            x=reg.lapse_prob,
            nbinsx=30,
            marker_color=ui.BLUE,
            hovertemplate="risk %{x}<br>%{y} households<extra></extra>",
        )
    )
    fig.update_layout(
        title="Distribution of lapse risk",
        height=330,
        bargap=0.08,
        xaxis={"title": "probability of going quiet", "tickformat": ".0%"},
        yaxis_title="households",
    )
    st.plotly_chart(fig, width="stretch")
with right:
    n = st.slider("Show the riskiest", 10, 200, 25, step=5)
    view = reg.sort_values("lapse_prob", ascending=False).head(n)[
        [
            "household_id",
            "lapse_prob",
            "spend_16w",
            "trips_4w",
            "recency_weeks",
            "momentum_4v16",
            "uplift",
            "expected_profit",
        ]
    ]
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=330,
        column_config={
            "household_id": st.column_config.NumberColumn("household", format="%d"),
            "lapse_prob": st.column_config.ProgressColumn("risk", format="%.0f%%", min_value=0, max_value=1),
            "spend_16w": st.column_config.NumberColumn("$/wk (16w)", format="$%.0f"),
            "trips_4w": st.column_config.NumberColumn("trips (4w)", format="%d"),
            "recency_weeks": st.column_config.NumberColumn("weeks since trip", format="%d"),
            "momentum_4v16": st.column_config.NumberColumn("4w vs 16w spend", format="%.2f"),
            "uplift": st.column_config.NumberColumn("booklet uplift $/wk", format="%+.2f"),
            "expected_profit": st.column_config.NumberColumn("booklet profit (sim.)", format="$%.2f"),
        },
    )

st.divider()
st.subheader("Household detail")
default = int(reg.sort_values("lapse_prob", ascending=False).household_id.iloc[0])
hid = st.number_input(
    "Household id",
    min_value=int(scores.household_id.min()),
    max_value=int(scores.household_id.max()),
    value=default,
    step=1,
)
row = scores[scores.household_id == hid]
if row.empty:
    st.info("This household didn't shop in the last 8 weeks of data, so it isn't scored.")
else:
    row = row.iloc[0]
    hist = ui.query(
        "SELECT week, week_start, spend_ex_fuel, trips FROM household_week WHERE household_id = ? ORDER BY week",
        (hid,),
    )
    camps = ui.query(
        """SELECT c.campaign_id, c.campaign_type, greatest(c.start_week, 1) AS s, least(c.end_week, 53) AS e
           FROM campaign_exposure x JOIN campaigns c USING (campaign_id) WHERE x.household_id = ?""",
        (hid,),
    )
    a, b = st.columns([1.6, 1])
    with a:
        fig = go.Figure()
        for c in camps.itertuples():
            color = ui.BLUE if c.campaign_type == "Type A" else ui.NEUTRAL
            fig.add_vrect(x0=c.s - 0.5, x1=c.e + 0.5, fillcolor=color, opacity=0.08, line_width=0)
        fig.add_bar(
            x=hist.week,
            y=hist.spend_ex_fuel,
            marker_color=ui.BLUE,
            name="spend",
            customdata=hist.trips,
            hovertemplate="week %{x}: $%{y:.2f}, %{customdata} trips<extra></extra>",
        )
        fig.update_layout(
            title=f"Household {hid}: weekly spend (shaded = campaign it received; blue = Type A booklet)",
            height=330,
            showlegend=False,
            xaxis_title="week",
            yaxis_title="$ ex fuel",
            bargap=0.15,
        )
        st.plotly_chart(fig, width="stretch")
    with b:
        if bool(row.is_regular):
            st.metric(
                "Risk of going quiet (next 8 weeks)",
                f"{row.lapse_prob:.0%}",
                f"decile {int(row.risk_decile)} of 10",
                delta_color="off",
            )
            drivers = json.loads(row.top_drivers) if isinstance(row.top_drivers, str) else []
            if drivers:
                st.markdown("**Why** (TreeSHAP, red pushes risk up):")
                d = pd.DataFrame(drivers)
                fig = go.Figure(
                    go.Bar(
                        x=d.effect,
                        y=d.feature,
                        orientation="h",
                        marker_color=[ui.RED if v > 0 else ui.BLUE for v in d.effect],
                        customdata=d.value,
                        hovertemplate="%{y} = %{customdata}<br>effect %{x:+.2f}<extra></extra>",
                    )
                )
                fig.update_layout(
                    height=200,
                    margin={"t": 10, "l": 10, "r": 10, "b": 10},
                    yaxis={"autorange": "reversed"},
                    xaxis_title="log-odds contribution",
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.metric("Risk of going quiet", "not scored", "not a regular shopper", delta_color="off")
        st.metric(
            "Expected response to a booklet",
            f"{row.uplift:+.2f} $/wk",
            f"{'mail' if row.expected_profit > 0 else 'skip'} (simulated profit {ui.money(row.expected_profit, 2)})",
            delta_color="off",
        )

ui.footer()
