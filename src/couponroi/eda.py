"""Six business questions, each answered with one SQL query, one chart and one sentence.

The queries live in sql/analysis/ so they can be read (and reused) without Python.
"""

from __future__ import annotations

import json
import logging

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from . import plotting as P
from .warehouse import connect

log = logging.getLogger(__name__)
ANALYSIS_SQL = config.SQL_DIR / "analysis"


def q(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute((ANALYSIS_SQL / f"{name}.sql").read_text()).df()


def growth_segments(con) -> dict:
    seg = q(con, "growth_segments")
    con.register("segments_df", seg)
    con.execute("CREATE OR REPLACE TEMP TABLE segments AS SELECT * FROM segments_df")
    shift = q(con, "department_shift")
    counts = seg["segment"].value_counts()
    order = ["growing", "stable", "declining", "new"]
    counts = counts.reindex(order).fillna(0).astype(int)

    dec = shift[shift.segment == "declining"].sort_values("change_weekly_per_hh").head(8)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    colors = [P.POS, P.NEUTRAL, P.NEG, P.MUTED]
    a1.bar(counts.index, counts.values, color=colors, width=0.55)
    for i, v in enumerate(counts.values):
        a1.text(i, v + 15, f"{v:,}", ha="center", va="bottom", fontsize=9, color=P.INK)
    a1.set_title("Households by spend trend")
    P.subtitle(a1, "avg weekly spend, weeks 27-51 vs 2-26 (±25%)")
    a1.set_ylim(0, counts.max() * 1.15)

    y = np.arange(len(dec))
    a2.barh(y, dec["change_weekly_per_hh"], color=P.NEG, height=0.5)
    a2.set_yticks(y, [P.dept_label(d) for d in dec["department"]])
    a2.invert_yaxis()
    a2.axvline(0, color=P.AXIS, lw=0.8)
    a2.grid(axis="x")
    a2.grid(axis="y", visible=False)
    a2.set_title("Where declining households cut back")
    P.subtitle(a2, "change in weekly spend per household, $")
    for yi, v in zip(y, dec["change_weekly_per_hh"], strict=True):
        a2.text(v - 0.15, yi, f"{v:,.2f}", ha="right", va="center", fontsize=8.5, color=P.INK_2)
    a2.set_xlim(dec["change_weekly_per_hh"].min() * 1.25, 0.5)
    fig.tight_layout()
    P.save(fig, "eda_1_growth_segments")

    top_cut = dec.iloc[0]
    return {
        "counts": counts.to_dict(),
        "declining_top_departments": dec[["department", "change_weekly_per_hh"]].round(2).to_dict("records"),
        "sentence": (
            f"{counts['growing']:,} households grew their weekly spend by 25%+ in the second half of the year and "
            f"{counts['declining']:,} cut it by 25%+; decliners pulled back most in {top_cut['department'].title()} "
            f"({top_cut['change_weekly_per_hh']:+.2f} $/week per household)."
        ),
    }


def cohort_retention(con) -> dict:
    df = q(con, "cohort_retention")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = {0: "first shopped wk 1-4", 1: "wk 5-8", 2: "wk 9-12", 3: "wk 13-16"}
    sizes = {}
    for i, (cb, label) in enumerate(labels.items()):
        d = df[df.cohort_block == cb].sort_values("blocks_since_first")
        if d.empty:
            continue
        n = int(d.loc[d.blocks_since_first == 0, "households"].iloc[0])
        sizes[label] = n
        ax.plot(d.blocks_since_first, d.retention, color=P.SERIES[i], label=f"{label} (n={n:,})")
        ax.plot(
            d.blocks_since_first.iloc[-1], d.retention.iloc[-1], "o", color=P.SERIES[i], mec=P.SURFACE, mew=2
        )
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("4-week blocks since first purchase")
    ax.set_title("Share of each cohort still shopping")
    P.subtitle(ax, "a household counts as retained if it shopped at least once in the block")
    ax.legend(loc="lower left")
    P.save(fig, "eda_2_cohort_retention")
    main = df[df.cohort_block == 0].sort_values("blocks_since_first")
    month6 = float(main.loc[main.blocks_since_first == 6, "retention"].iloc[0])
    last = float(main.retention.iloc[-1])
    return {
        "cohort_sizes": sizes,
        "main_cohort_retention_block6": round(month6, 3),
        "main_cohort_retention_last": round(last, 3),
        "sentence": (
            f"Of households that first shopped in weeks 1-4, {month6:.0%} still shopped six 4-week blocks later "
            f"and {last:.0%} in the final block, so outright churn is rare in this panel; the real risk is "
            "households quietly spending less."
        ),
    }


def revenue_concentration(con) -> dict:
    df = q(con, "revenue_concentration")
    top20 = float(df.loc[df.household_share <= 0.2, "cum_spend_share"].max())
    top10 = float(df.loc[df.household_share <= 0.1, "cum_spend_share"].max())
    deciles = df.groupby("decile")["spend"].agg(["sum", "median"])
    x = np.concatenate([[0], df.household_share.to_numpy()])
    y = np.concatenate([[0], df.cum_spend_share.to_numpy()])
    # area under the (descending) cumulative curve; Gini = 2 * area - 1
    area = float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))
    gini = 2 * area - 1

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([0, 1], [0, 1], color=P.NEUTRAL, lw=1, label="if every household spent the same")
    ax.plot(x, y, color=P.BLUE, label="actual")
    ax.fill_between(x, x, y, color=P.BLUE, alpha=0.08, lw=0)
    ax.plot([0.2], [top20], "o", color=P.BLUE, mec=P.SURFACE, mew=2, ms=8)
    ax.annotate(
        f"top 20% of households\n= {top20:.0%} of spend",
        (0.2, top20),
        xytext=(0.32, top20 - 0.18),
        fontsize=9,
        color=P.INK,
        arrowprops={"arrowstyle": "-", "color": P.MUTED, "lw": 0.8},
    )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("households, ranked by spend")
    ax.set_ylabel("cumulative share of spend")
    ax.set_title("Revenue is concentrated")
    P.subtitle(ax, "cumulative spend (ex fuel) by household rank")
    ax.legend(loc="lower right")
    ax.grid(axis="x")
    P.save(fig, "eda_3_revenue_concentration")
    return {
        "top10_share": round(top10, 3),
        "top20_share": round(top20, 3),
        "gini": round(gini, 3),
        "decile_median_spend": deciles["median"].round(0).to_dict(),
        "sentence": f"The top 20% of households bring in {top20:.0%} of spend (top 10%: {top10:.0%}).",
    }


def targeting_bias(con) -> dict:
    df = q(con, "targeting_bias")
    piv = df.pivot(index="campaign_id", columns="recipient", values="avg_pre_weekly_spend")
    piv.columns = ["non-recipients", "recipients"]
    order = con.execute(
        "SELECT campaign_id, start_week FROM dim_campaign WHERE campaign_type='Type A' ORDER BY start_week"
    ).df()
    piv = piv.reindex([c for c in order.campaign_id if c in piv.index])
    labels = [f"wave {i + 1}\n(campaign {c})" for i, c in enumerate(piv.index)]
    ratio = piv["recipients"] / piv["non-recipients"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(piv))
    w = 0.32
    ax.bar(x - w / 2 - 0.01, piv["recipients"], width=w, color=P.BLUE, label="got the booklet")
    ax.bar(x + w / 2 + 0.01, piv["non-recipients"], width=w, color=P.NEUTRAL, label="did not")
    for xi, (r, n, k) in enumerate(zip(piv["recipients"], piv["non-recipients"], ratio, strict=True)):
        ax.text(xi - w / 2, r + 1, f"${r:.0f}", ha="center", fontsize=8.5, color=P.INK)
        ax.text(xi + w / 2, n + 1, f"${n:.0f}", ha="center", fontsize=8.5, color=P.INK_2)
        ax.text(
            xi, max(r, n) + 9, f"{k:.1f}x more", ha="center", fontsize=9.5, color=P.INK, fontweight="semibold"
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, piv.values.max() * 1.35)
    ax.set_ylabel("avg weekly spend before the wave ($)")
    ax.set_title("Booklets go to people who already spend more")
    P.subtitle(ax, "weekly spend in the 8 weeks BEFORE each Type A wave started")
    ax.legend(loc="upper left", ncols=2)
    P.save(fig, "eda_4_targeting_bias")
    return {
        "pre_spend": piv.round(2).reset_index().to_dict("records"),
        "ratio_mean": round(float(ratio.mean()), 2),
        "sentence": (
            f"Before any booklet arrived, recipients already spent {ratio.min():.1f}-{ratio.max():.1f}x more per week "
            "than non-recipients, so any naive 'recipients spend more' comparison is mostly targeting bias."
        ),
    }


def redemption(con) -> dict:
    df = q(con, "redemption_by_type")
    conc = (
        con.execute(
            """WITH r AS (SELECT household_id, count(*) AS n FROM fct_redemptions GROUP BY 1),
           ranked AS (SELECT n, row_number() OVER (ORDER BY n DESC) AS rk, count(*) OVER () AS hh,
                      sum(n) OVER () AS total FROM r)
           SELECT max(hh) AS redeeming_households, max(total) AS redemptions,
                  sum(n) FILTER (WHERE rk <= ceil(hh * 0.1)) / max(total) AS top10_share FROM ranked"""
        )
        .df()
        .iloc[0]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.bar(df.campaign_type, df.household_redemption_rate, color=P.BLUE, width=0.5)
    for i, (r, n) in enumerate(zip(df.household_redemption_rate, df.households_targeted, strict=True)):
        ax.text(i, r + 0.004, f"{r:.1%}", ha="center", fontsize=9, color=P.INK)
        ax.text(i, 0.004, f"{int(n):,} sent", ha="center", fontsize=8, color=P.SURFACE)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, df.household_redemption_rate.max() * 1.3)
    ax.set_title("Few households ever redeem")
    P.subtitle(ax, "share of targeted households that redeemed at least one coupon")
    P.save(fig, "eda_5_redemption")
    a = df.set_index("campaign_type")
    return {
        "by_type": df.round(4).to_dict("records"),
        "redeeming_households": int(conc.redeeming_households),
        "top10_redeemers_share": round(float(conc.top10_share), 3),
        "sentence": (
            f"Only {a.loc['Type A', 'household_redemption_rate']:.0%} of Type A recipients redeemed anything "
            f"(Type B: {a.loc['Type B', 'household_redemption_rate']:.0%}), and the top 10% of redeemers "
            f"made {conc.top10_share:.0%} of all redemptions."
        ),
    }


def demographics_coverage(con) -> dict:
    df = q(con, "demographics_coverage").set_index("has_demographics")
    metrics = [
        ("median_weekly_spend", "median weekly spend ($)", "{:.0f}"),
        ("avg_active_weeks", "avg weeks with a purchase", "{:.0f}"),
        ("avg_campaigns_received", "avg campaigns received", "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (col, title, fmt) in zip(axes, metrics, strict=True):
        vals = [df.loc[True, col], df.loc[False, col]]
        ax.bar(["with demo-\ngraphics", "without"], vals, color=[P.BLUE, P.NEUTRAL], width=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, fmt.format(v), ha="center", va="bottom", fontsize=9, color=P.INK)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle(
        "The 801 households with demographics are not typical",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    P.save(fig, "eda_6_demographics_coverage")
    w, wo = df.loc[True], df.loc[False]
    return {
        "table": df.round(3).reset_index().to_dict("records"),
        "sentence": (
            f"The {int(w.households)} households with demographics ({w.households / (w.households + wo.households):.0%} "
            f"of the panel) bring {w.share_of_revenue:.0%} of revenue and have {w.median_weekly_spend / wo.median_weekly_spend:.1f}x "
            "the median weekly spend, so demographic findings describe heavy shoppers, not everyone."
        ),
    }


def run() -> dict:
    con = connect(read_only=False)  # temp tables need a writable connection
    try:
        results = {
            "growth_segments": growth_segments(con),
            "cohort_retention": cohort_retention(con),
            "revenue_concentration": revenue_concentration(con),
            "targeting_bias": targeting_bias(con),
            "redemption": redemption(con),
            "demographics_coverage": demographics_coverage(con),
        }
    finally:
        con.close()
    (config.RESULTS_DIR / "eda.json").write_text(json.dumps(results, indent=2, default=float))
    for k, v in results.items():
        log.info("%s: %s", k, v["sentence"])
    return results
