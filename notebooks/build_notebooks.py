"""Generate and execute the walkthrough notebooks.

    python notebooks/build_notebooks.py          (after `couponroi all`)

The notebooks are generated from this file so they never drift from the code: each one
runs the key SQL and model code live against the warehouse and pulls the rest from
reports/results/*.json. Outputs are saved, so GitHub shows them without running anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent

SETUP = """\
import json
import warnings

import pandas as pd
from IPython.display import Image, Markdown, display

from couponroi import config
from couponroi.warehouse import connect

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
con = connect()  # read-only DuckDB warehouse built by `couponroi build`
R = lambda name: json.loads((config.RESULTS_DIR / f"{name}.json").read_text())  # noqa: E731
fig = lambda name, width=900: display(Image(filename=str(config.FIG_DIR / f"{name}.png"), width=width))  # noqa: E731
sql = lambda path: (config.SQL_DIR / path).read_text()  # noqa: E731"""


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


# --------------------------------------------------------------------------- 01


NB1 = [
    md("""
# 1. Data, SQL and EDA

**The business:** a grocery retailer with 2,469 loyalty-card households, one year of transactions (2017),
and 27 marketing campaigns. The big ones are four waves of personalised coupon booklets ("Type A").

**This notebook:** how the raw tables become a clean warehouse, what had to be cleaned, and six questions a
CRM manager would ask before trusting any model. Every query lives in `sql/` so it can be read without Python.
"""),
    code(SETUP),
    md("## The warehouse\n\n8 raw tables in, a small star schema out. The main table for analysis is `hh_week`: one row per household per week, dense (weeks with no shopping are zeros, which matters for averages)."),
    code("""
tables = ["raw_transactions", "raw_promotions", "raw_products", "raw_campaigns", "raw_coupons",
          "fct_transactions", "dim_household", "dim_week", "dim_campaign", "fct_campaign_exposure", "hh_week"]
pd.DataFrame({"table": tables, "rows": [con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables]})
"""),
    md("""
## Data checks that run on every build

These are dbt-style tests in `src/couponroi/warehouse.py`. The build fails if any of them fail.
"""),
    code("pd.DataFrame(R('data_quality')['checks'])"),
    md("""
## Things that needed cleaning

* **Time zones.** Timestamps are stored in UTC, but retailer weeks flip at midnight US Eastern. A fixed
  "minus 5 hours" shift was wrong for 34 weeks because of daylight saving, so local time uses a proper
  `America/New_York` conversion. The check above ("weeks 2-52 have exactly 7 local days") is what caught it.
* **Partial weeks.** Week 1 is a single Sunday, and week 53 lost Christmas Day. Week 1 is never used as a baseline.
* **Fuel.** Fuel quantities are in odd units and fuel prices move spend for reasons coupons can't touch, so the
  main outcome is spend excluding fuel.
* **Orphan products.** About 4.8K transaction lines point at products missing from the product table. They get an
  UNKNOWN placeholder instead of being silently dropped by an inner join.
* **Demographics.** Only 801 of 2,469 households have them, and they're not a random subset (question 6).
"""),
    code("""
con.execute(\"\"\"
    SELECT week, week_start, week_end, days_observed, baskets, active_households, round(sales) AS sales
    FROM dim_week WHERE week IN (1, 2, 52, 53) ORDER BY week
\"\"\").df()
"""),
    md("## Q1. Who is growing and who is shrinking?"),
    code("""
seg = con.execute(sql("analysis/growth_segments.sql")).df()
seg.segment.value_counts()
"""),
    code("fig('eda_1_growth_segments'); display(Markdown(R('eda')['growth_segments']['sentence']))"),
    md("## Q2. Do households stick around?"),
    code("""
ret = con.execute(sql("analysis/cohort_retention.sql")).df()
ret.pivot(index="cohort_block", columns="blocks_since_first", values="retention").head(4).round(2)
"""),
    code("fig('eda_2_cohort_retention', 700); display(Markdown(R('eda')['cohort_retention']['sentence']))"),
    md("## Q3. How concentrated is revenue? (window functions + NTILE)"),
    code("print(sql('analysis/revenue_concentration.sql'))"),
    code("fig('eda_3_revenue_concentration', 600); display(Markdown(R('eda')['revenue_concentration']['sentence']))"),
    md("""
## Q4. Who gets the coupon booklets?

This is the question everything else depends on. If recipients were picked at random, comparing them to
non-recipients would measure the booklet. They weren't.
"""),
    code("con.execute(sql('analysis/targeting_bias.sql')).df()"),
    code("fig('eda_4_targeting_bias', 760); display(Markdown(R('eda')['targeting_bias']['sentence']))"),
    md("## Q5. How many people actually redeem?"),
    code("con.execute(sql('analysis/redemption_by_type.sql')).df()"),
    code("fig('eda_5_redemption', 560); display(Markdown(R('eda')['redemption']['sentence']))"),
    md("## Q6. Can we trust the demographics?"),
    code("con.execute(sql('analysis/demographics_coverage.sql')).df()"),
    code("fig('eda_6_demographics_coverage'); display(Markdown(R('eda')['demographics_coverage']['sentence']))"),
    md("""
## What this sets up

* Booklets went to heavier shoppers, so the impact analysis has to compare recipients with look-alikes
  (notebook 2), not with everyone else.
* Outright churn is rare. The useful risk signal is a *regular* shopper going quiet (notebook 3).
* Demographics describe heavy shoppers only, so the models use behaviour, not demographics.
"""),
]

# --------------------------------------------------------------------------- 02


NB2 = [
    md("""
# 2. Did the coupon booklets work?

Recipients spent far more than non-recipients during the campaigns. The question is how much of that the
booklet *caused*. Because recipients were picked from their purchase history, the naive gap mostly measures
the targeting rule.

**Design:** for each of the 4 Type A waves:
1. a baseline window of up to 8 weeks before the wave (never overlapping the previous wave),
2. match each recipient to a non-recipient who looked the same in the baseline,
3. compare the *change* in weekly spend (difference-in-differences on matched pairs),
4. check it: covariate balance, event study, placebo, robustness specs, a matching-free estimator.
"""),
    code(SETUP + "\nfrom couponroi.causal import analyze_wave, list_waves, unit_table"),
    code("""
waves = list_waves(con)
pd.DataFrame([{**w.__dict__, "pre_weeks": w.pre_weeks, "post_weeks": w.post_weeks} for w in waves])
"""),
    md("## One wave up close: wave 2 (campaign 8)\n\nOne row per household that shopped in the baseline. `dy` is campaign-period weekly spend minus baseline weekly spend."),
    code("""
u = unit_table(con, waves[1])
g = u.groupby("treated")[["pre_spend", "pre_trips", "pre_active_share", "prior_waves", "co_exposure_post", "post_spend", "dy"]].mean()
display(g)
print(f"recipients spent {g.loc[1, 'pre_spend'] / g.loc[0, 'pre_spend']:.1f}x more per week BEFORE the booklet; "
      f"change in weekly spend: recipients {g.loc[1, 'dy']:+.2f}, non-recipients {g.loc[0, 'dy']:+.2f}")
"""),
    md("""
The levels are far apart before anything was mailed, but the *changes* are close. That's the whole story in one
table; the matching below makes it rigorous by comparing each recipient with a non-recipient who looked the same.

**Matching:** propensity score (logistic regression on 12 baseline covariates), exact match on two discrete
variables (got another campaign during the wave; number of earlier booklets), then the nearest look-alike on
spend and activity (Mahalanobis distance) inside a tight propensity caliper.
"""),
    code("""
res, matched = analyze_wave(con, waves[1])
pd.DataFrame(res["balance"]).set_index("covariate").round(3)
"""),
    code("""
pd.DataFrame({k: res[k] for k in ["naive_post", "did_unmatched", "did_matched", "did_matched_dr"]}).T[["est", "lo", "hi", "p"]]
"""),
    md("## All four waves"),
    code("""
c = R("causal")
rows = [{"wave": w["wave"], "recipients": w["n_treated"], "matched": w["n_pairs"],
         "naive": w["naive_post"]["est"], "matched DiD": w["did_matched"]["est"],
         "CI low": w["did_matched"]["lo"], "CI high": w["did_matched"]["hi"],
         "max |SMD| after": w["max_abs_smd_after"]} for w in c["waves"]]
p = c["pooled"]
rows.append({"wave": "pooled", "recipients": p["n_treated"], "matched": p["n_pairs"], "naive": p["naive_post"]["est"],
             "matched DiD": p["did_matched"]["est"], "CI low": p["did_matched"]["lo"], "CI high": p["did_matched"]["hi"]})
display(pd.DataFrame(rows))
display(Markdown(c["headline"]))
"""),
    code("fig('causal_1_balance', 1000)"),
    code("fig('causal_2_estimates', 1000)"),
    md("""
## Is the comparison fair? Event study

If matched recipients and their look-alikes were already drifting apart before the booklet, the DiD would be
biased. The weeks before 0 should hover around zero.
"""),
    code("fig('causal_3_event_study', 820)"),
    code("pd.DataFrame(R('causal')['event_study'])[['rel_week', 'waves', 'est', 'lo', 'hi', 'p']]"),
    md("## Robustness and placebo"),
    code("pd.DataFrame(R('causal')['robustness'])[['spec', 'est', 'lo', 'hi', 'p', 'n_pairs']]"),
    code("fig('causal_4_robustness', 820)"),
    md("""
## Who responds? And a trap worth showing

My first version cut segments (light / medium / heavy shoppers) on spend in the *same* baseline window used for
the DiD. It found big effects for light shoppers and negative effects for heavy ones:
"""),
    code("pd.DataFrame(R('causal')['segments_regression_to_mean_trap'])[['segment', 'n_pairs', 'est', 'lo', 'hi']]"),
    md("""
That's regression to the mean. A household with an unusually quiet baseline lands in "light" and then bounces
back, which looks like a booklet effect. Defining segments on behaviour from *before* the baseline fixes it,
and the differences disappear:
"""),
    code("pd.DataFrame(R('causal')['segments'])[['dimension', 'segment', 'n_pairs', 'est', 'lo', 'hi']]"),
    code("fig('causal_6_segments', 820)"),
    md("## Planning the next test: CUPED and power"),
    code("""
cu, pw = R("causal")["cuped"], R("causal")["power"]
display(pd.DataFrame({k: cu[k] for k in ["post_only", "did", "cuped"]}).T[["est", "se", "lo", "hi"]])
print(f"CUPED theta = {cu['theta']:.2f}, variance reduction vs post-only = {cu['variance_reduction_vs_post']:.0%}")
print(f"households per arm to detect {pw['effect_used']:.2f} $/wk at 80% power: "
      f"{pw['n_per_arm_plain']:,.0f} plain, {pw['n_per_arm_cuped']:,.0f} with CUPED (a typical wave has ~1,100)")
"""),
    code("fig('causal_5_cuped_power', 1000)"),
    md("## Did a booklet pay for itself? (simulated)"),
    code("""
roi = R("causal")["roi"]
print(json.dumps(roi["assumptions"]))
print(f"break-even: {roi['breakeven_lift_per_week']:.2f} $/wk ({roi['breakeven_rel']:.1%} of baseline spend)")
print(f"profit per booklet: {roi['simulated_profit_per_booklet']:+.2f} (95% CI {roi['simulated_profit_lo']:+.2f} to {roi['simulated_profit_hi']:+.2f})")
"""),
    md("""
## Conclusion

The booklets added about **$1 per household per week** (+2.4%), which is roughly break-even at a 28% margin and
$1.50 a booklet. The naive comparison would have said $29. A single wave is far too small to measure an effect
this size, so the next campaign should include a randomised holdout, pooled across waves, with CUPED.
"""),
]

# --------------------------------------------------------------------------- 03


NB3 = [
    md("""
# 3. Which regular shoppers are about to go quiet?

**Population:** households that shopped in at least 8 of the last 16 weeks ("regulars").
**Target:** they shop in 2 or fewer of the next 8 weeks.
**Validation:** rolling-origin backtest. For each test week the model only sees cutoffs whose labels were
complete by then: train, then calibrate on the most recent of those, then test.
"""),
    code(SETUP + """
from sklearn.metrics import roc_auc_score
from couponroi.churn import split_for, topk_metrics
from couponroi.features import FEATURE_DESCRIPTIONS, features_at, is_regular, labels_at
from couponroi.models import LogitModel"""),
    md("""
## Choosing the target

My first target was "spend over the next 8 weeks falls below half of the last 16". It's noisy: grocery spend
swings week to week for boring reasons. This cell compares three definitions with the same features and a plain
logistic regression on the same 3 backtest folds.
"""),
    code("""
cfg = config.SETTINGS.churn
cutoffs = cfg.cutoffs()
frames = []
for c in cutoffs:
    f = features_at(con, c)
    y = labels_at(con, c)[["household_id", "lapse", "spend_halved", "full_lapse"]]
    frames.append(f.merge(y, on="household_id"))
frame = pd.concat(frames, ignore_index=True)
frame = frame[is_regular(frame)]

rows = []
for target in ["spend_halved", "lapse", "full_lapse"]:
    for test in cfg.test_cutoffs:
        train, cal = split_for(test, cutoffs, cfg)
        tr, te = frame[frame.cutoff.isin(train + cal)], frame[frame.cutoff == test]
        p = LogitModel().fit(tr, tr[target].to_numpy()).predict(te)
        rows.append({"target": target, "test_week": test, "base_rate": te[target].mean(),
                     "roc_auc": roc_auc_score(te[target], p), **topk_metrics(te[target].to_numpy(), p, 0.1)})
pd.DataFrame(rows).groupby("target")[["base_rate", "roc_auc", "recall_at_k", "lift_at_k"]].mean()
"""),
    md("""
"Regular goes quiet" is both more predictable and more useful: it's a real change in behaviour a CRM team can act
on. Full lapse (zero trips) is rarer still, about 1% of regulars, too rare to build a programme around.

## The training frame
"""),
    code("""
frame.groupby("cutoff").agg(households=("lapse", "size"), lapse_rate=("lapse", "mean")).T.round(3)
"""),
    md("""
Every feature is computed from data up to the cutoff week only. `tests/test_leakage.py` proves it: it scrambles
every row after the cutoff (spend, trips, campaigns) and checks the features don't move.
"""),
    code("pd.Series(FEATURE_DESCRIPTIONS).to_frame('what it measures')"),
    md("## Backtest results"),
    code("""
ch = R("churn")
folds = pd.DataFrame(ch["folds"])
folds.pivot(index="model", columns="test_cutoff", values="roc_auc").round(3)
"""),
    code("pd.DataFrame(ch['summary']).set_index('model')[['pr_auc', 'roc_auc', 'recall_at_k', 'lift_at_k', 'brier', 'ece']].round(3)"),
    code("fig('churn_1_backtest', 1000); display(Markdown(ch['headline']))"),
    md("""
**Honest note:** LightGBM and logistic regression are tied (differences are smaller than the fold-to-fold
swing). Both beat the day-one rule by a wide margin. The gain came from the target definition and the features,
not the algorithm. LightGBM is kept because its TreeSHAP values give per-household reasons in the app and API;
the logistic model would be a perfectly good production choice.

## Calibration

The targeting maths multiplies probabilities by dollars, so they need to mean what they say. Platt scaling is
fitted on the calibration weeks (always earlier than the test week). Isotonic regression was tried first and
overfit the sparse top end (it jumped to 100% for a few households).
"""),
    code("folds[folds.model == 'LightGBM'][['test_cutoff', 'ece_uncalibrated', 'ece_isotonic', 'ece', 'platt_a', 'platt_b']]"),
    code("fig('churn_2_calibration', 520)"),
    md("## What drives the risk (SHAP)"),
    code("fig('churn_3_shap_importance', 760)"),
    code("fig('churn_4_shap_beeswarm', 820)"),
    code("fig('churn_5_shap_cases', 1000)"),
    md("""
## Drift

PSI between the first training week and the scoring week. The first version of the features had "to date"
counters (tenure, booklets received to date) that grow mechanically through the year. This check flagged them,
so every feature now uses a fixed window. What's left flagged is campaign exposure, which really did ramp up
in the second half of the year.
"""),
    code("pd.DataFrame(R('drift')['table']).head(8)"),
    code("fig('drift_1_psi', 760)"),
]

# --------------------------------------------------------------------------- 04


NB4 = [
    md("""
# 4. Who should get the next booklet?

Risk answers "who might leave". Targeting needs "whose spend goes up *because* they got a booklet", the
conditional treatment effect, or uplift. Someone at high risk may not respond at all; a loyal heavy shopper buys
anyway. Three learners are compared (T, X and DR learners, all on LightGBM).

Uplift models can't be checked against labels, because nobody is observed both with and without a booklet.
So there are two checks before anything is used:

1. **Planted effects.** Real households and the real (non-random) targeting, with outcomes simulated from known
   per-household effects. If a learner can't find planted heterogeneity, its real output means nothing.
2. **Out of time.** Train on earlier waves, predict the next one, and compare with what the matched pairs
   actually did.
"""),
    code(SETUP),
    md("## Check 1: planted effects"),
    code("""
u = R("uplift")
pd.DataFrame(u["semi_synthetic"]).pivot(index="learner", columns="scenario", values="policy_share_of_oracle").round(3)
"""),
    code("fig('uplift_1_semisynthetic', 820)"),
    md("""
Targeting the top 30% by the X-learner captures about three quarters of the best possible uplift, against 20% for
random targeting, even with the real biased assignment. The learner used downstream is the winner of this
benchmark (the rule is in `uplift.choose_learner`), not the winner of the real out-of-time check below, and all
three learners are shown there.

## Check 2: out of time on real waves
"""),
    code("pd.DataFrame(u['out_of_time']).round(3)"),
    code("fig('uplift_2_out_of_time', 1000); display(Markdown(u['headline']))"),
    md("""
The X-learner's predictions line up with what actually happened (calibration slope near 1 and significant),
while the T and DR learners don't hold up. The top-30% gain is promising but its interval still touches zero,
so predictions are shrunk toward the average effect by the calibration slope before they drive any decision.

## The decision: who to mail, and how many

Expected profit of mailing a household = shrunk uplift x weeks x margin - cost. Mail everyone with positive
expected profit, best first.
"""),
    code("""
pol = R("policy")
display(pd.Series(pol["profit_at_budget"]).rename("simulated profit at current budget").to_frame())
display(Markdown(pol["headline"]))
"""),
    code("fig('policy_1_expected_profit', 820)"),
    md("""
That curve scores every ranking with the model's own estimates, so the model wins by construction. The honest
version uses held-out waves and what the matched pairs actually spent:
"""),
    code("""
hold = pd.DataFrame({"top 30% of the list": pol["holdout_profit_per_1000_at_30pct"],
                     "everyone": pol["holdout_profit_per_1000_all"]})
(hold / pd.Series({"top 30% of the list": 300, "everyone": 1000})).round(2).rename(
    columns=lambda c: f"{c}: profit per booklet ($, simulated)")
"""),
    code("fig('policy_2_holdout_profit', 760)"),
    md("## How sensitive is this to the assumptions?"),
    code("""
s = pd.DataFrame(pol["sensitivity"])
s.pivot(index="margin", columns="cost", values="share_to_mail").round(2)
"""),
    code("fig('policy_3_sensitivity', 700)"),
    md("## The output a CRM team would use"),
    code("pd.read_csv(config.RESULTS_DIR / 'next_wave_mailing_list.csv').head(10)"),
    md("""
## What I'd do next

* Run the next wave as an experiment: target by uplift, keep a random holdout in every wave, pool the holdouts.
  That turns this observational estimate into a measured one and gives clean data to retrain the uplift model.
* Add real margins by category (coupons on high-margin categories are worth more).
* Retrain the lapse model monthly and watch the drift report.
"""),
]


def build(name: str, cells: list, execute: bool = True) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path = HERE / name
    if execute:
        ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
    nbformat.write(nb, path)
    print("wrote", path)
    return path


if __name__ == "__main__":
    run = "--no-execute" not in sys.argv
    build("01_sql_and_eda.ipynb", NB1, run)
    build("02_campaign_impact.ipynb", NB2, run)
    build("03_lapse_model.ipynb", NB3, run)
    build("04_uplift_and_targeting.ipynb", NB4, run)
