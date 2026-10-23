# Decision log: what I tried, what broke, what changed

Most of the interesting work in this project was catching things that looked fine and weren't. Each entry has
the symptom, the cause, the fix and what it did to the numbers.

## 1. Daylight saving time broke the week boundaries

**Symptom.** A data check asserting that every full week has exactly 7 local days failed for 34 weeks.

**Cause.** Timestamps are stored in UTC and retailer weeks flip at midnight US Eastern. I had converted with a fixed
"minus 5 hours", which is only right in winter. During daylight saving the first hour of each Monday landed in the
previous week.

**Fix.** A proper `America/New_York` conversion (`to_local()` in `sql/01_dim_week.sql`). The check now passes, and
it's also a unit test on the synthetic data.

## 2. Campaign weeks before 2017 came out wrong

Some campaigns start in late 2016, so their week numbers are zero or negative. DuckDB's integer division `//`
truncates toward zero, which put 25 December 2016 in week 1 instead of week 0. The `week_of()` macro uses
`floor()` and has a test for exactly this date.

## 3. Matching: balance vs sample size

The first version was textbook: 1:1 nearest neighbour on the propensity score, caliper 0.2 SD. Balance on waves
2 to 4 was not good enough (max |SMD| around 0.2), because the heaviest recipients have no comparable
non-recipients at all.

| matching spec | recipients matched | mean abs SMD | max abs SMD | pooled effect, $/week (± 95% CI) |
|---|---|---|---|---|
| propensity score only, caliper 0.2 SD | 47% | 0.074 | 0.208 | +1.11 ± 1.67 |
| + exact match on other campaigns and earlier booklets | 44% | 0.079 | 0.210 | +0.94 ± 1.73 |
| + Mahalanobis on spend and activity, with replacement | 98% | 0.052 | 0.207 | +1.66 ± 2.85 |
| **exact strata + Mahalanobis inside a 0.05 SD caliper (final)** | **40%** | **0.038** | **0.109** | **+0.97 ± 1.82** |

Matching with replacement matched almost everyone but reused each control 3 to 4 times, so the interval got
wider. The final spec gives the best balance. The estimate barely moves across specs, which is the reassuring part. The cost
is that the estimate describes the 40% of recipients who have a real look-alike (lighter shoppers, about
$40/week); it says nothing about the very heaviest shoppers.

## 4. The regression-to-the-mean trap in segment effects

My first segment analysis cut recipients into light / medium / heavy by spend in the **same baseline window** used
by the DiD. It found a clean, significant pattern:

| tier (cut on baseline spend) | effect, $/week | 95% CI |
|---|---|---|
| light | +4.27 | +1.98 to +6.56 |
| medium | +2.77 | -0.09 to +5.63 |
| heavy | -4.12 | -8.07 to -0.17 |

It's an artefact. A household whose baseline happened to be unusually quiet lands in "light" and then bounces back
on its own, which looks exactly like a booklet effect (and the reverse for "heavy"). Cutting tiers on spend from
**before** the baseline window removes the selection on noise:

| tier (cut on earlier spend) | effect, $/week | 95% CI |
|---|---|---|
| light | -1.81 | -4.34 to +0.71 |
| medium | +0.57 | -2.53 to +3.67 |
| heavy | +1.78 | -2.46 to +6.03 |

No real differences. Both versions are kept in `reports/results/causal.json` and shown in the app.

## 5. The churn target

First target: "spend over the next 8 weeks falls below half of the last 16 weeks". ROC-AUC about 0.70, and the
top 10% caught only 20 to 23% of cases. Grocery spend swings for boring reasons, so most "drops" are noise.

Final target: a **regular** shopper (8+ of the last 16 weeks) **goes quiet** (2 or fewer of the next 8 weeks). Same
features, same validation: ROC-AUC 0.83, and the top 10% catch 36 to 40% of cases depending on the model. It's
also the version a CRM team can act on. Notebook 3 reruns this comparison live.

## 6. Features that drifted by construction

The drift check (PSI between the first training week and the scoring week) flagged "booklets received to date"
(PSI 6.4) and "weeks since first purchase" (PSI 0.95). They grow mechanically through the year, so at scoring time
the model saw values it never saw in training. Tenure is also meaningless here because the data starts on 1 January
for everyone.

Fix: every feature now uses a fixed 4, 8 or 16 week window. The lapse model's AUC didn't change (0.83). The uplift
model's out-of-time calibration slope went from 0.49 to 0.93, so the drifting features had been hurting the model
that depends on transferring across waves.

## 7. Calibration: isotonic overfit the top end

Isotonic regression on the calibration weeks gave a similar expected calibration error, but it's a step function
fitted on very few high-risk households, and it pushed one household straight to a 100% lapse probability. Platt
scaling (two parameters) is smooth and gave an ECE of 0.014 vs 0.015 for isotonic. Since the targeting step
multiplies probabilities by dollars, I'd rather have smooth than jumpy.

## 8. LightGBM didn't beat logistic regression

On the lapse model, LightGBM and a well-prepared logistic regression tie (ROC-AUC 0.828 vs 0.827, PR-AUC 0.27 vs
0.28). Both beat the day-one rule (0.66) by a wide margin. The improvement came from the target and the features,
not the algorithm. I kept LightGBM because TreeSHAP gives per-household reasons in the app and API, but the
logistic model would be a perfectly good production choice.

## 9. Picking the uplift learner

I started with the DR-learner because it's the most principled under targeted assignment. Out of time it didn't
hold up (calibration slope 0.22, not significant). I then made the choice a fixed rule: the learner that captures the
most uplift in the planted-effect benchmark under real (non-random) targeting. That's the X-learner (74% of the
oracle vs 54% for DR, 65% for T). Because I changed the rule after seeing one result, all three learners'
out-of-time results are always reported side by side.

## 10. Outlier caps leaking across waves

The uplift outcome is winsorised at the 1st and 99th percentiles. The first version computed the caps on all waves,
including the held-out one. Small leak, but a leak. Caps now come from the training waves only (top-30% gain moved
from +3.08 to +3.63 $/week).

## 11. Synthetic runs could overwrite real results

`couponroi all --synthetic` (what CI runs) originally wrote to the same folders as the real run, so running it locally
would have replaced the real models and app bundle. Synthetic runs now write to `build/synthetic/`.

## 12. Reruns weren't reproducible

A clean-clone rerun reproduced the SQL, EDA, causal and drift results bit for bit, but the LightGBM-based results
(lapse model, uplift, policy) drifted in the third decimal, with identical package versions. LightGBM sums histograms
across threads, and the order depends on scheduling, so it isn't deterministic by default. Setting
`deterministic=True` and `force_row_wise=True` fixed it: two consecutive runs now produce byte-identical results
files and model files.

## 13. CI and Docker targeted a Python the models couldn't run on

The first Dockerfile and CI used Python 3.11, but the models were trained with NumPy 2.5, which needs Python 3.12 or
newer, so the API image would have failed to build. Both now use Python 3.13 (the training version), and
`requirements-api.txt` pins the exact training versions so the pickled uplift model always loads the same way.
