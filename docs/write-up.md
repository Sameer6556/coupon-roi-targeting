# Coupon booklet recipients spent $29 a week more. The booklets caused about $1 of it.

*A write-up of my Coupon ROI & Targeting Engine project. Code, notebooks and a live app are in the repo.*

A grocery retailer sends personalised coupon booklets to about 1,100 of its loyalty-card households at a time. I had a
year of their real transactions (2,469 households, 1.47 million line items, 27 campaigns) and three questions a CRM
team actually asks: did the booklets work, who is about to drift away, and who should get the next one?

## The obvious answer is wrong

During the four booklet waves, recipients spent **$29 a week more** than everyone else. Easy win for marketing.

Except the booklets weren't sent at random. Each household's coupons were picked from its own purchase history, and
the households picked were the retailer's best customers. In the 8 weeks *before* any booklet arrived, recipients
were already spending 1.4 to 2.6 times as much as non-recipients. Comparing them afterwards mostly measures the
targeting rule.

This is the most common mistake in marketing analytics, and it's the reason the rest of the project exists.

## Building a fair comparison

The idea is simple: for each booklet recipient, find a non-recipient who looked the same before the wave, and
compare how their spending *changed*.

* **Same before.** I matched on 12 things measured in the weeks before each wave: weekly spend, trips, how many weeks
  they shopped, departments, store-brand and promo habits, coupon use, spend trend, spend before that, earlier
  booklets, and other campaigns they got at the same time. Two of those had to match exactly; then each recipient got
  the closest non-recipient on spend and activity, but only among households with a very similar probability of
  being picked (the propensity score).
* **Compare changes, not levels.** Difference-in-differences: (recipient's change in weekly spend) minus (match's
  change). Anything that hits both, like Christmas, cancels out.

Result: **+$0.97 a week**, or 2.4% of what matched recipients spent at baseline. The 95% interval runs from -$0.85 to
+$2.79. The $29 was almost entirely selection.

## Checking I wasn't fooling myself

A number like that is only worth something if you try hard to break it.

* **Balance.** After matching, every covariate's standardised difference is below 0.11 (before matching it was up
  to 1.5).
* **Pre-trends.** An event study shows the weekly gap between recipients and their matches hovering around zero for
  the 6 weeks before each wave. If they'd been drifting apart already, DiD would be biased.
* **Placebo.** Pretend each wave started halfway through its baseline, when nothing happened. The estimate is +$0.28,
  as it should be.
* **Other reasonable choices.** Seven alternative specifications (a looser caliper, matching with replacement,
  propensity score only, dropping households with other campaigns, fewer covariates, a regression-adjusted version,
  and a doubly robust estimator that doesn't match at all) all land between +$0.53 and +$1.92 a week.

The one that taught me the most was a mistake. I first split recipients into light, medium and heavy shoppers using
their spend in the *same* baseline window, and found a beautiful pattern: light shoppers responded by +$4.27 a week,
heavy shoppers by -$4.12. It was regression to the mean. A household that happened to have a quiet baseline lands in
"light" and then bounces back on its own. Splitting on spend from *before* the baseline made the pattern vanish. I
kept both versions in the project because it's such an easy trap to fall into.

## So were the booklets worth it?

The data has no margins or costs, so this part is simulated: assume a 28% gross margin and $1.50 to print and mail a
booklet. A booklet is in hand for about 7 weeks, so break-even is $0.74 a week of extra spend. The estimate is $0.97.
Roughly break-even, with an interval that runs from losing $3.22 to making $4.17 per booklet.

And it's too small to measure well. Even using pre-period spend to cut the noise (CUPED, which removed 68% of the
variance here), detecting a $1 effect needs about 8,300 households per group. A wave has about 1,100.

## Who should get the next one?

If the average effect is small, the next question is whether it's small for everyone. An uplift model estimates the
effect for each household. I tried three (T, X and DR learners on LightGBM) and couldn't check them the normal way,
because nobody is observed both with and without a booklet. So two other checks:

1. **Planted effects.** Keep the real households and the real (biased) targeting, simulate outcomes with known
   per-household effects, and see if the learners recover them. The X-learner's top 30% captured 75% of the best
   possible uplift; random targeting gets 20%.
2. **Out of time.** Train on earlier waves, predict a later wave, and compare with what its matched pairs actually did.
   The X-learner held up (a calibration slope of 0.95, p = 0.01). The other two didn't.

On those held-out waves, mailing every recipient lost about **$0.76 per booklet**. The 30% the model ranked highest
responded with about $3.26 a week, roughly **$5.10 profit per booklet**. The heaviest-spender rule the retailer seems
to use made $0.38. That's the most exciting number in the project, and also the least certain: the interval on the
gain still touches zero, so predictions get shrunk toward the average before they drive decisions. It's touchy too.
Before I made reruns reproducible, noise in the last digit of the inputs moved it anywhere from $2.82 to $6.98.

The pattern behind it is the classic one. The model expects the most loyal heavy shoppers to show no response at all,
because they'd buy anyway. The response is among regular shoppers who are starting to drift.

## Who is about to drift?

A separate model flags regular shoppers (8 of the last 16 weeks) who are about to go quiet (2 or fewer of the next 8).
It's validated the way it would be used: for each test week it only trains on data whose outcomes were known by then.
On three out-of-time folds it reaches a ROC-AUC of 0.83, and the 10% it flags contain 36% of the households that go
quiet. Two honest notes: my first target ("spend halves") was mostly noise (AUC 0.70), and LightGBM didn't beat a
well-prepared logistic regression. The gain came from framing the problem better, not from the algorithm.

## What I'd tell the CRM team

Send the next wave by predicted response, and hold back a random 20% of the chosen list in every wave. Pool the
holdouts. That turns an observational estimate into a measured one and gives clean data to improve the model. The
memo in the repo says this in one page.

## What I learned

* **Ask how the treatment was assigned before anything else.** That one question changed the answer from $29 to $1.
* **The bugs that matter don't crash.** A daylight saving shift put the first hour of 34 weeks in the wrong week; a
  segment analysis found effects that were pure regression to the mean; two features drifted just because time passed;
  parallel sums in the warehouse made every rerun a tiny bit different, and LightGBM turned that into a $4 swing in
  the headline profit. Each one was caught by a check I wrote to break my own results, not by an error message.
* **Say how sure you are.** Most of the interesting numbers here have wide intervals. Saying so, and designing the
  test that would settle it, is more useful than a confident number that's wrong.

*Built with Python, DuckDB, LightGBM, statsmodels, SHAP, MLflow, Streamlit and FastAPI. Data: 84.51° The Complete
Journey (CC0).*
