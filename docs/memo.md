# Memo: what the coupon booklets did, and how to send the next ones

**To:** Head of CRM
**From:** Sameer Singh
**Data:** 2,469 loyalty-card households, January to December 2017, four waves of personalised coupon booklets

## Bottom line

The booklets added about **$1 per household per week**, which is roughly what they cost. The big gap between
recipients and everyone else comes from who we chose to send them to, not from the booklets. Choosing recipients by
predicted response instead of past spend looks like it could make the booklets clearly profitable, and the next wave
should test that with a proper holdout.

## What happened

1. **Recipients spend $29 a week more than non-recipients, but they already spent 1.4 to 2.6 times more before any
   booklet was sent.** Compared against look-alike households (same spending, trips, trend and coupon habits
   beforehand), the booklets raised spend by **$0.97 a week, or 2.4%** (95% range: -$0.85 to +$2.79).
2. **Past waves were roughly break-even.** At an assumed 28% margin and $1.50 per booklet, break-even is $0.74 a week
   of extra spend. The 95% range runs from losing $3.22 to making $4.17 per booklet.
3. **Most recipients never use them.** Only 17% of booklet recipients redeemed a single coupon.

## What I recommend

1. **Pick recipients by predicted response, not by past spend.** Today's lists skew to our heaviest shoppers. The model
   expects the most loyal heavy shoppers to show no response at all (they buy anyway), while regular shoppers who are
   starting to drift respond the most. On two past waves the model had not seen, the 30% of recipients it ranked highest
   responded with about **$4.00 a week**, against $0.37 for all recipients. If that holds, it's roughly **$6.60 profit
   per booklet**, compared with a loss of about $0.76 per booklet when mailing everyone. This is promising, not proven:
   the interval on that gain still includes zero.
2. **Keep a random holdout in every wave.** Hold back 20% of the chosen list. An effect of $1 a week needs about
   **8,300 households per group** to detect even with variance reduction (CUPED), so one wave can't settle it; pooling
   holdouts across waves can. This is the only way to get a clean read, and it gives us clean data to improve the model.
3. **Use the lapse-risk list for retention, not as the mailing list.** It flags regular shoppers likely to go quiet in
   the next 8 weeks (its top 10% contain 36% of the households that do). Risk and response overlap but aren't the same:
   even in the riskiest tenth, only about half are expected to respond enough to cover the booklet.

## Caveats

* Matching only controls for what we can see in the data. Pre-trends were flat and a fake start date found nothing,
  which helps, but a randomised holdout is still the real test.
* Margin and booklet cost are assumptions. By the model's estimates, mailing only predicted responders stays profitable
  for every combination tested (20 to 35% margin, $0.50 to $3.00 per booklet), while mailing everyone turns into a loss
  once a booklet costs about $2. The Targeting Simulator lets you try your own numbers.
* The impact estimate describes lighter recipients (about $40 a week). Our very heaviest shoppers have no look-alikes
  among non-recipients, so we can't say much about them, which is another reason to hold some of them out.

Interactive version: the Streamlit app (Campaign Impact and Targeting Simulator pages). Full method: notebook 2.
