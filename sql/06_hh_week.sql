-- hh_week: the household x week panel every analysis is built on.
--
-- It is dense on purpose: a household that didn't shop in a week still gets a row with
-- zeros. Averages over windows ("spend per week in the 8 weeks before the campaign")
-- are only correct if the zero weeks are there.
--
-- spend_xf = spend excluding fuel, the main outcome everywhere.

CREATE OR REPLACE TABLE hh_week AS
WITH weekly AS (
    SELECT
        household_id,
        week,
        sum(sales_value) AS spend,
        sum(CASE WHEN is_fuel THEN 0 ELSE sales_value END) AS spend_xf,
        count(DISTINCT basket_id) AS trips,
        count(DISTINCT department) FILTER (WHERE NOT is_fuel) AS departments,
        sum(CASE WHEN brand = 'Private' AND NOT is_fuel THEN sales_value ELSE 0 END) AS private_spend,
        sum(CASE WHEN (on_display OR in_mailer) AND NOT is_fuel THEN sales_value ELSE 0 END) AS promo_spend,
        sum(discount_amount) AS discount_amount,
        sum(coupon_disc + coupon_match_disc) AS coupon_discount
    FROM fct_transactions
    GROUP BY household_id, week
),
redeemed AS (
    SELECT household_id, week, count(*) AS redemptions
    FROM fct_redemptions
    GROUP BY household_id, week
),
grid AS (
    SELECT h.household_id, w.week
    FROM dim_household h
    CROSS JOIN dim_week w
),
live_campaigns AS (
    -- number of campaigns a household had "in hand" during each week
    SELECT g.household_id, g.week, count(*) AS active_campaigns
    FROM grid g
    JOIN fct_campaign_exposure e
      ON e.household_id = g.household_id
     AND g.week BETWEEN e.start_week AND e.end_week
    GROUP BY g.household_id, g.week
)
SELECT
    g.household_id,
    g.week,
    coalesce(w.spend, 0) AS spend,
    coalesce(w.spend_xf, 0) AS spend_xf,
    coalesce(w.trips, 0) AS trips,
    coalesce(w.departments, 0) AS departments,
    coalesce(w.private_spend, 0) AS private_spend,
    coalesce(w.promo_spend, 0) AS promo_spend,
    coalesce(w.discount_amount, 0) AS discount_amount,
    coalesce(w.coupon_discount, 0) AS coupon_discount,
    coalesce(r.redemptions, 0) AS redemptions,
    coalesce(l.active_campaigns, 0) AS active_campaigns,
    coalesce(w.trips, 0) > 0 AS is_active
FROM grid g
LEFT JOIN weekly w USING (household_id, week)
LEFT JOIN redeemed r USING (household_id, week)
LEFT JOIN live_campaigns l USING (household_id, week)
ORDER BY g.household_id, g.week;
