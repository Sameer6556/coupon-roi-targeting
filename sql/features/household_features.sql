-- Point-in-time household features as of the END of week $cutoff.
--
-- Every row of every input is filtered to week <= $cutoff. That single rule is what
-- keeps the churn model honest, and tests/test_leakage.py proves it by scrambling
-- all data after the cutoff and checking that these features do not move.
--
-- Parameters: $cutoff (last week we are allowed to see), $lookback (window length).
-- Population: households that shopped at least once in the last 8 weeks.

WITH panel AS (
    SELECT *
    FROM hh_week
    WHERE week <= $cutoff
),
win AS (
    SELECT
        household_id,
        avg(spend_xf) FILTER (WHERE week > $cutoff - 4) AS spend_4w,
        avg(spend_xf) FILTER (WHERE week > $cutoff - 8) AS spend_8w,
        avg(spend_xf) FILTER (WHERE week > $cutoff - $lookback) AS spend_16w,
        sum(trips) FILTER (WHERE week > $cutoff - 4) AS trips_4w,
        sum(trips) FILTER (WHERE week > $cutoff - 8) AS trips_8w,
        sum(trips) FILTER (WHERE week > $cutoff - $lookback) AS trips_16w,
        sum(CAST(is_active AS INTEGER)) FILTER (WHERE week > $cutoff - 8) AS active_weeks_8w,
        sum(CAST(is_active AS INTEGER)) FILTER (WHERE week > $cutoff - $lookback) AS active_weeks_16w,
        stddev_samp(spend_xf) FILTER (WHERE week > $cutoff - $lookback) AS spend_sd_16w,
        regr_slope(spend_xf, week) FILTER (WHERE week > $cutoff - $lookback) AS spend_slope_16w,
        sum(private_spend) FILTER (WHERE week > $cutoff - $lookback) AS private_spend_16w,
        sum(promo_spend) FILTER (WHERE week > $cutoff - $lookback) AS promo_spend_16w,
        sum(spend_xf) FILTER (WHERE week > $cutoff - $lookback) AS spend_xf_sum_16w,
        sum(spend) FILTER (WHERE week > $cutoff - $lookback) AS spend_sum_16w,
        sum(discount_amount) FILTER (WHERE week > $cutoff - $lookback) AS discount_16w,
        sum(redemptions) FILTER (WHERE week > $cutoff - $lookback) AS redemptions_16w,
        sum(redemptions) AS redemptions_to_date,
        max(week) FILTER (WHERE is_active) AS last_active_week,
        min(week) FILTER (WHERE is_active) AS first_active_week,
        avg(spend_xf) FILTER (WHERE week >= 2) AS spend_to_date
    FROM panel
    GROUP BY household_id
),
breadth AS (
    SELECT
        household_id,
        count(DISTINCT department) AS departments_16w,
        count(DISTINCT product_category) AS categories_16w,
        count(DISTINCT store_id) AS stores_16w
    FROM fct_transactions
    WHERE week <= $cutoff AND week > $cutoff - $lookback AND NOT is_fuel
    GROUP BY household_id
),
camps AS (
    SELECT
        household_id,
        count(*) FILTER (WHERE start_week > $cutoff - $lookback) AS campaigns_16w,
        count(*) FILTER (WHERE campaign_type = 'Type A') AS type_a_to_date
    FROM fct_campaign_exposure
    WHERE start_week <= $cutoff
    GROUP BY household_id
)
SELECT
    w.household_id,
    $cutoff AS cutoff,
    $cutoff - w.last_active_week AS recency_weeks,
    $cutoff - w.first_active_week + 1 AS tenure_weeks,
    w.spend_4w,
    w.spend_8w,
    w.spend_16w,
    w.spend_to_date,
    w.trips_4w,
    w.trips_8w,
    w.trips_16w,
    w.active_weeks_8w,
    w.active_weeks_16w,
    w.spend_xf_sum_16w / nullif(w.trips_16w, 0) AS basket_value_16w,
    w.spend_4w / nullif(w.spend_16w, 0) AS momentum_4v16,
    w.spend_8w / nullif(w.spend_to_date, 0) AS momentum_8vall,
    coalesce(w.spend_sd_16w / nullif(w.spend_16w, 0), 0) AS spend_cv_16w,
    coalesce(w.spend_slope_16w / nullif(w.spend_16w, 0), 0) AS spend_slope_rel_16w,
    coalesce(w.private_spend_16w / nullif(w.spend_xf_sum_16w, 0), 0) AS private_share_16w,
    coalesce(w.promo_spend_16w / nullif(w.spend_xf_sum_16w, 0), 0) AS promo_share_16w,
    coalesce(w.discount_16w / nullif(w.spend_sum_16w + w.discount_16w, 0), 0) AS discount_share_16w,
    CAST(w.spend_sum_16w - w.spend_xf_sum_16w > 0 AS INTEGER) AS buys_fuel_16w,
    w.redemptions_16w,
    w.redemptions_to_date,
    coalesce(b.departments_16w, 0) AS departments_16w,
    coalesce(b.categories_16w, 0) AS categories_16w,
    coalesce(b.stores_16w, 0) AS stores_16w,
    coalesce(c.campaigns_16w, 0) AS campaigns_16w,
    coalesce(c.type_a_to_date, 0) AS type_a_to_date
FROM win w
LEFT JOIN breadth b USING (household_id)
LEFT JOIN camps c USING (household_id)
WHERE w.active_weeks_8w >= 1
ORDER BY w.household_id;
