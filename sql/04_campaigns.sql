-- Campaign tables.
--   dim_campaign          one row per campaign, with its window in retailer weeks
--   fct_campaign_exposure which households were sent which campaign
--   fct_redemptions       coupon redemptions with the retailer week they happened in
--
-- Campaign types (from the data documentation):
--   Type A  personalised booklets: each household gets a subset of coupons picked from
--           its own purchase history. This is targeted, NOT random, which is the whole
--           reason the impact analysis needs matching.
--   Type B/C  smaller campaigns where every recipient gets the same coupons.

CREATE OR REPLACE TABLE dim_campaign AS
WITH reach AS (
    SELECT campaign_id, count(DISTINCT household_id) AS households_targeted
    FROM raw_campaigns
    GROUP BY campaign_id
),
coupon_stats AS (
    SELECT
        campaign_id,
        count(DISTINCT coupon_upc) AS coupons,
        count(DISTINCT product_id) AS coupon_products
    FROM raw_coupons
    GROUP BY campaign_id
),
redemption_stats AS (
    SELECT
        campaign_id,
        count(*) AS redemptions,
        count(DISTINCT household_id) AS redeeming_households
    FROM raw_coupon_redemptions
    GROUP BY campaign_id
)
SELECT
    d.campaign_id,
    d.campaign_type,
    CAST(d.start_date AS DATE) AS start_date,
    CAST(d.end_date AS DATE) AS end_date,
    week_of(d.start_date) AS start_week,
    week_of(d.end_date) AS end_week,
    week_of(d.end_date) - week_of(d.start_date) + 1 AS duration_weeks,
    coalesce(r.households_targeted, 0) AS households_targeted,
    coalesce(c.coupons, 0) AS coupons,
    coalesce(c.coupon_products, 0) AS coupon_products,
    coalesce(x.redemptions, 0) AS redemptions,
    coalesce(x.redeeming_households, 0) AS redeeming_households,
    coalesce(x.redeeming_households, 0) / nullif(r.households_targeted, 0) AS redemption_rate
FROM raw_campaign_descriptions d
LEFT JOIN reach r USING (campaign_id)
LEFT JOIN coupon_stats c USING (campaign_id)
LEFT JOIN redemption_stats x USING (campaign_id)
ORDER BY d.start_date;

CREATE OR REPLACE TABLE fct_campaign_exposure AS
SELECT
    c.household_id,
    c.campaign_id,
    d.campaign_type,
    d.start_week,
    d.end_week
FROM raw_campaigns c
JOIN dim_campaign d USING (campaign_id);

CREATE OR REPLACE TABLE fct_redemptions AS
SELECT
    r.household_id,
    r.campaign_id,
    r.coupon_upc,
    CAST(r.redemption_date AS DATE) AS redemption_date,
    week_of(r.redemption_date) AS week
FROM raw_coupon_redemptions r;
