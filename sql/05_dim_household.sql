-- dim_household: one row per household that ever shopped.
--
-- Only 801 of the 2,469 households have demographics, and they are not a random
-- subset (they shop more). has_demographics makes that easy to check and stops anyone
-- from quietly treating the 801 as representative.

CREATE OR REPLACE TABLE dim_household AS
WITH activity AS (
    SELECT
        household_id,
        min(week) AS first_week,
        max(week) AS last_week,
        count(DISTINCT week) AS active_weeks,
        count(DISTINCT basket_id) AS baskets,
        sum(sales_value) AS total_spend,
        sum(CASE WHEN is_fuel THEN 0 ELSE sales_value END) AS total_spend_ex_fuel
    FROM fct_transactions
    GROUP BY household_id
),
camps AS (
    SELECT
        household_id,
        count(*) AS campaigns_received,
        count(*) FILTER (WHERE campaign_type = 'Type A') AS type_a_received
    FROM fct_campaign_exposure
    GROUP BY household_id
),
redeemed AS (
    SELECT household_id, count(*) AS redemptions
    FROM fct_redemptions
    GROUP BY household_id
)
SELECT
    a.*,
    coalesce(c.campaigns_received, 0) AS campaigns_received,
    coalesce(c.type_a_received, 0) AS type_a_received,
    coalesce(r.redemptions, 0) AS redemptions,
    d.household_id IS NOT NULL AS has_demographics,
    d.age,
    d.income,
    d.home_ownership,
    d.marital_status,
    d.household_size,
    d.household_comp,
    d.kids_count
FROM activity a
LEFT JOIN camps c USING (household_id)
LEFT JOIN redeemed r USING (household_id)
LEFT JOIN raw_demographics d USING (household_id);
