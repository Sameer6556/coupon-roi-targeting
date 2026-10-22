-- Read-only views with friendly column names. These are the ONLY objects the
-- "ask the data" feature is allowed to query, and the app ships a slim copy of them.

CREATE OR REPLACE VIEW v_household_week AS
SELECT
    h.household_id,
    h.week,
    w.week_start,
    h.spend,
    h.spend_xf AS spend_ex_fuel,
    h.trips,
    h.departments AS departments_shopped,
    h.private_spend AS private_label_spend,
    h.promo_spend,
    h.discount_amount,
    h.redemptions AS coupon_redemptions,
    h.active_campaigns,
    h.is_active
FROM hh_week h
JOIN dim_week w USING (week);

CREATE OR REPLACE VIEW v_households AS
SELECT
    household_id,
    first_week,
    last_week,
    active_weeks,
    baskets,
    total_spend,
    total_spend_ex_fuel,
    campaigns_received,
    type_a_received,
    redemptions,
    has_demographics,
    age,
    income,
    home_ownership,
    marital_status,
    household_size,
    household_comp,
    kids_count
FROM dim_household;

CREATE OR REPLACE VIEW v_campaigns AS
SELECT
    campaign_id,
    campaign_type,
    start_date,
    end_date,
    start_week,
    end_week,
    duration_weeks,
    households_targeted,
    coupons,
    redemptions,
    redeeming_households,
    redemption_rate
FROM dim_campaign;

CREATE OR REPLACE VIEW v_campaign_exposure AS
SELECT household_id, campaign_id, campaign_type
FROM fct_campaign_exposure;

CREATE OR REPLACE VIEW v_department_week AS
SELECT
    department,
    week,
    sum(sales_value) AS spend,
    count(DISTINCT basket_id) AS baskets,
    count(DISTINCT household_id) AS households
FROM fct_transactions
GROUP BY department, week;

CREATE OR REPLACE VIEW v_redemptions AS
SELECT household_id, campaign_id, week, redemption_date
FROM fct_redemptions;
