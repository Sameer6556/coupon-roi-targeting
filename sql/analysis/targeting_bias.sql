-- Who gets the big Type A coupon booklets? Average weekly spend in the 8 weeks
-- BEFORE each wave starts, recipients vs everyone else who shopped in that window.
-- If recipients already spent more before the booklet arrived, comparing them to
-- non-recipients afterwards measures the targeting rule, not the coupon.

WITH waves AS (
    SELECT campaign_id, start_week, greatest(start_week - 8, 2) AS pre_start
    FROM dim_campaign
    WHERE campaign_type = 'Type A' AND start_week >= 6
),
pre AS (
    SELECT
        w.campaign_id,
        h.household_id,
        avg(h.spend_xf) AS pre_weekly_spend,
        sum(h.trips) AS pre_trips
    FROM waves w
    JOIN hh_week h ON h.week >= w.pre_start AND h.week < w.start_week
    GROUP BY w.campaign_id, h.household_id
)
SELECT
    p.campaign_id,
    e.household_id IS NOT NULL AS recipient,
    count(*) AS households,
    avg(p.pre_weekly_spend) AS avg_pre_weekly_spend,
    median(p.pre_weekly_spend) AS median_pre_weekly_spend
FROM pre p
LEFT JOIN fct_campaign_exposure e
       ON e.household_id = p.household_id AND e.campaign_id = p.campaign_id
WHERE p.pre_trips > 0
GROUP BY p.campaign_id, recipient
ORDER BY p.campaign_id, recipient;
