-- RFM segmentation as of the last week of data, with NTILE scores and LAG to get each
-- household's longest gap between shopping weeks (a decent early sign of drifting away).

WITH last_week AS (SELECT max(week) AS w FROM dim_week),
active_weeks AS (
    SELECT household_id, week
    FROM hh_week
    WHERE is_active
),
gaps AS (
    SELECT
        household_id,
        week - lag(week) OVER (PARTITION BY household_id ORDER BY week) AS gap
    FROM active_weeks
),
base AS (
    SELECT
        h.household_id,
        (SELECT w FROM last_week) - max(h.week) FILTER (WHERE h.is_active) AS recency_weeks,
        sum(h.trips) AS frequency,
        sum(h.spend_xf) AS monetary
    FROM hh_week h
    GROUP BY h.household_id
),
scored AS (
    SELECT
        b.*,
        (SELECT max(gap) FROM gaps g WHERE g.household_id = b.household_id) AS longest_gap_weeks,
        ntile(5) OVER (ORDER BY recency_weeks DESC) AS r_score,  -- 5 = most recent
        ntile(5) OVER (ORDER BY frequency) AS f_score,
        ntile(5) OVER (ORDER BY monetary) AS m_score
    FROM base b
)
SELECT
    *,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'champions'
        WHEN r_score >= 3 AND f_score >= 3 THEN 'loyal'
        WHEN r_score <= 2 AND f_score >= 3 THEN 'at risk'
        WHEN r_score <= 2 THEN 'lapsed'
        ELSE 'occasional'
    END AS rfm_segment
FROM scored;
