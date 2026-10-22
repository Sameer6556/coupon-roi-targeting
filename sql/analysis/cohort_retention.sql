-- Cohort retention: households grouped by the 4-week block of their first purchase,
-- then the share of each cohort that shopped in each later 4-week block.

WITH first_seen AS (
    SELECT household_id, min(week) FILTER (WHERE is_active) AS first_week
    FROM hh_week
    GROUP BY household_id
),
blocks AS (
    SELECT
        h.household_id,
        (f.first_week - 1) // 4 AS cohort_block,
        (h.week - 1) // 4 AS block,
        max(CAST(h.is_active AS INTEGER)) AS active_in_block
    FROM hh_week h
    JOIN first_seen f USING (household_id)
    WHERE h.week >= f.first_week AND h.week <= 52  -- 13 full blocks
    GROUP BY 1, 2, 3
)
SELECT
    cohort_block,
    block - cohort_block AS blocks_since_first,
    count(*) AS households,
    avg(active_in_block) AS retention
FROM blocks
GROUP BY cohort_block, blocks_since_first
ORDER BY cohort_block, blocks_since_first;
