-- Change in weekly spend by department between the two halves of the year,
-- split by the household growth segment. Needs the temp table `segments`
-- (output of growth_segments.sql) to exist.

WITH dept_halves AS (
    SELECT
        t.household_id,
        t.department,
        sum(t.sales_value) FILTER (WHERE t.week BETWEEN 2 AND 26) / 25.0 AS h1_weekly,
        sum(t.sales_value) FILTER (WHERE t.week BETWEEN 27 AND 51) / 25.0 AS h2_weekly
    FROM fct_transactions t
    WHERE NOT t.is_fuel
    GROUP BY t.household_id, t.department
)
SELECT
    s.segment,
    d.department,
    count(DISTINCT s.household_id) AS households,
    sum(coalesce(d.h2_weekly, 0) - coalesce(d.h1_weekly, 0)) AS change_weekly_total,
    sum(coalesce(d.h2_weekly, 0) - coalesce(d.h1_weekly, 0)) / count(DISTINCT s.household_id) AS change_weekly_per_hh
FROM dept_halves d
JOIN segments s USING (household_id)
WHERE s.segment IN ('growing', 'declining')
GROUP BY s.segment, d.department;
