-- Which households are growing or shrinking their spend, and which departments drive it?
-- Compares average weekly spend (ex fuel) in weeks 2-26 vs 27-51: two 25-week halves,
-- so neither half gets the partial week 1 or the Christmas-shortened week 53.

WITH halves AS (
    SELECT
        household_id,
        avg(spend_xf) FILTER (WHERE week BETWEEN 2 AND 26) AS h1_weekly,
        avg(spend_xf) FILTER (WHERE week BETWEEN 27 AND 51) AS h2_weekly
    FROM hh_week
    GROUP BY household_id
)
SELECT
    household_id,
    h1_weekly,
    h2_weekly,
    CASE
        WHEN h1_weekly = 0 AND h2_weekly = 0 THEN 'inactive'
        WHEN h1_weekly = 0 THEN 'new'
        WHEN h2_weekly >= 1.25 * h1_weekly THEN 'growing'
        WHEN h2_weekly <= 0.75 * h1_weekly THEN 'declining'
        ELSE 'stable'
    END AS segment
FROM halves;
