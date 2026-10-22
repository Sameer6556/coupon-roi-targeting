-- Lapse label for a cutoff: did the household's weekly spend over the next $horizon
-- weeks fall below $drop x its own average over the previous $lookback weeks?
--
-- Kept in its own file, separate from the features, so it is obvious which query is
-- allowed to look past the cutoff (this one) and which is not (household_features.sql).

WITH baseline AS (
    SELECT household_id, avg(spend_xf) AS base_weekly
    FROM hh_week
    WHERE week <= $cutoff AND week > $cutoff - $lookback
    GROUP BY household_id
),
future AS (
    SELECT household_id, avg(spend_xf) AS future_weekly, sum(trips) AS future_trips
    FROM hh_week
    WHERE week > $cutoff AND week <= $cutoff + $horizon
    GROUP BY household_id
)
SELECT
    b.household_id,
    $cutoff AS cutoff,
    f.future_weekly,
    f.future_trips,
    CAST(f.future_weekly < $drop * b.base_weekly AS INTEGER) AS lapse,
    CAST(f.future_trips = 0 AS INTEGER) AS full_lapse
FROM baseline b
JOIN future f USING (household_id);
