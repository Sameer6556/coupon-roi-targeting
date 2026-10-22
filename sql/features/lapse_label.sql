-- Lapse label for a cutoff: does a REGULAR shopper go quiet over the next $horizon weeks?
--
--   regular  = shopped in at least $min_regular of the last $lookback weeks
--   lapse    = shops in at most $quiet_max of the next $horizon weeks
--
-- Why this definition: in grocery almost nobody "cancels", and a household that shops
-- once a month going on shopping once a month isn't news. A regular who suddenly goes
-- quiet is the signal a CRM team can act on. Spend-halving and full lapse are kept as
-- secondary labels for comparison.
--
-- Kept in its own file, separate from the features, so it is obvious which query is
-- allowed to look past the cutoff (this one) and which is not (household_features.sql).

WITH baseline AS (
    SELECT
        household_id,
        avg(spend_xf) AS base_weekly,
        sum(CAST(is_active AS INTEGER)) AS base_active_weeks
    FROM hh_week
    WHERE week <= $cutoff AND week > $cutoff - $lookback
    GROUP BY household_id
),
future AS (
    SELECT
        household_id,
        avg(spend_xf) AS future_weekly,
        sum(trips) AS future_trips,
        sum(CAST(is_active AS INTEGER)) AS future_active_weeks
    FROM hh_week
    WHERE week > $cutoff AND week <= $cutoff + $horizon
    GROUP BY household_id
)
SELECT
    b.household_id,
    $cutoff AS cutoff,
    f.future_weekly,
    f.future_active_weeks,
    CAST(f.future_active_weeks <= $quiet_max AS INTEGER) AS lapse,
    CAST(f.future_weekly < $drop * b.base_weekly AS INTEGER) AS spend_halved,
    CAST(f.future_trips = 0 AS INTEGER) AS full_lapse
FROM baseline b
JOIN future f USING (household_id)
WHERE b.base_active_weeks >= $min_regular;
