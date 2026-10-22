-- Are the 801 households with demographics representative of all 2,469?
-- (They are not. Anything built on demographics only describes heavier shoppers.)

SELECT
    has_demographics,
    count(*) AS households,
    median(total_spend_ex_fuel / 53.0) AS median_weekly_spend,
    avg(active_weeks) AS avg_active_weeks,
    avg(campaigns_received) AS avg_campaigns_received,
    avg(CAST(type_a_received > 0 AS INTEGER)) AS share_got_type_a,
    sum(total_spend_ex_fuel) / (SELECT sum(total_spend_ex_fuel) FROM dim_household) AS share_of_revenue
FROM dim_household
GROUP BY has_demographics
ORDER BY has_demographics DESC;
