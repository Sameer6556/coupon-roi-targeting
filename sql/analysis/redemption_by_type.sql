-- Redemption by campaign type: how many targeted households redeem at least once,
-- and how concentrated redemptions are among the people who do redeem.

WITH per_type AS (
    SELECT
        c.campaign_type,
        count(*) AS campaigns,
        sum(c.households_targeted) AS households_targeted,
        sum(c.redeeming_households) AS redeeming_households,
        sum(c.redemptions) AS redemptions
    FROM dim_campaign c
    GROUP BY c.campaign_type
)
SELECT
    campaign_type,
    campaigns,
    households_targeted,
    redeeming_households,
    redemptions,
    redeeming_households / households_targeted AS household_redemption_rate,
    100.0 * redemptions / households_targeted AS redemptions_per_100_targeted
FROM per_type
ORDER BY campaign_type;
