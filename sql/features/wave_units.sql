-- One row per household for one campaign wave: baseline behaviour in the pre window,
-- what else they were sent around the same time, and spend during the campaign.
-- This is the "campaign pre/post window join" behind the impact analysis.
--
-- Parameters: $campaign_id, $pre_start, $start (first campaign week), $end (last
-- campaign week, clipped to the data), $first_week (first full week of data).
-- Population: households that shopped at least once in the pre window.

WITH pre AS (
    SELECT
        household_id,
        avg(spend_xf) AS pre_spend,
        avg(trips) AS pre_trips,
        avg(CAST(is_active AS INTEGER)) AS pre_active_share,
        avg(departments) AS pre_departments,
        coalesce(sum(private_spend) / nullif(sum(spend_xf), 0), 0) AS pre_private_share,
        coalesce(sum(promo_spend) / nullif(sum(spend_xf), 0), 0) AS pre_promo_share,
        avg(CAST(coupon_discount > 0 AS INTEGER)) AS pre_coupon_use,
        coalesce(regr_slope(spend_xf, week), 0) AS pre_trend,
        sum(trips) AS pre_trips_total
    FROM hh_week
    WHERE week BETWEEN $pre_start AND $start - 1
    GROUP BY household_id
),
prior AS (
    -- the weeks BEFORE the baseline window. Segments are defined on these so that
    -- picking a segment never selects on noise in the baseline itself.
    SELECT
        household_id,
        avg(spend_xf) AS prior_spend,
        avg(CAST(coupon_discount > 0 AS INTEGER)) AS prior_coupon_use,
        count(*) AS prior_weeks
    FROM hh_week
    WHERE week BETWEEN $first_week AND $pre_start - 1
    GROUP BY household_id
),
post AS (
    SELECT household_id, avg(spend_xf) AS post_spend, avg(trips) AS post_trips
    FROM hh_week
    WHERE week BETWEEN $start AND $end
    GROUP BY household_id
),
exposure AS (
    SELECT
        e.household_id,
        max(CAST(e.campaign_id = $campaign_id AS INTEGER)) AS treated,
        count(*) FILTER (
            WHERE e.campaign_type = 'Type A' AND e.campaign_id <> $campaign_id AND e.start_week < $start
        ) AS prior_waves,
        count(*) FILTER (
            WHERE e.campaign_id <> $campaign_id AND e.start_week <= $start - 1 AND e.end_week >= $pre_start
        ) AS co_exposure_pre,
        count(*) FILTER (
            WHERE e.campaign_id <> $campaign_id AND e.start_week <= $end AND e.end_week >= $start
        ) AS co_exposure_post
    FROM fct_campaign_exposure e
    GROUP BY e.household_id
)
SELECT
    p.household_id,
    coalesce(x.treated, 0) AS treated,
    p.pre_spend,
    p.pre_trips,
    p.pre_active_share,
    p.pre_departments,
    p.pre_private_share,
    p.pre_promo_share,
    p.pre_coupon_use,
    p.pre_trend,
    coalesce(pr.prior_spend, p.pre_spend) AS prior_spend,
    pr.prior_coupon_use,
    coalesce(pr.prior_weeks, 0) AS prior_weeks,
    coalesce(x.prior_waves, 0) AS prior_waves,
    coalesce(x.co_exposure_pre, 0) AS co_exposure_pre,
    coalesce(x.co_exposure_post, 0) AS co_exposure_post,
    coalesce(po.post_spend, 0) AS post_spend,
    coalesce(po.post_trips, 0) AS post_trips,
    coalesce(po.post_spend, 0) - p.pre_spend AS dy,
    coalesce(po.post_trips, 0) - p.pre_trips AS dy_trips
FROM pre p
LEFT JOIN prior pr USING (household_id)
LEFT JOIN post po USING (household_id)
LEFT JOIN exposure x USING (household_id)
WHERE p.pre_trips_total > 0
ORDER BY p.household_id;
