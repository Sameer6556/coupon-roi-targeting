-- How concentrated is revenue? Households ranked by spend, cumulative share via
-- window functions, plus a 10-way NTILE for the decile table.

WITH spend AS (
    SELECT household_id, total_spend_ex_fuel AS spend
    FROM dim_household
),
ranked AS (
    SELECT
        household_id,
        spend,
        row_number() OVER (ORDER BY spend DESC) AS rank_desc,
        count(*) OVER () AS n_households,
        sum(spend) OVER (ORDER BY spend DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_spend,
        sum(spend) OVER () AS total_spend,
        ntile(10) OVER (ORDER BY spend DESC) AS decile
    FROM spend
)
SELECT
    household_id,
    spend,
    decile,
    rank_desc / n_households AS household_share,
    cum_spend / total_spend AS cum_spend_share
FROM ranked
ORDER BY rank_desc;
