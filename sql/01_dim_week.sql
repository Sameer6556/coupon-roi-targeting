-- dim_week: one row per retailer week.
--
-- Cleaning note: transaction timestamps are stored in UTC. Retailer weeks flip at
-- midnight US Eastern on Mondays, which is 05:00 UTC in winter but 04:00 UTC during
-- daylight saving time. A fixed "-5 hours" shift puts the first hour of 34 weeks into
-- the wrong day, so local time is a proper time zone conversion (to_local below).
-- Week 1 is a single Sunday (1 Jan 2017) and is left out of baselines elsewhere.
-- Week 53 has 6 trading days because stores were closed on Christmas Day.

CREATE OR REPLACE MACRO to_local(ts) AS timezone('America/New_York', timezone('UTC', ts));

CREATE OR REPLACE TABLE dim_week AS
WITH observed AS (
    SELECT
        week,
        min(CAST(to_local(transaction_timestamp) AS DATE)) AS first_date,
        max(CAST(to_local(transaction_timestamp) AS DATE)) AS last_date,
        count(DISTINCT basket_id) AS baskets,
        count(DISTINCT household_id) AS active_households,
        sum(sales_value) AS sales
    FROM raw_transactions
    GROUP BY week
),
anchor AS (
    -- the Monday that starts week 2 is the reference point for date -> week maths
    SELECT first_date AS week2_start FROM observed WHERE week = 2
)
SELECT
    o.week,
    a.week2_start + CAST((o.week - 2) * 7 AS INTEGER) AS week_start,
    a.week2_start + CAST((o.week - 2) * 7 + 6 AS INTEGER) AS week_end,
    o.first_date,
    o.last_date,
    date_diff('day', o.first_date, o.last_date) + 1 AS days_observed,
    o.baskets,
    o.active_households,
    o.sales
FROM observed o
CROSS JOIN anchor a
ORDER BY o.week;

-- date -> retailer week, valid before and after the data range too (campaigns start in 2016
-- and end in 2018). floor() on purpose: DuckDB's // truncates toward zero for negatives.
CREATE OR REPLACE MACRO week_of(d) AS
    2 + CAST(floor(date_diff('day', (SELECT week_start FROM dim_week WHERE week = 2), CAST(d AS DATE)) / 7.0) AS INTEGER);
