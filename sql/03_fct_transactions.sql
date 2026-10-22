-- fct_transactions: one row per line item, enriched with product attributes and
-- whether the item was on an in-store display or in the weekly mailer at that store
-- that week (from the 20.9M-row promotions table).
--
-- Cleaning notes:
--   * discounts are stored as positive amounts in this release (the original dunnhumby
--     release stores them as negatives), so "discount_amount" is a plain sum.
--   * quantity = 0 rows (~8.9K) are kept: they carry sales/discount values.
--   * display_location / mailer_location use '0' for "not promoted".

CREATE OR REPLACE TABLE fct_transactions AS
WITH promo AS (
    SELECT
        product_id,
        store_id,
        week,
        bool_or(display_location <> '0') AS on_display,
        bool_or(mailer_location <> '0') AS in_mailer
    FROM raw_promotions
    GROUP BY product_id, store_id, week
)
SELECT
    t.household_id,
    t.basket_id,
    t.store_id,
    t.product_id,
    t.week,
    CAST(to_local(t.transaction_timestamp) AS TIMESTAMP) AS ts_local,
    CAST(to_local(t.transaction_timestamp) AS DATE) AS date_local,
    t.quantity,
    t.sales_value,
    t.retail_disc,
    t.coupon_disc,
    t.coupon_match_disc,
    t.retail_disc + t.coupon_disc + t.coupon_match_disc AS discount_amount,
    p.department,
    p.brand,
    p.product_category,
    p.is_fuel,
    coalesce(pr.on_display, false) AS on_display,
    coalesce(pr.in_mailer, false) AS in_mailer
FROM raw_transactions t
JOIN dim_product p USING (product_id)
LEFT JOIN promo pr USING (product_id, store_id, week);
