-- dim_product: product attributes with explicit UNKNOWN buckets.
--
-- Cleaning notes:
--   * ~540 products have no category; they become 'UNKNOWN' rather than being dropped.
--   * ~4.8K transaction lines point at product ids that are not in the product table.
--     They get a placeholder row so joins never silently drop sales.
--   * FUEL is flagged because fuel quantities are in odd units (thousandths of a gallon)
--     and fuel prices move spend around for reasons coupons can't touch.

CREATE OR REPLACE TABLE dim_product AS
WITH known AS (
    SELECT
        product_id,
        manufacturer_id,
        coalesce(department, 'UNKNOWN') AS department,
        coalesce(brand, 'UNKNOWN') AS brand,
        coalesce(product_category, 'UNKNOWN') AS product_category,
        coalesce(product_type, 'UNKNOWN') AS product_type,
        package_size
    FROM raw_products
),
orphans AS (
    SELECT DISTINCT t.product_id
    FROM raw_transactions t
    ANTI JOIN known k USING (product_id)
)
SELECT
    product_id,
    manufacturer_id,
    department,
    brand,
    product_category,
    product_type,
    package_size,
    department = 'FUEL' AS is_fuel,
    false AS is_orphan
FROM known
UNION ALL
SELECT
    product_id,
    NULL AS manufacturer_id,
    'UNKNOWN' AS department,
    'UNKNOWN' AS brand,
    'UNKNOWN' AS product_category,
    'UNKNOWN' AS product_type,
    NULL AS package_size,
    false AS is_fuel,
    true AS is_orphan
FROM orphans;
