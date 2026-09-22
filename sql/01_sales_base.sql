-- The sale-level base view.
--
-- One row per recorded arm's-length sale, carrying only what comparable selection
-- needs: where it is, when it sold, how big it is, and what it fetched.  Everything
-- downstream joins back to `properties` on PID.
--
-- `sale_time` is decimal years (2006.0 = January 2006) so the strict inequality in
-- 02_comparable_sales.sql orders sales without date arithmetic in every predicate.

CREATE OR REPLACE VIEW sales_base AS
SELECT
    CAST(PID AS VARCHAR)                         AS pid,
    "Neighborhood"                               AS neighborhood,
    "Latitude"                                   AS lat,
    "Longitude"                                  AS lon,
    "Gr Liv Area"                                AS living_area,
    "Yr Sold"                                    AS year_sold,
    sale_time,
    "SalePrice"                                  AS sale_price,
    "SalePrice" / NULLIF("Gr Liv Area", 0)       AS price_per_sqft,
    is_training
FROM properties
WHERE "Sale Condition" = 'Normal'
  AND NOT is_outlier_grlivarea
  AND "Gr Liv Area" > 0
  AND "Latitude" IS NOT NULL;
