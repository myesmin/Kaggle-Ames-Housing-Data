-- Comparable-sales features: what did nearby houses fetch, before this one sold?
--
-- This is how a commercial AVM gets its location signal, and it is the feature
-- notebook 04's Moran's I says is missing: valuation errors cluster in space, so
-- `Neighborhood` has not absorbed location.
--
-- Two rules make it honest, and both are enforced here rather than trusted:
--
--   c.sale_time < s.sale_time    A comparable must have *already sold*. A model asked
--                                to value a house in March 2008 cannot know what the
--                                house next door fetched in September.
--
--   c.is_training                The comparable's price must come from a training row.
--                                The time cutoff alone is not enough: in a shuffled
--                                split an earlier sale can sit in the test set, and
--                                using its price to predict another test row leaks the
--                                target into the feature. Production would use every
--                                recorded prior sale; an honest evaluation may not.
--
-- Distance is the equirectangular approximation, which agrees with the haversine in
-- ames.features to under a foot at Ames's latitude over a 2-mile radius, and unlike
-- the haversine it stays a join predicate the optimiser can push down.

CREATE OR REPLACE VIEW comparable_pairs AS
WITH pairs AS (
    SELECT
        s.pid                                    AS pid,
        c.price_per_sqft                         AS comp_ppsf,
        c.sale_price                             AS comp_price,
        s.sale_time - c.sale_time                AS months_stale,
        SQRT(
            POW((c.lat - s.lat) * 69.0545, 2) +
            POW((c.lon - s.lon) * 69.0545 * COS(RADIANS(s.lat)), 2)
        )                                        AS miles_away,
        ABS(LN(c.living_area / s.living_area))   AS size_gap
    FROM sales_base s
    JOIN sales_base c
      ON  c.pid <> s.pid
      AND c.sale_time < s.sale_time              -- strictly prior: no future prices
      AND c.is_training                          -- and no test-row prices, ever
)
SELECT *
FROM pairs
WHERE miles_away <= 1.5
  AND size_gap   <= 0.35;                        -- within ~+/-42% of subject floor area

-- Rank each subject's candidates the way an appraiser would -- nearest first, then
-- most similar in size, then most recent -- and keep the closest ten.
CREATE OR REPLACE VIEW comparable_sales AS
SELECT
    pid,
    COUNT(*)                                     AS comp_n,
    MEDIAN(comp_ppsf)                            AS comp_ppsf_median,
    QUANTILE_CONT(comp_ppsf, 0.25)               AS comp_ppsf_p25,
    QUANTILE_CONT(comp_ppsf, 0.75)               AS comp_ppsf_p75,
    MEDIAN(comp_price)                           AS comp_price_median,
    AVG(miles_away)                              AS comp_miles_mean,
    MIN(miles_away)                              AS comp_miles_min,
    AVG(months_stale)                            AS comp_years_stale
FROM (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY pid
               ORDER BY miles_away, size_gap, months_stale
           ) AS rn
    FROM comparable_pairs
)
WHERE rn <= 10
GROUP BY pid;
