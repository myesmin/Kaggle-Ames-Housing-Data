"""The feature layer, in SQL.

Comparable sales (what nearby houses fetched before this one sold) is the feature
notebook 04's Moran's I says is missing. It is a SQL self-join on the sales table with a
distance predicate, a strict time cutoff, and a window function keeping the nearest ten.
As a pandas loop over 2,412 properties it would be a slow nested scan with leakage
harder to see.

DuckDB runs it in-process over the existing parquet files: no server, no credentials,
and ``make data`` still builds everything from a clean clone. The warehouse is a view
over files on disk, not a second copy.

Two leakage rules, enforced by the query in ``sql/02_comparable_sales.sql``:

* a comparable must have **already sold** (``c.sale_time < s.sale_time``), and
* its price must come from a **training row** (``c.is_training``).

A time cutoff alone is not enough: under a shuffled split an earlier sale can be in
the test set, and using its price as a feature for another test row leaks the target.
``tests/test_warehouse.py`` asserts both rules and checks the assertions are not
vacuous.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED, ROOT

SQL_DIR = ROOT / "sql"

#: Columns the comparable-sales view produces, in the order the SQL emits them.
COMPARABLE_FEATURES = (
    "comp_n", "comp_ppsf_median", "comp_ppsf_p25", "comp_ppsf_p75",
    "comp_price_median", "comp_miles_mean", "comp_miles_min", "comp_years_stale",
)


def connect(properties: pd.DataFrame):
    """An in-process DuckDB connection with ``properties`` registered and views built.

    ``properties`` must carry an ``is_training`` flag.  It is a required column rather
    than a default because the flag decides which prices may become features, and a
    default of "everything is training" would silently leak in every caller that
    forgot to set it.
    """
    import duckdb

    if "is_training" not in properties.columns:
        raise ValueError(
            "properties needs an `is_training` column: comparable prices may only "
            "come from training rows. See sql/02_comparable_sales.sql.")

    con = duckdb.connect(":memory:")
    con.register("properties", properties)
    for path in sorted(SQL_DIR.glob("*.sql")):
        con.execute(path.read_text())
    return con


def comparable_sales(properties: pd.DataFrame) -> pd.DataFrame:
    """One row per property with its comparable-sales features.

    Properties with no qualifying comparable (the earliest sales, isolated parcels) are
    returned with ``comp_n = 0`` and null features rather than dropped. This happens in
    production (the first sale of the year has nothing prior), so the model must handle
    it; imputing a neighbourhood median would hide how often it occurs.
    """
    con = connect(properties)
    try:
        comps = con.execute("SELECT * FROM comparable_sales").df()
    finally:
        con.close()

    # SQL hands PID back as text; the frame keys on whatever dtype it arrived with.
    # Joining across the two silently produces an all-null result in pandas <2 and an
    # exception in pandas 2, so the cast is explicit and the original dtype is kept.
    out = properties[["PID"]].copy()
    out["_key"] = out["PID"].astype(str)
    comps["_key"] = comps.pop("pid").astype(str)
    out = out.merge(comps, on="_key", how="left").drop(columns="_key")
    out["comp_n"] = out["comp_n"].fillna(0).astype(int)
    return out


def attach_comparables(properties: pd.DataFrame) -> pd.DataFrame:
    """``properties`` with the comparable-sales columns joined on."""
    comps = comparable_sales(properties)
    return properties.merge(comps, on="PID", how="left", validate="one_to_one")
