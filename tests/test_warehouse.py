"""Tests for the SQL feature layer.

A comparable-sales feature is the easiest place in this whole project to leak the
target, because the feature *is* other houses' prices. Two rules keep it honest, and
both are asserted here against a hand-built frame where the right answer is known by
construction rather than by re-running the query:

1. a comparable must have already sold, and
2. its price must come from a training row.

The last two tests check that the first two are not vacuous -- that the query really
would pick up a future sale or a test-row price if the predicates were removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ames.warehouse import (COMPARABLE_FEATURES, attach_comparables, comparable_sales,
                            connect)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal `properties` frame: only what sql/01_sales_base.sql selects."""
    base = {"Sale Condition": "Normal", "is_outlier_grlivarea": False,
            "Neighborhood": "NAmes", "Gr Liv Area": 1_500,
            "Latitude": 42.030, "Longitude": -93.620, "Yr Sold": 2008,
            "is_training": True}
    return pd.DataFrame([{**base, **r} for r in rows])


# Four houses on the same block. The subject sells last; one neighbour sells after it.
NEARBY = 0.002          # ~0.14 miles at this latitude


@pytest.fixture
def block() -> pd.DataFrame:
    return _frame([
        {"PID": 1, "sale_time": 2006.0, "SalePrice": 150_000, "Latitude": 42.030},
        {"PID": 2, "sale_time": 2007.0, "SalePrice": 170_000,
         "Latitude": 42.030 + NEARBY},
        {"PID": 3, "sale_time": 2009.0, "SalePrice": 900_000,      # sells AFTER
         "Latitude": 42.030 + 2 * NEARBY},
        {"PID": 4, "sale_time": 2008.0, "SalePrice": 200_000,      # the subject
         "Latitude": 42.030 + 3 * NEARBY},
    ])


# --------------------------------------------------------------------------
# The two rules
# --------------------------------------------------------------------------

def test_a_later_sale_is_never_a_comparable(block):
    """PID 3 sells in 2009 for $900k. The subject sells in 2008. A model valuing the
    subject cannot know what the house two doors down will fetch next year."""
    comps = comparable_sales(block).set_index("PID")
    subject = comps.loc[4]
    assert subject["comp_n"] == 2                       # PIDs 1 and 2 only
    # $900k / 1,500 sq ft = $600. If it had leaked, the median would move sharply.
    assert subject["comp_ppsf_median"] == pytest.approx((100.0 + 113.3333) / 2, abs=0.1)


def test_a_test_row_price_is_never_a_comparable(block):
    """The time cutoff alone is not enough. Under a shuffled split an *earlier* sale
    can sit in the test set, and using its price to build a feature for another test
    row puts the target into the design matrix by a side door."""
    held_out = block.copy()
    held_out.loc[held_out["PID"] == 2, "is_training"] = False
    comps = comparable_sales(held_out).set_index("PID")
    assert comps.loc[4, "comp_n"] == 1                  # PID 1 survives, PID 2 does not
    assert comps.loc[4, "comp_ppsf_median"] == pytest.approx(100.0)


def test_the_time_rule_is_not_vacuous(block):
    """If the query did not filter on sale_time, PID 3's $600/sqft would arrive. This
    asserts the fixture can actually detect that -- a guard that always passes because
    the data could never trip it is worse than no guard."""
    rewritten = block.copy()
    rewritten.loc[rewritten["PID"] == 3, "sale_time"] = 2007.5     # now sells *before*
    comps = comparable_sales(rewritten).set_index("PID")
    assert comps.loc[4, "comp_n"] == 3
    assert comps.loc[4, "comp_ppsf_median"] > 110.0                # the $600 pulled it up


def test_the_training_rule_is_not_vacuous(block):
    """Same check for the is_training predicate."""
    comps_all_train = comparable_sales(block).set_index("PID")
    held_out = block.copy()
    held_out["is_training"] = False
    comps_none_train = comparable_sales(held_out).set_index("PID")
    assert comps_all_train.loc[4, "comp_n"] == 2
    assert comps_none_train["comp_n"].sum() == 0


def test_a_property_is_never_its_own_comparable(block):
    comps = comparable_sales(block).set_index("PID")
    # PID 1 sells first, so it has nothing prior -- including itself.
    assert comps.loc[1, "comp_n"] == 0


# --------------------------------------------------------------------------
# Shape and behaviour
# --------------------------------------------------------------------------

def test_properties_with_no_comparable_are_kept_not_dropped(block):
    """The first sale of the panel has nothing to compare to. That is a real state in
    production, so the model has to see it rather than have it quietly imputed."""
    comps = comparable_sales(block)
    assert len(comps) == len(block)
    assert comps.loc[comps["PID"] == 1, "comp_n"].iloc[0] == 0
    assert comps.loc[comps["PID"] == 1, "comp_ppsf_median"].isna().all()


def test_every_documented_feature_is_produced(block):
    comps = comparable_sales(block)
    assert set(COMPARABLE_FEATURES) <= set(comps.columns)


def test_distant_sales_are_excluded(block):
    far = block.copy()
    far.loc[far["PID"].isin([1, 2, 3]), "Latitude"] = 42.30      # ~18 miles north
    assert comparable_sales(far).set_index("PID").loc[4, "comp_n"] == 0


def test_very_different_sizes_are_excluded(block):
    odd = block.copy()
    odd.loc[odd["PID"].isin([1, 2]), "Gr Liv Area"] = 6_000      # 4x the subject
    assert comparable_sales(odd).set_index("PID").loc[4, "comp_n"] == 0


def test_at_most_ten_comparables_are_kept():
    rows = [{"PID": i, "sale_time": 2006.0 + i * 0.01, "SalePrice": 150_000,
             "Latitude": 42.030 + i * 0.0002} for i in range(1, 26)]
    rows.append({"PID": 99, "sale_time": 2009.0, "SalePrice": 200_000,
                 "Latitude": 42.030})
    comps = comparable_sales(_frame(rows)).set_index("PID")
    assert comps.loc[99, "comp_n"] == 10


def test_attach_preserves_the_row_count_and_key_dtype(block):
    out = attach_comparables(block)
    assert len(out) == len(block)
    assert out["PID"].dtype == block["PID"].dtype
    assert set(COMPARABLE_FEATURES) <= set(out.columns)


def test_a_missing_training_flag_is_refused_not_defaulted(block):
    """Defaulting to "everything is training" would leak in every caller that forgot
    to set the flag, and would leak silently."""
    with pytest.raises(ValueError, match="is_training"):
        connect(block.drop(columns="is_training"))


def test_the_sql_distance_agrees_with_the_python_one(block):
    """sql/02 uses an equirectangular approximation so the predicate stays pushable;
    ames.features uses the haversine. Over one city they must not disagree."""
    from ames.features import haversine_miles

    con = connect(block)
    try:
        pairs = con.execute(
            "SELECT pid, miles_away FROM comparable_pairs ORDER BY pid").df()
    finally:
        con.close()
    assert not pairs.empty
    subject = block[block["PID"] == 4].iloc[0]
    others = block[block["PID"].isin([1, 2])]
    expected = np.sort(haversine_miles(subject["Latitude"], subject["Longitude"],
                                       others["Latitude"], others["Longitude"]))
    got = np.sort(pairs.loc[pairs["pid"] == "4", "miles_away"].to_numpy())
    assert got == pytest.approx(expected, abs=1e-3)     # under ~5 feet
