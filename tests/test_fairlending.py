"""Tests for the fair-lending analysis.

The arithmetic here reaches conclusions about people, so the primitives are checked
against closed forms and hand-worked cases rather than against themselves.  Most of
this file needs no data at all: the interval, the E-value and the DTI mapping are pure
functions, so they run in CI on a clean clone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ames.config import EXTERNAL
from ames.fairlending import (DENIAL_REASONS, DTI_BANDS, NOT_A_GROUP, POOLED_LABEL,
                              _wilson, adjusted_disparity, decisioned, denial_rates,
                              denial_reason_mix, dti_band, e_value,
                              e_value_for_interval, ltv_distribution)


# --------------------------------------------------------------------------
# DTI banding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (19, "<20%"), (19.9, "<20%"), (20, "20%-<30%"), (29.9, "20%-<30%"),
    (30, "30%-<36%"), (35.9, "30%-<36%"), (36, "36%-<40%"), (39, "36%-<40%"),
    (40, "40%-<45%"), (44, "40%-<45%"), (45, "45%-<50%"), (49, "45%-<50%"),
    (50, "50%-60%"), (60, "50%-60%"), (61, ">60%"),
])
def test_numeric_dti_lands_in_the_right_band(raw, expected):
    assert dti_band(raw) == expected


@pytest.mark.parametrize("raw", DTI_BANDS)
def test_an_already_banded_dti_passes_through(raw):
    assert dti_band(raw) == raw


@pytest.mark.parametrize("raw", ["Exempt", "NA", "", None, np.nan, "garbage"])
def test_sentinels_become_missing_not_zero(raw):
    """HMDA puts "Exempt" in numeric columns. Reading that as 0 would put those
    applicants in the lowest-risk DTI band, which is the opposite of the truth."""
    assert pd.isna(dti_band(raw))


# --------------------------------------------------------------------------
# The interval
# --------------------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = _wilson(43, 174)
    assert lo < 43 / 174 < hi


def test_wilson_stays_inside_zero_and_one_at_the_extremes():
    """The normal approximation returns negative lower bounds for small p. That is
    the reason this is not the normal approximation."""
    for successes, n in [(0, 20), (20, 20), (1, 500), (2, 13)]:
        lo, hi = _wilson(successes, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_the_interval_narrows_as_the_sample_grows():
    width = lambda k, n: np.diff(_wilson(k, n))[0]      # noqa: E731
    assert width(25, 100) > width(250, 1_000) > width(2_500, 10_000)


def test_an_empty_group_gives_no_interval():
    assert all(np.isnan(v) for v in _wilson(0, 0))


# --------------------------------------------------------------------------
# The E-value
# --------------------------------------------------------------------------

def test_e_value_of_the_null_is_one():
    assert e_value(1.0) == 1.0


def test_e_value_matches_the_published_closed_form():
    """VanderWeele & Ding (2017): E = RR + sqrt(RR * (RR - 1))."""
    for rr in (1.5, 2.0, 3.0, 5.0):
        assert e_value(rr) == pytest.approx(rr + np.sqrt(rr * (rr - 1)))


def test_e_value_is_symmetric_under_reciprocal():
    """A halving and a doubling are equally hard to explain away."""
    assert e_value(0.5) == pytest.approx(e_value(2.0))


def test_e_value_grows_with_the_association():
    assert e_value(1.2) < e_value(2.0) < e_value(4.0)


def test_an_interval_containing_the_null_needs_no_confounder():
    """The trap this guards: mapping e_value over the lower bound would report 1.65
    for an interval of (0.84, 1.13) -- asserting robustness for a null result."""
    assert e_value_for_interval(0.84, 1.13) == 1.0


def test_the_interval_e_value_uses_the_limit_nearest_the_null():
    assert e_value_for_interval(1.24, 3.16) == pytest.approx(e_value(1.24))
    assert e_value_for_interval(0.50, 0.80) == pytest.approx(e_value(0.80))


def test_the_interval_e_value_never_exceeds_the_point_estimate():
    assert e_value_for_interval(1.24, 3.16) < e_value(1.98)


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------

def _synthetic(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.choice(["White", "Black or African American", "Race Not Available"],
                       n, p=[0.8, 0.1, 0.1])
    ltv = rng.uniform(40, 97, n)
    denied = rng.random(n) < (0.08 + 0.004 * (ltv - 40)
                              + 0.10 * (group == "Black or African American"))
    action = np.where(denied, 3, rng.choice([1, 2], n, p=[0.95, 0.05]))
    # A block of purchased loans: other lenders' originations, never decisions here.
    action[: n // 10] = 6
    return pd.DataFrame({
        "action_taken": action, "derived_race": group,
        "loan_to_value_ratio": ltv,
        "income": rng.uniform(40, 200, n).round(),
        "loan_amount": rng.uniform(50_000, 400_000, n).round(-3),
        "debt_to_income_ratio": rng.choice(DTI_BANDS, n),
        "loan_purpose": rng.choice(["1", "2"], n),
        "lien_status": "1", "occupancy_type": "1",
        "activity_year": rng.choice([2022, 2023], n),
        "denial_reason-1": np.where(denied, rng.choice([3, 4], n), 10),
        "denial_reason-2": 10, "denial_reason-3": 10, "denial_reason-4": 10,
    })


def test_purchased_loans_are_excluded_from_decisions():
    """action_taken == 6 is a loan bought on the secondary market. Counting it as an
    approval would dilute every denial rate toward zero."""
    df = _synthetic()
    assert (df["action_taken"] == 6).sum() > 0
    assert 6 not in set(decisioned(df)["action_taken"])


def test_denial_rates_match_a_hand_count():
    df = decisioned(_synthetic())
    table = denial_rates(df, "derived_race").set_index("derived_race")
    for group, row in table.iterrows():
        subset = df[df["derived_race"] == group]
        assert row["n"] == len(subset)
        assert row["denials"] == (subset["action_taken"] == 3).sum()
        assert row["denial_rate"] == pytest.approx(row["denials"] / row["n"])
        assert row["ci_low"] <= row["denial_rate"] <= row["ci_high"]


def test_denial_rates_are_ordered_by_group_size():
    table = denial_rates(decisioned(_synthetic()), "derived_race")
    assert list(table["n"]) == sorted(table["n"], reverse=True)


def test_reason_shares_may_exceed_one_because_reasons_are_not_exclusive():
    mix = denial_reason_mix(decisioned(_synthetic()))
    assert set(mix["reason"]) <= set(DENIAL_REASONS.values())
    assert (mix["share"] <= 1).all()


def test_missing_data_categories_are_kept_out_of_the_model():
    """"Race Not Available" is missingness wearing a category label. Including it
    would dilute every estimate toward the population mean."""
    result = adjusted_disparity(_synthetic(4_000), "derived_race", "White",
                                min_group=50)
    assert not set(result["group"]) & set(NOT_A_GROUP)


def test_the_reference_group_is_not_reported_against_itself():
    result = adjusted_disparity(_synthetic(4_000), "derived_race", "White",
                                min_group=50)
    assert "White" not in set(result["group"])
    assert result.attrs["reference"] == "White"


def test_the_model_recovers_a_disparity_that_is_in_the_data():
    """The synthetic frame is generated with a real penalty on one group. If the
    model cannot find a disparity that was put there on purpose, it cannot be
    trusted to find one that was not."""
    result = adjusted_disparity(_synthetic(6_000), "derived_race", "White",
                                min_group=50)
    black = result.set_index("group").loc["Black or African American"]
    assert black["odds_ratio"] > 1.3
    assert black["ci_low"] > 1.0


def test_small_groups_are_pooled_and_flagged_unreliable():
    result = adjusted_disparity(_synthetic(4_000), "derived_race", "White",
                                min_group=100_000)
    pooled = result[result["group"] == POOLED_LABEL]
    assert not pooled.empty and not pooled["reliable"].any()


def test_the_model_reports_what_it_could_not_control_for():
    """The omitted-variable list is part of the result, not a footnote someone may
    forget to carry."""
    result = adjusted_disparity(_synthetic(4_000), "derived_race", "White",
                                min_group=50)
    assert "credit score" in result.attrs["omitted"]
    assert result.attrs["converged"]


def test_ltv_bands_partition_the_population():
    table = ltv_distribution(decisioned(_synthetic()))
    assert table["n"].sum() == table.attrs["n"]
    assert table["share"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Contracts against the real filings
# --------------------------------------------------------------------------

real = pytest.mark.skipif(not (EXTERNAL / "hmda_2023.csv").exists(),
                          reason="run `make data` first")


@pytest.fixture(scope="module")
def hmda():
    from ames.data import load_hmda
    return load_hmda()


@real
def test_every_documented_year_is_present(hmda):
    from ames.data import HMDA_YEARS
    assert set(hmda["activity_year"]) == set(HMDA_YEARS)


@real
def test_the_panel_is_the_documented_size(hmda):
    """Quoted in DATA_SOURCES.md. Ranges, not equality: the CFPB re-publishes years
    as institutions amend filings."""
    assert 30_000 < len(hmda) < 36_000
    assert 23_000 < hmda["decisioned"].sum() < 27_000


@real
def test_the_denial_rate_is_in_a_plausible_range(hmda):
    rate = hmda.loc[hmda["decisioned"], "denied"].mean()
    assert 0.08 < rate < 0.16


@real
def test_the_protected_class_attributes_survive_the_load(hmda):
    """The whole reason for this source. If these columns vanish, the notebook is
    computing disparities on nothing."""
    for column in ("derived_race", "derived_ethnicity", "derived_sex"):
        assert hmda[column].notna().all()
        assert hmda[column].nunique() > 1


@real
def test_collateral_denials_are_present_and_material(hmda):
    """The bridge to notebook 08: a collateral denial is a valuation killing a loan."""
    mix = denial_reason_mix(decisioned(hmda)).set_index("reason")
    assert mix.loc["Collateral", "denials"] > 100
    assert 0.05 < mix.loc["Collateral", "share"] < 0.35
