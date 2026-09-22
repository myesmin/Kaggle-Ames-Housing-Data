"""Data-contract tests: the assertions behind ``make data``.

Every number checked here is quoted in docs/DATA_SOURCES.md.  If a source silently
changes shape, revises history, or stops publishing, these fail rather than letting a
downstream chart quietly go wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ames.config import EXTERNAL, PROCESSED
from ames.data import FRED_SERIES, load_macro, load_properties

# Each group skips on the file it actually reads. CI builds the property frame but not
# the macro panel, so one file-wide gate would either skip the property contracts or
# fail the macro ones.

@pytest.fixture(scope="module")
def props() -> pd.DataFrame:
    if not (PROCESSED / "properties.parquet").exists():
        pytest.skip("run `make data` first")
    return load_properties()


@pytest.fixture(scope="module")
def macro() -> pd.DataFrame:
    if not (PROCESSED / "macro.parquet").exists():
        pytest.skip("run `make data` first")
    return load_macro()


# --------------------------------------------------------------------------
# The property frame
# --------------------------------------------------------------------------

def test_full_de_cock_extract_is_present(props):
    """2,930 properties -- the 878 rows the first-pass split withheld now have a target."""
    assert len(props) == 2930
    assert props["PID"].is_unique
    assert props["SalePrice"].notna().all()


def test_de_cock_column_set_is_intact(props):
    """82 original columns, plus exactly the derived ones this project adds."""
    derived = {"pid_str", "Latitude", "Longitude", "sale_date",
               "baseline_split", "temporal_split", "is_outlier_grlivarea"}
    assert len(set(props.columns) - derived) == 82


def test_geo_join_covers_at_least_99_percent(props):
    coverage = props["Latitude"].notna().mean()
    assert coverage >= 0.99
    assert props.loc[props["Latitude"].notna(), "Latitude"].between(41.9, 42.1).all()
    assert props.loc[props["Longitude"].notna(), "Longitude"].between(-93.8, -93.5).all()


def test_the_baseline_split_is_recovered(props):
    """2,051 train + 878 test = 2,929; one De Cock row was in neither split file."""
    counts = props["baseline_split"].value_counts()
    assert counts["train"] == 2051
    assert counts["test"] == 878
    assert counts["unassigned"] == 1


def test_sales_span_2006_to_2010_with_the_documented_annual_counts(props):
    counts = props["Yr Sold"].value_counts().sort_index()
    assert counts.to_dict() == {2006: 625, 2007: 694, 2008: 622, 2009: 648, 2010: 341}


def test_the_temporal_split_matches_the_documented_sizes(props):
    counts = props["temporal_split"].value_counts()
    assert counts["train"] == 1941      # 2006-2008
    assert counts["test"] == 989        # 2009-2010


def test_arms_length_share_is_as_documented(props):
    assert (props["Sale Condition"] == "Normal").mean() == pytest.approx(0.824, abs=0.002)


def test_known_data_errors_are_corrected(props):
    """Defect #8: the fixes the original notebooks left commented out."""
    assert (props["Garage Yr Blt"] == 2207).sum() == 0
    assert props["Garage Yr Blt"].max() <= props["Yr Sold"].max()
    assert props["is_outlier_grlivarea"].sum() == 5


def test_informative_nulls_survive_to_the_feature_layer(props):
    """Defect #9: these five columns were dropped for nullity; NA means 'absent'."""
    for col in ["Pool QC", "Alley", "Fence", "Fireplace Qu", "Misc Feature"]:
        assert col in props.columns
        assert props[col].isna().sum() > 0


# --------------------------------------------------------------------------
# The macro panel
# --------------------------------------------------------------------------

EXPECTED_SPANS = {
    "hpi_ames_msa": (1986, 2025),
    "hpi_story_county": (1978, 2024),
    "hpi_case_shiller_us": (1987, 2025),
    "mortgage30us": (1971, 2025),
    "cpi": (1947, 2025),
    "treasury10y": (1962, 2025),
    "income_pc_story": (1969, 2023),
    "income_median_hh_iowa": (1984, 2024),
    "unemployment_ames": (1990, 2024),
    "zhvi": (2009, 2025),
    "zori": (2017, 2025),
}


def test_every_external_series_downloaded_and_is_non_empty(macro):
    for name in EXPECTED_SPANS:
        assert name in macro.columns, name
        assert macro[name].notna().sum() > 0, name


@pytest.mark.parametrize("name,span", EXPECTED_SPANS.items())
def test_series_date_ranges_are_as_documented(macro, name, span):
    """Starts exactly where documented; ends at or after the documented year."""
    observed = macro[name].dropna()
    start_year, min_end_year = span
    assert observed.index.min().year == start_year
    assert observed.index.max().year >= min_end_year


def test_fred_series_all_have_a_tidy_csv_on_disk(macro):
    for s in FRED_SERIES:
        path = EXTERNAL / f"{s.name}.csv"
        assert path.exists(), s.name
        df = pd.read_csv(path)
        assert list(df.columns) == ["date", s.name]
        assert len(df) > 0


def test_macro_panel_is_monthly_and_strictly_increasing(macro):
    assert macro.index.is_monotonic_increasing
    assert macro.index.is_unique
    assert (macro.index.day == 1).all()


def test_no_series_is_back_filled_into_the_past(macro):
    """Forward-fill only.  A value must never appear before its first true observation."""
    raw = pd.read_csv(EXTERNAL / "zori_ames.csv", parse_dates=["date"])
    first_true = raw["date"].min().to_period("M").to_timestamp()
    assert macro.loc[macro.index < first_true, "zori"].isna().all()


def test_cpi_deflator_is_one_at_the_end_and_above_one_in_the_past(macro):
    deflator = macro["cpi_deflator_2026"].dropna()
    assert deflator.iloc[-1] == pytest.approx(1.0)
    assert deflator.loc["2008-01-01"] > 1.0


def test_ames_and_national_hpi_tell_the_documented_crisis_story(macro):
    """Ames held up through 2006-2011 while the national index fell hard."""
    window = macro.loc["2006-01-01":"2011-12-31"]
    ames_drawdown = window["hpi_ames_msa"].min() / window["hpi_ames_msa"].max() - 1
    us_drawdown = window["hpi_case_shiller_us"].min() / window["hpi_case_shiller_us"].max() - 1
    assert us_drawdown < ames_drawdown        # the nation fell further
    assert ames_drawdown > -0.10              # Ames barely dipped


def test_zillow_levels_are_in_the_documented_range(macro):
    zhvi = macro["zhvi"].dropna()
    assert 140_000 < zhvi.iloc[0] < 165_000
    assert 260_000 < zhvi.iloc[-1] < 300_000
    zori = macro["zori"].dropna()
    assert 800 < zori.iloc[0] < 950
    assert 1_000 < zori.iloc[-1] < 1_300


# --------------------------------------------------------------------------
# Cited property-tax reference data
# --------------------------------------------------------------------------

def test_config_tax_rate_matches_the_committed_citation():
    """`config.PROPERTY_TAX_RATE` must not drift from data/reference/story_county_tax.csv."""
    from ames.config import (AMES_CONSOLIDATED_LEVY_PER_1000, DATA,
                             IOWA_RESIDENTIAL_ROLLBACK, PROPERTY_TAX_RATE)

    table = pd.read_csv(DATA / "reference" / "story_county_tax.csv", comment="#")
    latest = table.sort_values("assessment_year").iloc[-1]

    assert latest["levy_per_1000"] == pytest.approx(AMES_CONSOLIDATED_LEVY_PER_1000)
    assert latest["rollback"] == pytest.approx(IOWA_RESIDENTIAL_ROLLBACK)
    assert latest["effective_rate"] == pytest.approx(PROPERTY_TAX_RATE, abs=1e-6)


def test_effective_rate_column_is_internally_consistent():
    from ames.config import DATA

    table = pd.read_csv(DATA / "reference" / "story_county_tax.csv", comment="#")
    computed = table["levy_per_1000"] / 1000 * table["rollback"]
    assert np.allclose(computed, table["effective_rate"], atol=1e-6)
