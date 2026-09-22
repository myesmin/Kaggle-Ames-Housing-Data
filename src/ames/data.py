"""Download every external source and assemble the analysis frames.

Run as a script (``python -m ames.data``) or via ``make data``.

Why the data is assembled rather than read
------------------------------------------
The first pass used a pre-split copy of the Ames data: train.csv (2,051 rows, with
SalePrice) and test.csv (878 rows, target withheld). With no holdout labels, test error
could not be measured locally, so a corrupted test frame went unnoticed
(MODEL_RISK_LOG.md, defect #1).

De Cock's publication carries all 2,930 rows with SalePrice, so the 878 withheld rows
get their target back and the first pass can be scored.  Everything else here adds the macro and geographic context the two CSVs
do not have.

Sources are listed in docs/DATA_SOURCES.md with retrieval URLs and date ranges.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import (BASELINE, EXTERNAL, PROCESSED, RAW, TEMPORAL_TEST_YEARS,
                     TEMPORAL_TRAIN_YEARS, ensure_writable_dirs)

# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------

#: De Cock (2011), *Journal of Statistics Education* 19(3).  Tab-delimited, 2,930 x 82.
DECOCK_URL = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"

#: Property coordinates from Max Kuhn's ``AmesHousing`` R package.  2,932 rows keyed on
#: a zero-padded 10-character PID; joins to 99.6% of the De Cock rows.
GEO_URL = "https://github.com/topepo/AmesHousing/raw/master/data/ames_geo.rda"


@dataclass(frozen=True)
class FredSeries:
    """One FRED series.  Fetched key-free through the ``fredgraph.csv`` endpoint."""

    series_id: str
    name: str
    freq: str
    description: str


FRED_SERIES: tuple[FredSeries, ...] = (
    FredSeries("ATNHPIUS11180Q", "hpi_ames_msa", "Q",
               "FHFA All-Transactions House Price Index, Ames IA MSA"),
    FredSeries("ATNHPIUS19169A", "hpi_story_county", "A",
               "FHFA All-Transactions House Price Index, Story County IA"),
    FredSeries("CSUSHPINSA", "hpi_case_shiller_us", "M",
               "S&P CoreLogic Case-Shiller U.S. National Home Price Index, NSA"),
    FredSeries("MORTGAGE30US", "mortgage30us", "W",
               "Freddie Mac 30-Year Fixed Rate Mortgage Average (PMMS)"),
    FredSeries("CPIAUCSL", "cpi", "M",
               "CPI for All Urban Consumers: All Items, SA"),
    FredSeries("DGS10", "treasury10y", "D",
               "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity"),
    FredSeries("PCPI19169", "income_pc_story", "A",
               "Per Capita Personal Income, Story County IA"),
    FredSeries("MEHOINUSIAA672N", "income_median_hh_iowa", "A",
               "Real Median Household Income in Iowa"),
    FredSeries("LAUMT191118000000003A", "unemployment_ames", "A",
               "Unemployment Rate, Ames IA MSA"),
)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

#: Zillow Research public CSVs.  Both are metro-level wide panels; only the Ames row is kept.
ZILLOW_SOURCES = {
    "zhvi_ames": ("https://files.zillowstatic.com/research/public_csvs/zhvi/"
                  "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"),
    "zori_ames": ("https://files.zillowstatic.com/research/public_csvs/zori/"
                  "Metro_zori_uc_sfrcondomfr_sm_month.csv"),
}
ZILLOW_METRO = "Ames, IA"

#: Home Mortgage Disclosure Act filings for the Ames MSA, from the CFPB's Data
#: Browser.  No key, no registration.  The endpoint *pre-computes* each filtered query
#: and answers with a 301 to a static file on files.ffiec.cfpb.gov, so redirects must
#: be followed. ``requests`` does this by default; ``curl`` without ``-L`` gets a
#: 182-byte HTML stub that looks like an empty result set.
HMDA_CSV = ("https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
            "?years={year}&msamds={msa}")
HMDA_MSA = "11180"                       # Ames, IA MSA (same market as the AVM)

#: FHFA All-Transactions House Price Index for every US metropolitan area, quarterly
#: back to 1975.  One 4 MB CSV, no key.  Lets Ames be placed in the distribution of
#: metros ("82% of metros fell further than Ames") instead of compared with one
#: national average.
#:
#: The file ships without a header row, and uses "-" for quarters an index does not
#: cover.  Both are handled in ``load_metro_hpi``.
FHFA_METRO_URL = ("https://www.fhfa.gov/hpi/download/quarterly_datasets/"
                  "hpi_at_metro.csv")
FHFA_METRO_COLUMNS = ("metro", "cbsa", "year", "quarter", "index_nsa", "pct_change")
AMES_CBSA = 11180
HMDA_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)

#: ``action_taken`` values that represent a decision *this* institution made.  6
#: ("purchased loan") is another lender's origination bought on the secondary market,
#: so it carries no decision and must be excluded from any denial rate.
HMDA_DECISIONED = (1, 2, 3)
HMDA_DENIED = 3

#: Columns kept from the 99-column filing.  Everything here is either an outcome, a
#: legitimate underwriting factor, or a protected-class attribute.
HMDA_COLUMNS = (
    "activity_year", "action_taken", "loan_purpose", "loan_type", "lien_status",
    "occupancy_type", "loan_amount", "property_value", "income",
    "loan_to_value_ratio", "debt_to_income_ratio", "interest_rate", "rate_spread",
    "derived_race", "derived_ethnicity", "derived_sex", "applicant_age",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "census_tract", "tract_minority_population_percent",
    "tract_to_msa_income_percentage", "derived_dwelling_category",
)

TIMEOUT = 120
RETRIES = 4

# NB: do *not* spoof a browser User-Agent here.  FRED's edge silently black-holes
# requests that claim to be Chrome but do not look like Chrome at the TLS layer; the
# connection hangs until the read timeout.  The stock requests User-Agent is accepted.


def _get(url: str) -> bytes:
    """GET with bounded exponential backoff.  Public endpoints rate-limit."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:  # noqa: BLE001 - retry any transport/HTTP failure
            last = exc
            if attempt < RETRIES - 1:
                wait = 2 ** attempt
                print(f"    retry {attempt + 1}/{RETRIES - 1} in {wait}s ({exc.__class__.__name__})")
                time.sleep(wait)
    raise RuntimeError(f"failed to download {url}") from last


def _cached(path: Path, url: str, force: bool = False) -> Path:
    """Download ``url`` to ``path`` unless it already exists."""
    if path.exists() and not force:
        print(f"  cached  {path.name}")
        return path
    print(f"  GET     {url}")
    path.write_bytes(_get(url))
    return path


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download_raw(force: bool = False) -> None:
    """Fetch the two property-level sources into ``data/raw/``."""
    print("raw property data")
    _cached(RAW / "AmesHousing.txt", DECOCK_URL, force)
    _cached(RAW / "ames_geo.rda", GEO_URL, force)


def download_external(force: bool = False) -> None:
    """Fetch macro series into ``data/external/``, one tidy CSV per series."""
    print("FRED series")
    for s in FRED_SERIES:
        raw = _cached(EXTERNAL / f"fred_{s.series_id}.csv",
                      FRED_CSV.format(series_id=s.series_id), force)
        df = pd.read_csv(raw)
        df.columns = ["date", s.name]
        df["date"] = pd.to_datetime(df["date"])
        # FRED writes "." for missing observations on daily/weekly series.
        df[s.name] = pd.to_numeric(df[s.name], errors="coerce")
        df.to_csv(EXTERNAL / f"{s.name}.csv", index=False)

    print("FHFA metro panel")
    _cached(EXTERNAL / "hpi_at_metro.csv", FHFA_METRO_URL, force)

    print("HMDA (CFPB Data Browser)")
    for year in HMDA_YEARS:
        # Each year is a separate pre-computed query; the endpoint can take a minute
        # to build one the first time it is asked for.
        _cached(EXTERNAL / f"hmda_{year}.csv",
                HMDA_CSV.format(year=year, msa=HMDA_MSA), force)

    print("Zillow Research")
    for name, url in ZILLOW_SOURCES.items():
        raw = _cached(EXTERNAL / f"zillow_{name}.csv", url, force)
        wide = pd.read_csv(raw)
        row = wide.loc[wide["RegionName"] == ZILLOW_METRO]
        if row.empty:
            raise RuntimeError(f"{ZILLOW_METRO} not found in {url}")
        date_cols = [c for c in wide.columns if c[:4].isdigit() and "-" in c]
        tidy = (row[date_cols].T.reset_index()
                .set_axis(["date", name.replace("_ames", "")], axis=1)
                .dropna())
        tidy["date"] = pd.to_datetime(tidy["date"])
        tidy.to_csv(EXTERNAL / f"{name}.csv", index=False)


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------

def load_decock() -> pd.DataFrame:
    """The full De Cock extract: 2,930 properties x 82 columns, SalePrice included."""
    df = pd.read_csv(RAW / "AmesHousing.txt", sep="\t")
    if df.shape != (2930, 82):
        raise AssertionError(f"expected (2930, 82) from De Cock, got {df.shape}")
    return df


def load_geo() -> pd.DataFrame:
    """Latitude/longitude keyed on the zero-padded PID string."""
    import pyreadr

    geo = pyreadr.read_r(str(RAW / "ames_geo.rda"))["ames_geo"]
    return geo.rename(columns={"PID": "pid_str"})[["pid_str", "Latitude", "Longitude"]]


def _sale_date(df: pd.DataFrame) -> pd.Series:
    """Month-start timestamp of the sale, for joining to the macro panel."""
    return pd.to_datetime(dict(year=df["Yr Sold"], month=df["Mo Sold"], day=1))


def build_properties() -> pd.DataFrame:
    """Assemble the property-level analysis frame.

    Adds to the De Cock columns:
      ``pid_str``      zero-padded PID, the geo join key
      ``Latitude``/``Longitude``
      ``sale_date``    month-start timestamp
      ``baseline_split``  which side of the first-pass modelling split the row sat on
      ``temporal_split``  2006-08 -> train, 2009-10 -> test
    """
    df = load_decock()
    df["pid_str"] = df["PID"].astype(str).str.zfill(10)

    geo = load_geo()
    df = df.merge(geo, on="pid_str", how="left", validate="one_to_one")
    coverage = df["Latitude"].notna().mean()
    if coverage < 0.99:
        raise AssertionError(f"geo join coverage {coverage:.3%} < 99%")
    print(f"  geo join: {coverage:.2%} of {len(df):,} properties")

    df["sale_date"] = _sale_date(df)

    # Which rows the first pass could see the target for.
    baseline_train = set(pd.read_csv(BASELINE / "train.csv")["PID"])
    baseline_test = set(pd.read_csv(BASELINE / "test.csv")["PID"])
    df["baseline_split"] = np.select(
        [df["PID"].isin(baseline_train), df["PID"].isin(baseline_test)],
        ["train", "test"], default="unassigned")

    df["temporal_split"] = np.select(
        [df["Yr Sold"].isin(TEMPORAL_TRAIN_YEARS), df["Yr Sold"].isin(TEMPORAL_TEST_YEARS)],
        ["train", "test"], default="unassigned")

    return clean_known_errors(df)


def clean_known_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the data corrections the original notebooks left commented out.

    Two fixes, both documented by De Cock himself:

    1. ``Garage Yr Blt == 2207`` is a typo for 2007 (house built 2006, sold 2007).
       Left in place it puts a garage 200 years in the future.
    2. Five partial/unusual sales with ``Gr Liv Area > 4000`` sq ft.  De Cock (2011, s4)
       explicitly recommends removing them: "three of them were partial sales that
       likely don't represent actual market values". Flagged rather than dropped, so
       the exclusion is visible and reversible.
    """
    df = df.copy()
    n_typo = int((df["Garage Yr Blt"] == 2207).sum())
    df.loc[df["Garage Yr Blt"] == 2207, "Garage Yr Blt"] = 2007

    df["is_outlier_grlivarea"] = df["Gr Liv Area"] > 4000
    print(f"  fixed {n_typo} Garage Yr Blt typo(s); "
          f"flagged {int(df['is_outlier_grlivarea'].sum())} Gr Liv Area > 4000 sq ft")
    return df


def build_macro() -> pd.DataFrame:
    """One monthly macro panel, forward-filled from each series' native frequency.

    Quarterly and annual series are placed at their period start and forward-filled;
    daily and weekly series are averaged to month.  Nothing is back-filled, so no
    observation is ever informed by the future.
    """
    frames = []
    for s in FRED_SERIES:
        df = pd.read_csv(EXTERNAL / f"{s.name}.csv", parse_dates=["date"])
        df = df.dropna().set_index("date")
        if s.freq in ("D", "W"):
            df = df.resample("MS").mean()
        else:
            df = df.resample("MS").asfreq()
        frames.append(df)

    for name in ("zhvi_ames", "zori_ames"):
        df = pd.read_csv(EXTERNAL / f"{name}.csv", parse_dates=["date"]).set_index("date")
        df.index = df.index.to_period("M").to_timestamp()
        frames.append(df.resample("MS").asfreq())

    macro = pd.concat(frames, axis=1).sort_index()
    macro = macro.ffill()
    macro.index.name = "date"

    # Real (2026-dollar) deflator off the latest available CPI print.
    macro["cpi_deflator_2026"] = macro["cpi"].dropna().iloc[-1] / macro["cpi"]
    return macro


def build(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download everything and write ``data/processed/{properties,macro}.parquet``."""
    ensure_writable_dirs()
    download_raw(force)
    download_external(force)

    print("assembling")
    props = build_properties()
    macro = build_macro()

    props.to_parquet(PROCESSED / "properties.parquet", index=False)
    macro.to_parquet(PROCESSED / "macro.parquet")
    print(f"  properties.parquet  {props.shape}")
    print(f"  macro.parquet       {macro.shape}  {macro.index.min():%Y-%m} -> {macro.index.max():%Y-%m}")
    return props, macro


def build_properties_only(force: bool = False) -> pd.DataFrame:
    """Write ``data/processed/properties.parquet`` from the two raw sources alone.

    The leakage tests need the property frame and nothing else, so CI builds just this:
    two small downloads, no FRED, Zillow or HMDA, and nothing that can fail because a
    third party moved a URL.
    """
    ensure_writable_dirs()
    download_raw(force)
    print("assembling")
    props = build_properties()
    props.to_parquet(PROCESSED / "properties.parquet", index=False)
    print(f"  properties.parquet  {props.shape}")
    return props


def load_metro_hpi() -> pd.DataFrame:
    """Quarterly house-price index for every US metro, long format.

    Adds ``t`` (the quarter as a decimal year) so windows can be selected with an
    inequality instead of tuple comparisons on (year, quarter).
    """
    path = EXTERNAL / "hpi_at_metro.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make data`")
    df = pd.read_csv(path, header=None, names=list(FHFA_METRO_COLUMNS),
                     na_values=["-", ""])
    df["index_nsa"] = pd.to_numeric(df["index_nsa"], errors="coerce")
    df = df.dropna(subset=["index_nsa"]).copy()
    df["t"] = df["year"] + (df["quarter"] - 1) / 4
    return df.sort_values(["cbsa", "t"]).reset_index(drop=True)


def load_hmda() -> pd.DataFrame:
    """Every Ames-MSA mortgage application 2018-2024, one row per filing.

    Adds two derived columns the raw filing does not carry:

    ``decisioned``  the application was originated, approved-not-accepted, or denied,
                    i.e. a credit decision was made on it
    ``denied``      that decision was a denial

    Numeric fields arrive as strings because HMDA uses sentinels ("Exempt", "NA") in
    numeric columns; they are coerced, and the sentinels become NaN rather than being
    silently read as zero.
    """
    frames = []
    for year in HMDA_YEARS:
        path = EXTERNAL / f"hmda_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run `make data`")
        df = pd.read_csv(path, low_memory=False)
        keep = [c for c in HMDA_COLUMNS if c in df.columns]
        frames.append(df[keep])

    hmda = pd.concat(frames, ignore_index=True)
    for col in ("loan_amount", "property_value", "income", "loan_to_value_ratio",
                "interest_rate", "rate_spread", "tract_minority_population_percent",
                "tract_to_msa_income_percentage"):
        if col in hmda.columns:
            hmda[col] = pd.to_numeric(hmda[col], errors="coerce")

    hmda["decisioned"] = hmda["action_taken"].isin(HMDA_DECISIONED)
    hmda["denied"] = hmda["action_taken"].eq(HMDA_DENIED)
    return hmda


def load_properties() -> pd.DataFrame:
    """Read the assembled property frame, with a helpful error if ``make data`` was skipped."""
    path = PROCESSED / "properties.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make data` first")
    return pd.read_parquet(path)


def load_macro() -> pd.DataFrame:
    path = PROCESSED / "macro.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `make data` first")
    return pd.read_parquet(path)


if __name__ == "__main__":
    if "--properties-only" in sys.argv:
        build_properties_only(force="--force" in sys.argv)
    else:
        build(force="--force" in sys.argv)
