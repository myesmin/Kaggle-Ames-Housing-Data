"""Feature engineering and the leak-proof preprocessing pipeline.

Design note
-----------
The first pass preprocessed train and test as separate dataframes (separate ``fillna``,
``get_dummies``, ``StandardScaler.fit``). That caused four of the ten defects in the
Model Risk Log.

Here, every statistic that has to be *learned* (imputation medians, one-hot vocabularies,
scaler moments) lives inside a single ``ColumnTransformer`` fit on training rows only.
Everything applied outside the pipeline (ordinal ladders, "NA means absent", geometry)
is a fixed function of a single row and reads nothing from the sample, so it cannot
leak.  ``tests/test_leakage.py`` asserts this directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DOWNTOWN_AMES, ISU_CAMPUS

# --------------------------------------------------------------------------
# Ordinal ladders
# --------------------------------------------------------------------------
# De Cock's data dictionary rates ten features on the same Ex > Gd > TA > Fa > Po
# ladder. One-hot encoding would drop the ordering and use five columns for one.  0 is reserved for "the feature is absent" (see below), which
# keeps the ladder monotone: no basement < poor basement < excellent basement.

QUALITY = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

ORDINAL_MAPS: dict[str, dict[str, int]] = {
    # --- the Ex/Gd/TA/Fa/Po ladder ---
    "Exter Qual": QUALITY,
    "Exter Cond": QUALITY,
    "Bsmt Qual": QUALITY,
    "Bsmt Cond": QUALITY,
    "Heating QC": QUALITY,
    "Kitchen Qual": QUALITY,
    "Fireplace Qu": QUALITY,
    "Garage Qual": QUALITY,
    "Garage Cond": QUALITY,
    "Pool QC": QUALITY,
    # --- other ordered scales ---
    "Bsmt Exposure": {"No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFin Type 2": {"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Garage Finish": {"Unf": 1, "RFn": 2, "Fin": 3},
    "Fence": {"MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                   "Min2": 6, "Min1": 7, "Typ": 8},
    "Lot Shape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "Land Slope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "Paved Drive": {"N": 1, "P": 2, "Y": 3},
    "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    "Electrical": {"Mix": 1, "FuseP": 2, "FuseF": 3, "FuseA": 4, "SBrkr": 5},
    "Central Air": {"N": 0, "Y": 1},
}

# --------------------------------------------------------------------------
# "NA means absent", not "NA means missing"
# --------------------------------------------------------------------------
# The first pass dropped Pool QC, Alley, Fence, Fireplace Qu and Misc Feature for
# high nullity (76-99% missing). De Cock's dictionary says NA in these
# columns codes *"No Pool"*, *"No Alley Access"*, *"No Fence"*, *"No Fireplace"*, *"None"*.
# So the nulls carry information: 2,732 of 2,930 Ames homes have no alley. Dropping
# them discards a fully-observed binary feature and, for Fireplace Qu, a quality
# ladder on the 1,508 homes that have a fireplace.

ABSENT_ORDINAL = [
    "Pool QC", "Fence", "Fireplace Qu",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
    "Garage Qual", "Garage Cond", "Garage Finish",
]
ABSENT_NOMINAL = ["Alley", "Misc Feature", "Garage Type", "Mas Vnr Type"]

# Basement/garage areas and counts are NaN only for homes lacking the structure.
ABSENT_NUMERIC = [
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Bsmt Full Bath", "Bsmt Half Bath", "Garage Cars", "Garage Area", "Mas Vnr Area",
]

# Actually missing: imputed inside the pipeline, from training rows only.
TRULY_MISSING = ["Lot Frontage", "Garage Yr Blt", "Latitude", "Longitude"]

# Dropped from the feature set: identifiers, split labels, and transaction attributes.
# Sale Type and Sale Condition describe the *transaction*, which does not exist yet at
# the moment an AVM is asked for a value; including them would leak the outcome.  They stay in the frame for the distressed-discount analysis.
NON_FEATURES = [
    "Order", "PID", "pid_str", "SalePrice", "sale_date",
    "baseline_split", "temporal_split", "is_outlier_grlivarea",
    "Sale Type", "Sale Condition",
]


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

EARTH_RADIUS_MI = 3958.7613


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in statute miles.  Vectorised over the first pair."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_degrees(lat1, lon1, lat2, lon2):
    """Initial compass bearing from point 1 to point 2, in degrees clockwise from north."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance and bearing to the two anchors that organise Ames.

    Iowa State dominates Ames. Distance to campus proxies
    the student-rental gradient; distance to downtown proxies walkable amenity.  Bearing
    is decomposed into sin/cos so that north-of-campus and 359-degrees are adjacent
    rather than maximally far apart.
    """
    df = df.copy()
    lat, lon = df["Latitude"], df["Longitude"]
    for name, (alat, alon) in (("isu", ISU_CAMPUS), ("downtown", DOWNTOWN_AMES)):
        df[f"dist_{name}_mi"] = haversine_miles(lat, lon, alat, alon)
        brg = bearing_degrees(alat, alon, lat, lon)
        df[f"bearing_{name}_sin"] = np.sin(np.radians(brg))
        df[f"bearing_{name}_cos"] = np.cos(np.radians(brg))
    return df


# --------------------------------------------------------------------------
# Engineered structural features
# --------------------------------------------------------------------------

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates a hedonic appraiser would use.

    All are row-local arithmetic on existing columns (no sample statistics), so they
    are safe to compute before the train/test split.
    """
    df = df.copy()
    df["total_sf"] = df["Total Bsmt SF"] + df["1st Flr SF"] + df["2nd Flr SF"]
    df["total_baths"] = (df["Full Bath"] + 0.5 * df["Half Bath"]
                         + df["Bsmt Full Bath"] + 0.5 * df["Bsmt Half Bath"])
    df["total_porch_sf"] = (df["Open Porch SF"] + df["Enclosed Porch"]
                            + df["3Ssn Porch"] + df["Screen Porch"] + df["Wood Deck SF"])
    df["house_age"] = df["Yr Sold"] - df["Year Built"]
    df["years_since_remodel"] = df["Yr Sold"] - df["Year Remod/Add"]
    df["is_remodeled"] = (df["Year Remod/Add"] > df["Year Built"]).astype(int)
    df["has_garage"] = (df["Garage Area"] > 0).astype(int)
    df["has_basement"] = (df["Total Bsmt SF"] > 0).astype(int)
    df["has_fireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["has_pool"] = (df["Pool Area"] > 0).astype(int)
    df["has_2nd_floor"] = (df["2nd Flr SF"] > 0).astype(int)
    df["qual_x_sf"] = df["Overall Qual"] * df["Gr Liv Area"]
    df["lot_coverage"] = df["1st Flr SF"] / df["Lot Area"].replace(0, np.nan)

    # Calendar position of the sale, for seasonality and the 2006-2010 price path.
    df["sale_time"] = df["Yr Sold"] + (df["Mo Sold"] - 1) / 12.0
    df["month_sin"] = np.sin(2 * np.pi * df["Mo Sold"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Mo Sold"] / 12)
    return df


#: Every column ``add_engineered`` and ``add_geo_features`` create.  Serving recomputes
#: these from the caller's inputs rather than imputing them (Model Risk Log defect #12);
#: a test asserts this list matches what the two functions produce.
DERIVED_FEATURES = (
    "total_sf", "total_baths", "total_porch_sf", "house_age", "years_since_remodel",
    "is_remodeled", "has_garage", "has_basement", "has_fireplace", "has_pool",
    "has_2nd_floor", "qual_x_sf", "lot_coverage", "sale_time", "month_sin", "month_cos",
    "dist_isu_mi", "bearing_isu_sin", "bearing_isu_cos",
    "dist_downtown_mi", "bearing_downtown_sin", "bearing_downtown_cos",
)


# --------------------------------------------------------------------------
# Row-local preparation (safe before the split)
# --------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Encode absences, apply the ordinal ladders, and derive features.

    Pure and row-local: ``prepare(df).loc[i] == prepare(df.loc[[i]]).loc[i]`` for every
    row, so applying it to the whole dataset before splitting leaks nothing.  Every
    learned statistic is deferred to :func:`build_preprocessor`.
    """
    df = df.copy()

    # "Absent" is a real level, not a missing value.
    for col in ABSENT_NUMERIC:
        if col in df:
            df[col] = df[col].fillna(0)
    for col in ABSENT_NOMINAL:
        if col in df:
            df[col] = df[col].fillna("None")

    # Ordinal ladders.  Absent -> 0, which sits below the worst present grade.
    for col, mapping in ORDINAL_MAPS.items():
        if col not in df:
            continue
        encoded = df[col].map(mapping)
        if col in ABSENT_ORDINAL:
            encoded = encoded.fillna(0)
        df[col] = encoded

    # MS SubClass is a numeric-looking code for building type; it is nominal.
    df["MS SubClass"] = df["MS SubClass"].astype(str)

    df = add_engineered(df)
    df = add_geo_features(df)
    return df


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition prepared columns into (numeric, nominal), excluding bookkeeping columns."""
    feats = [c for c in df.columns if c not in NON_FEATURES]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    nominal = [c for c in feats if c not in numeric]
    return numeric, nominal


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def build_preprocessor(numeric: list[str], nominal: list[str],
                       scale: bool = True) -> ColumnTransformer:
    """The only place sample statistics are learned.

    Parameters
    ----------
    scale
        ``True`` for penalised linear models, whose coefficients are only comparable on
        a common scale.  ``False`` for trees, which are invariant to monotone rescaling.

    ``handle_unknown="infrequent_if_exist"`` prevents defect #4: a
    categorical level that appears only in the test set is routed to an
    ``infrequent`` bucket instead of silently vanishing or shifting column order.
    """
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=5, sparse_output=False)),
            ]), nominal),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
