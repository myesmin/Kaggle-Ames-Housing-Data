"""HTTP interface to the AVM.

Three endpoints matter:

``POST /value``
    Property attributes in, a valuation with its uncertainty out. The caller supplies
    up to ten fields, not the eighty the model consumes. The rest are filled from the
    neighbourhood's median (numeric) or mode (categorical), and the response reports how
    many were imputed so the caller can discount accordingly.

``GET /properties/{pid}/report``
    The one-page HTML valuation report from notebook 10, rendered live from the same
    code path.

``GET /model-card``
    What the model is, how accurate it measured, where it is known to be weak, and what
    it must not be used for. Served from the promoted model artifacts, so it matches
    what is deployed.

Intervals are calibrated under the temporal regime (train on the past, predict the
future), which matches how a deployed AVM is used. The random-split band is narrower
and is not used.

Interval width is Mondrian conformal by valuation band: five bands, each with its own
half-width, re-fitted each quarter on the trailing twelve months of closed sales. The
band is chosen by the valuation, never by a sale price, which the caller does not have.

Run locally with ``make serve``; ``GET /docs`` is the generated OpenAPI browser.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .avm import MondrianConformal
from .features import DERIVED_FEATURES, add_engineered, add_geo_features
from .report import build_flags, build_valuation_report, find_comparables

ARTIFACTS = Path(__file__).resolve().parents[2] / "serving" / "artifacts"

DISCLAIMER = ("Demonstration service. Trained on 2006-2010 sales in one small city. "
              "Not for lending, underwriting or any real valuation decision.")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    """Model, comparables pool and card.  Cached: loaded once per process, not per call."""
    missing = [n for n in ("avm_model.joblib", "universe.parquet", "model_card.json")
               if not (ARTIFACTS / n).exists()]
    if missing:
        raise RuntimeError(
            f"missing promoted artifacts {missing} in {ARTIFACTS}. Run `make promote`.")

    artifacts = joblib.load(ARTIFACTS / "avm_model.joblib")
    universe = pd.read_parquet(ARTIFACTS / "universe.parquet")
    card = json.loads((ARTIFACTS / "model_card.json").read_text())

    # Per-neighbourhood defaults for every field the caller does not supply: medians
    # for numbers, modes for categories, computed once at load.
    numeric = [c for c in artifacts["numeric"] if c in universe.columns]
    nominal = [c for c in artifacts["nominal"] if c in universe.columns]
    by_hood = universe.groupby("Neighborhood", observed=True)
    defaults_num = by_hood[numeric].median(numeric_only=True)
    defaults_cat = by_hood[nominal].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])

    # The bias the model is known to carry in each neighbourhood, measured on sold
    # homes. Used to make report cautions neighbourhood-specific.
    fitted = _predict_frame(artifacts, universe)
    bias = ((fitted - universe["SalePrice"]) / universe["SalePrice"])
    bias_by_hood = bias.groupby(universe["Neighborhood"], observed=True).median()

    # Interval bands come from the promoted card, so they match what notebook 04
    # calibrated.
    interval = MondrianConformal.from_dict(card["interval"]["calibrator"])

    return {
        "artifacts": artifacts, "universe": universe, "card": card, "interval": interval,
        "numeric": numeric, "nominal": nominal,
        "defaults_num": defaults_num, "defaults_cat": defaults_cat,
        "bias_by_hood": bias_by_hood,
        "fallback_num": universe[numeric].median(numeric_only=True),
        "fallback_cat": {c: universe[c].mode().iloc[0] for c in nominal
                         if not universe[c].mode().empty},
    }


def _predict_frame(artifacts: dict, frame: pd.DataFrame) -> pd.Series:
    """Log-space prediction back to dollars, with the Duan smearing correction.

    Exponentiating a log prediction returns the conditional *median*, which biases
    every valuation low.  ``smearing`` is the factor measured on training residuals.
    """
    log_pred = artifacts["estimator"].predict(frame)
    return pd.Series(np.exp(log_pred) * artifacts["smearing"], index=frame.index)


# --------------------------------------------------------------------------
# Request / response shapes
# --------------------------------------------------------------------------

class ValuationRequest(BaseModel):
    """The ten fields a caller is expected to know about a house.

    Three are required. The rest are optional, and an omitted field means *unknown*: it is
    filled from the neighbourhood's typical home and counted in ``fields_imputed``. It
    never means zero: omitting a garage does not mean there is no garage.
    """
    neighborhood: Annotated[str, Field(description="Ames neighbourhood code, e.g. 'NAmes'")]
    gr_liv_area: Annotated[int, Field(gt=100, lt=10_000, description="Above-grade living area, sq ft")]
    overall_qual: Annotated[int, Field(ge=1, le=10, description="Overall material and finish quality")]
    overall_cond: Annotated[int | None, Field(ge=1, le=10, description="Overall condition")] = None
    year_built: Annotated[int | None, Field(ge=1850, le=2030)] = None
    year_remodeled: Annotated[int | None, Field(ge=1850, le=2030)] = None
    total_bsmt_sf: Annotated[int | None, Field(ge=0, lt=10_000, description="Total basement area, sq ft; 0 = no basement")] = None
    garage_cars: Annotated[int | None, Field(ge=0, le=6, description="Garage capacity in cars; 0 = no garage")] = None
    full_bath: Annotated[int | None, Field(ge=0, le=6)] = None
    lot_area: Annotated[int | None, Field(gt=0, lt=300_000, description="Lot size, sq ft")] = None

    model_config = {"json_schema_extra": {"examples": [{
        "neighborhood": "NAmes", "gr_liv_area": 1456, "overall_qual": 6,
        "overall_cond": 5, "year_built": 1977, "year_remodeled": 1977,
        "total_bsmt_sf": 1040, "garage_cars": 2, "full_bath": 2, "lot_area": 9600,
    }]}}

    def to_columns(self) -> dict[str, Any]:
        """Only the fields actually supplied; the rest are left for imputation."""
        columns = {
            "Neighborhood": self.neighborhood,
            "Gr Liv Area": self.gr_liv_area,
            "Overall Qual": self.overall_qual,
            "Overall Cond": self.overall_cond,
            "Year Built": self.year_built,
            # A house never remodelled was last "remodelled" when it was built.
            "Year Remod/Add": self.year_remodeled or self.year_built,
            "Total Bsmt SF": self.total_bsmt_sf,
            "Garage Cars": self.garage_cars,
            "Full Bath": self.full_bath,
            "Lot Area": self.lot_area,
        }
        return {k: v for k, v in columns.items() if v is not None}


class FlagOut(BaseModel):
    level: str
    title: str
    detail: str


class Valuation(BaseModel):
    value: int = Field(description="Point estimate, US dollars")
    low: int = Field(description="Lower bound of the conformal interval")
    high: int = Field(description="Upper bound")
    confidence: float = Field(description="Nominal coverage of [low, high]")
    measured_coverage: float | None = Field(
        description="What that interval actually covered out-of-time. Below "
                    "`confidence` is a known, documented gap, not a bug.")
    interval_basis: str
    interval_band: int = Field(
        description="Which valuation band set the width, 1 (cheapest) to 5. Cheaper "
                    "bands carry wider ranges because the model is less precise there.")
    flags: list[FlagOut]
    comparables_found: int
    fields_supplied: int
    fields_imputed: int = Field(
        description="Model inputs filled from the neighbourhood rather than supplied. "
                    "A high count means a thinner valuation.")
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _build_subject(state: dict, supplied: dict[str, Any]) -> tuple[pd.Series, int]:
    """One complete model row: what the caller gave, plus neighbourhood defaults."""
    hood = supplied["Neighborhood"]
    row: dict[str, Any] = {}

    if hood in state["defaults_num"].index:
        row.update(state["defaults_num"].loc[hood].to_dict())
        row.update(state["defaults_cat"].loc[hood].to_dict())
    else:                                    # unknown neighbourhood -> city-wide
        row.update(state["fallback_num"].to_dict())
        row.update(state["fallback_cat"])

    # Derived features are computed from the row, not imputed, so they do not count.
    imputed = len([c for c in row if c not in supplied and c not in DERIVED_FEATURES])
    return _derive(_reconcile(row, supplied)), imputed


#: Fields that only exist if the structure does.  Their "absent" encodings are the ones
#: ``features.prepare`` uses: 0 for areas, counts and ordinal grades, "None" for types.
_GARAGE_ABSENT = {"Garage Area": 0, "Garage Qual": 0, "Garage Cond": 0,
                  "Garage Finish": 0, "Garage Type": "None", "Garage Yr Blt": np.nan}
_BASEMENT_ABSENT = {"BsmtFin SF 1": 0, "BsmtFin SF 2": 0, "Bsmt Unf SF": 0,
                    "Bsmt Full Bath": 0, "Bsmt Half Bath": 0, "Bsmt Qual": 0,
                    "Bsmt Cond": 0, "Bsmt Exposure": 0, "BsmtFin Type 1": 0,
                    "BsmtFin Type 2": 0}


def _rescale(row: dict[str, Any], parts: tuple[str, ...], target: float) -> None:
    """Scale imputed component areas so they sum to a supplied total."""
    current = sum(float(row.get(p) or 0) for p in parts)
    if current > 0:
        for p in parts:
            row[p] = float(row.get(p) or 0) * target / current
    else:
        row[parts[0]] = target


def _reconcile(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    """Make the imputed fields agree with the supplied ones.

    Neighbourhood medians are imputed field by field, so without this a caller's
    2,600 sq ft house arrives with the neighbourhood's median 1st-floor area, and a
    house with no garage keeps the median 418 sq ft of one. Component areas are rescaled
    to the supplied totals; a supplied zero removes the structure entirely.
    """
    row = {**defaults, **supplied}
    if "Gr Liv Area" in supplied:
        low_qual = min(float(row.get("Low Qual Fin SF") or 0), float(supplied["Gr Liv Area"]))
        row["Low Qual Fin SF"] = low_qual
        _rescale(row, ("1st Flr SF", "2nd Flr SF"), float(supplied["Gr Liv Area"]) - low_qual)

    if "Garage Cars" in supplied:
        if supplied["Garage Cars"] == 0:
            row.update(_GARAGE_ABSENT)
        else:
            # The neighbourhood's garage area per car; if its median home has no garage,
            # fall back to a typical single bay.
            cars, area = float(defaults.get("Garage Cars") or 0), float(defaults.get("Garage Area") or 0)
            per_car = area / cars if cars and area else 260.0
            row["Garage Area"] = per_car * supplied["Garage Cars"]

    if "Total Bsmt SF" in supplied:
        if supplied["Total Bsmt SF"] == 0:
            row.update(_BASEMENT_ABSENT)
        else:
            _rescale(row, ("BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF"),
                     float(supplied["Total Bsmt SF"]))
    return row


def _derive(row: dict[str, Any]) -> pd.Series:
    """Recompute every derived feature from the completed row.

    The same row-local functions training used (``features.add_engineered`` and
    ``add_geo_features``), so a served valuation sees the same features the model was
    fitted on. Leaving them at the neighbourhood's median was Model Risk Log defect #12:
    ``qual_x_sf`` never moved, so neither did the value.
    """
    frame = pd.DataFrame([row])
    frame = add_geo_features(add_engineered(frame))
    return frame.iloc[0]


def value_property(supplied: dict[str, Any]) -> tuple[pd.Series, dict[str, Any]]:
    """Score one property.  Returns the completed row and the valuation payload."""
    state = _load()
    card = state["card"]
    hood = supplied["Neighborhood"]
    known = set(state["defaults_num"].index)
    if hood not in known:
        raise HTTPException(
            422, {"error": f"unknown neighbourhood {hood!r}",
                  "known_neighborhoods": sorted(known)})

    subject, imputed = _build_subject(state, supplied)
    frame = pd.DataFrame([subject])
    for col in state["numeric"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    value = float(_predict_frame(state["artifacts"], frame).iloc[0])

    # Multiplicative in dollars, because the model works in logs, with the half-width of
    # the valuation band the property falls in.
    interval = state["interval"]
    log_value = np.log([value])
    q = float(interval.half_width(log_value)[0])
    band = int(interval.band(log_value)[0]) + 1
    lo, hi = value * np.exp(-q), value * np.exp(q)

    comps = find_comparables(subject, state["universe"])
    flags = build_flags(subject, value, lo, hi,
                        neighborhood_bias=state["bias_by_hood"].get(hood),
                        n_comparables=len(comps))

    subject["avm_value"], subject["value_low"], subject["value_high"] = value, lo, hi
    payload = {
        "value": round(value), "low": round(lo), "high": round(hi),
        "confidence": card["interval"]["confidence"],
        "measured_coverage": card["interval"]["measured_coverage"],
        "interval_basis": card["interval"]["method"],
        "interval_band": band,
        "flags": [{"level": f.level, "title": f.title, "detail": f.detail} for f in flags],
        "comparables_found": len(comps),
        "fields_supplied": len(supplied),
        "fields_imputed": imputed,
    }
    return subject, payload


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(
    title="Ames AVM",
    summary="An automated valuation model, its uncertainty, and its own governance record.",
    description=__doc__,
    version="1.0.0",
    license_info={"name": "MIT"},
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health", tags=["operations"])
def health() -> dict[str, Any]:
    """Liveness, plus enough to tell *which* model answered."""
    try:
        state = _load()
    except RuntimeError as exc:
        return JSONResponse({"status": "unhealthy", "detail": str(exc)}, status_code=503)
    return {
        "status": "ok",
        "model": state["artifacts"]["model_name"],
        "comparables_pool": len(state["universe"]),
        "neighborhoods": len(state["defaults_num"]),
    }


@app.get("/model-card", tags=["governance"])
def model_card() -> dict[str, Any]:
    """Accuracy, interval behaviour, monitoring status and known weaknesses."""
    return _load()["card"]


@app.get("/neighborhoods", tags=["reference"])
def neighborhoods() -> dict[str, Any]:
    """Valid `neighborhood` values, with how many sales back each one."""
    state = _load()
    counts = state["universe"]["Neighborhood"].value_counts()
    return {"n": len(counts),
            "neighborhoods": [{"code": k, "sales": int(v),
                               "median_model_bias": round(float(state["bias_by_hood"].get(k, np.nan)), 4)}
                              for k, v in counts.items()]}


@app.post("/value", response_model=Valuation, tags=["valuation"])
def value(request: ValuationRequest) -> Valuation:
    """Value one property, with a conformal interval and any applicable cautions."""
    _, payload = value_property(request.to_columns())
    return Valuation(**payload)


@app.get("/properties/{pid}/report", response_class=HTMLResponse, tags=["valuation"])
def property_report(
    pid: Annotated[str, PathParam(description="10-character zero-padded parcel ID")],
) -> HTMLResponse:
    """The one-page client valuation report for a property already in the universe."""
    state = _load()
    universe = state["universe"]
    match = universe[universe["PID"].astype(str) == str(pid)]
    if match.empty:
        raise HTTPException(404, f"no property with PID {pid!r}")

    subject = match.iloc[0].copy()
    supplied = {c: subject[c] for c in
                ("Neighborhood", "Gr Liv Area", "Overall Qual", "Overall Cond",
                 "Year Built", "Year Remod/Add", "Total Bsmt SF", "Garage Cars",
                 "Full Bath", "Lot Area") if c in subject.index}
    scored, payload = value_property(supplied)

    # Comparables are drawn from *other* sales: a property is not its own comparable.
    pool = universe[universe["PID"].astype(str) != str(pid)]
    comps = find_comparables(subject, pool)
    flags = build_flags(subject, payload["value"], payload["low"], payload["high"],
                        neighborhood_bias=state["bias_by_hood"].get(subject["Neighborhood"]),
                        n_comparables=len(comps))
    subject["avm_value"] = payload["value"]
    subject["value_low"], subject["value_high"] = payload["low"], payload["high"]
    html_text = build_valuation_report(subject, payload["value"], payload["low"],
                                       payload["high"], comps, flags,
                                       confidence=payload["confidence"],
                                       measured_coverage=payload["measured_coverage"])
    return HTMLResponse(html_text)
