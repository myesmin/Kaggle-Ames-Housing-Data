"""Promote the trained AVM into a deployable artifact set.

The notebooks write everything they produce into ``data/processed/``, which is
gitignored and rebuilt by ``make all``.  A container cannot run the full pipeline at
build time: it would need the network, take several minutes, and rebuild the model on
every deploy.

Serving instead reads a promoted set: a small, committed, versioned snapshot of the
artifacts ``ames.service`` needs. Promotion is explicit (``make promote``), so the
deployed model is tracked separately from the last-trained one.

Writes ``serving/artifacts/``:

    avm_model.joblib   the fitted pipeline + Duan smearing factor
    universe.parquet   sold properties: the comparables pool and the field defaults
    model_card.json    metrics, the interval's valuation bands and their measured
                       coverage, monitoring status, known weaknesses
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ames.config import ARMS_LENGTH_CONDITION, CONFORMAL_ALPHA, PROCESSED, ROOT
from ames.data import load_properties
from ames.features import prepare

ARTIFACTS = ROOT / "serving" / "artifacts"

#: Fields the caller supplies.  Everything else the model wants is filled from the
#: neighbourhood, so the API asks for ten values rather than eighty.
REQUESTED_FIELDS = (
    "Neighborhood", "Gr Liv Area", "Overall Qual", "Overall Cond", "Year Built",
    "Year Remod/Add", "Total Bsmt SF", "Garage Cars", "Full Bath", "Lot Area",
)


def _summary(name: str) -> dict:
    path = PROCESSED / name
    return json.loads(path.read_text()) if path.exists() else {}


def build_model_card() -> dict:
    """Everything the service reports about itself, assembled from the pipeline."""
    avm = _summary("avm_summary.json")
    conformal = _summary("conformal.json")
    governance = _summary("governance_summary.json")
    monitoring = _summary("monitoring_summary.json")
    credit = _summary("credit_summary.json")

    # The service predicts forward in time, so it quotes the temporal-split regime,
    # not the (narrower) random-split number.
    temporal = conformal.get("temporal", {})
    launch = temporal.get("mondrian")
    served = _summary("recalibration.json")
    if not launch or not served:
        raise SystemExit("missing interval bands -- re-run notebooks 04 and 05")
    backtest = served["backtest"]
    adaptive = governance.get("adaptive_intervals", {})

    # The served bands are the recalibrated ones (notebook 05): re-fitted on the latest
    # twelve months of closed sales.  One entry per valuation band; the outer bands are
    # open-ended, so a valuation below the cheapest calibration sale still gets a width.
    edges = [float(np.exp(e)) for e in served["edges_log"]]
    n_bands = len(edges) - 1
    bands = [{
        "band": b + 1,
        "valuation_from": None if b == 0 else round(edges[b]),
        "valuation_to": None if b == n_bands - 1 else round(edges[b + 1]),
        "q": served["q_by_band"][b],
        "relative_width": served["relative_width_by_band"][b],
        "backtest_coverage": backtest["coverage_by_band"][b],
        "calibration_n": served["n_by_band"][b],
    } for b in range(n_bands)]
    calibrator = {k: served[k] for k in
                  ("alpha", "min_group", "edges_log", "q_by_band", "n_by_band", "q_global")}

    return {
        "model": "LightGBM on log(SalePrice), Duan-smeared to dollars",
        "training_data": "De Cock (2011) Ames extract, arm's-length sales 2006-2010",
        "evaluation_regime": "temporal (train 2006-2008, test 2009-July 2010)",
        "accuracy": {
            "mdape": avm.get("mdape_temporal"),
            "ppe10": avm.get("ppe10_temporal"),
            "rmse": avm.get("corrected_rmse"),
        },
        "interval": {
            "method": "Mondrian split-conformal by valuation quintile, recalibrated each "
                      "quarter on the trailing twelve months of closed sales; "
                      "multiplicative in log space",
            "confidence": 1 - CONFORMAL_ALPHA,
            "calibrated_on": {"from": served["window_start"], "to": served["window_end"],
                              "sales": served["window_n"]},
            "relative_width": served["relative_width"],
            # No sale after the calibration window exists, so coverage is measured by
            # backtest: every 2009-10 sale scored with the bands issued before its quarter.
            "measured_coverage": backtest["coverage_rolling"],
            "measured_coverage_basis": "backtest over 862 sales, 2009-2010, each scored "
                                       "with the bands issued before its quarter began",
            "bands": bands,
            # What it replaced, so a reader of the card can see what each step changed.
            "previous_calibrated_once": {
                "method": "Mondrian split-conformal, calibrated once on 2006-2008 sales",
                "relative_width": launch["relative_width"],
                "measured_coverage": launch["empirical_coverage"],
                "cheapest_decile_coverage": adaptive.get("Mondrian · value", {}).get("bottom_decile"),
            },
            "previous_constant_width": {
                "method": "split-conformal, one half-width for every valuation",
                "q": temporal.get("q"),
                "relative_width": temporal.get("relative_width"),
                "measured_coverage": temporal.get("empirical_coverage"),
                "cheapest_decile_coverage": adaptive.get("Global", {}).get("bottom_decile"),
            },
            "calibrator": calibrator,
        },
        "governance": {
            "neighborhood_bias_spread": governance.get("neighborhood_bias_spread"),
            "morans_i": governance.get("morans_i"),
        },
        "monitoring": {
            "controls_in_limit": not monitoring.get("live_breach", False),
            "quarters_watched": monitoring.get("quarters_monitored"),
        },
        "credit_context": {
            "base_expected_loss_bps": credit.get("base_el_bps"),
            "stressed_expected_loss_bps": credit.get("stressed_el_bps"),
            "cost_of_valuation_uncertainty_bps": credit.get("uncertainty_uplift_base_bps"),
        },
        "known_weaknesses": [
            "Trained on one small city, 2006-2010. Values are historical, not current.",
            f"Backtested coverage is {backtest['coverage_rolling']:.1%} against the "
            f"{1 - CONFORMAL_ALPHA:.0%} the interval promises: quarterly recalibration closes "
            "most of the gap a once-calibrated interval leaves, not all of it.",
            "The cheapest valuations are still under-covered: the model over-values "
            "them by about 2-3%, and no choice of width corrects a biased centre.",
            "Valuation errors are spatially autocorrelated: Neighborhood has not "
            "absorbed location.",
            "Not a fair-lending assessment. No protected-class attribute exists in "
            "this data.",
        ],
        "not_for": "Lending, underwriting or any real valuation decision. "
                   "This is a demonstration of a modelling and governance pipeline.",
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    model_src = PROCESSED / "avm_model.joblib"
    if not model_src.exists():
        raise SystemExit("no trained model found -- run `make run` first")
    shutil.copy2(model_src, ARTIFACTS / "avm_model.joblib")

    artifacts = joblib.load(model_src)
    data = prepare(load_properties())
    universe = data[(data["Sale Condition"] == ARMS_LENGTH_CONDITION)
                    & (~data["is_outlier_grlivarea"])].copy()

    # Only the columns the estimator consumes, plus what comparables and defaults
    # need. The full 89-column frame would triple the image size.
    extras = {"PID", "SalePrice", "Neighborhood", "Latitude", "Longitude",
              "Gr Liv Area", "Year Built", "Yr Sold", "Mo Sold", "sale_time"}
    model_inputs = set(artifacts["numeric"]) | set(artifacts["nominal"])
    missing = model_inputs - set(universe.columns)
    if missing:
        raise SystemExit(f"model expects columns absent from the universe: {sorted(missing)}")
    keep = sorted(model_inputs | (extras & set(universe.columns)))
    universe[keep].to_parquet(ARTIFACTS / "universe.parquet", index=False)

    card = build_model_card()
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")

    total = sum(p.stat().st_size for p in ARTIFACTS.iterdir())
    print(f"promoted to {ARTIFACTS.relative_to(ROOT)}/")
    for p in sorted(ARTIFACTS.iterdir()):
        print(f"  {p.name:<20} {p.stat().st_size / 1024:>8,.0f} KB")
    print(f"  {'total':<20} {total / 1024:>8,.0f} KB")
    print(f"  universe: {len(universe):,} sold properties, {len(keep)} columns")


if __name__ == "__main__":
    main()
