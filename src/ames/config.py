"""Paths and project-wide constants.

Every constant that feeds a dollar figure is cited here or in docs/ASSUMPTIONS.md.
Nothing in the DCF is a magic number.
"""
from __future__ import annotations

from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
BASELINE = RAW / "baseline"   # the first-pass modelling split, committed (see DATA_SOURCES.md)
EXTERNAL = DATA / "external"    # downloaded macro series (gitignored)
PROCESSED = DATA / "processed"  # assembled analysis frames (gitignored)
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"

#: Directories the pipeline writes into.  ``report.save_report`` and ``viz.save``
#: create their own on demand; these are the ones ``ames.data`` needs to exist first.
WRITABLE = (RAW, EXTERNAL, PROCESSED)


def ensure_writable_dirs() -> None:
    """Create the output directories.  Call this from code that writes, not on import.

    Importing a module must not touch the filesystem.  The serving container runs as a
    non-root user on a tree it does not own, so an import-time ``mkdir`` turns
    ``import ames.service`` into a ``PermissionError`` for directories the service
    never writes to.  Read paths are computed above and are import-safe; creating them
    is the caller's business.
    """
    for path in WRITABLE:
        path.mkdir(parents=True, exist_ok=True)

# --- reproducibility -------------------------------------------------------
SEED = 42

# --- the two evaluation regimes -------------------------------------------
# Regime A ("random"): the shuffled split this project used in its first pass, kept so the
RANDOM_TEST_SIZE = 0.25

# Regime B ("temporal"): train on 2006-2008, predict 2009-2010.  Sales in the De Cock
# extract span 2006-2010 (625/694/622/648/341 by year), so this straddles the crash and
# is the valid design for a model used forward in time.
TEMPORAL_TRAIN_YEARS = (2006, 2007, 2008)
TEMPORAL_TEST_YEARS = (2009, 2010)

# --- valuation universe ----------------------------------------------------
# De Cock (2011) s4: "Sale Condition ... Partial ... homes ... not completed when
# assessed".  Non-arm's-length sales are priced by a different process than the open
# market, so the AVM is fit on Normal only; the rest feed the distressed-discount study.
ARMS_LENGTH_CONDITION = "Normal"

# --- Ames geography --------------------------------------------------------
# Iowa State University central campus (Beardshear Hall) and the Main Street downtown
# district.  Used for distance/bearing features: proximity to campus is the dominant
# non-structural price gradient.
ISU_CAMPUS = (42.0267, -93.6465)
DOWNTOWN_AMES = (42.0248, -93.6197)

# --- AVM industry thresholds ----------------------------------------------
# Institutional AVM acceptance thresholds as commonly specified by secondary-market
# purchasers and cited in AVM validation literature.  See docs/METHODOLOGY.md.
MDAPE_THRESHOLD = 0.05   # median absolute percentage error below 5%
PPE10_THRESHOLD = 0.75   # >=75% of predictions within +/-10% of sale price
CONFORMAL_ALPHA = 0.20   # 80% prediction intervals

# --- Story County / City of Ames property tax ------------------------------
# Story County Auditor, "LEVIES 2024 - Payable 2025-2026", consolidated rate for the
# AMES/AMES taxing district = 30.58245 per $1,000 of taxable value
#   (county 5.43987 + city 10.30432 + state/ag-extension 0.78046 + Ames CSD 14.05780).
# Iowa LSA Fiscal FactBook, "Rollback Percentages": AY2024 residential = 47.4316%.
# Effective rate on market value = 30.58245/1000 * 0.474316 = 1.4505%.
AMES_CONSOLIDATED_LEVY_PER_1000 = 30.58245
IOWA_RESIDENTIAL_ROLLBACK = 0.474316
PROPERTY_TAX_RATE = AMES_CONSOLIDATED_LEVY_PER_1000 / 1000.0 * IOWA_RESIDENTIAL_ROLLBACK

# --- CCAR / DFAST -----------------------------------------------------------
# Federal Reserve supervisory severely adverse scenario: house prices decline ~30%
# peak-to-trough over the nine-quarter planning horizon.
CCAR_SEVERELY_ADVERSE_HPI_SHOCK = -0.30
