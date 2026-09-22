"""Contract tests for the HTTP interface.

These are not "does FastAPI work" tests.  Each one pins a promise the service makes to
a caller: that the interval brackets the value, that an unknown neighbourhood is a 422
rather than a silent city-wide guess, that the report endpoint and the notebook
deliverable are the same code path, and that the model card cannot quietly diverge from
the model that is actually loaded.
"""
from __future__ import annotations

import pytest

from ames.service import ARTIFACTS

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "avm_model.joblib").exists(),
    reason="run `make promote` first")

VALID = {"neighborhood": "NAmes", "gr_liv_area": 1456, "overall_qual": 6,
         "overall_cond": 5, "year_built": 1977, "year_remodeled": 1977,
         "total_bsmt_sf": 1040, "garage_cars": 2, "full_bath": 2, "lot_area": 9600}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from ames.service import app
    return TestClient(app)


# --------------------------------------------------------------------------
# The valuation contract
# --------------------------------------------------------------------------

def test_a_valuation_comes_back_with_its_interval(client):
    body = client.post("/value", json=VALID).json()
    assert body["low"] < body["value"] < body["high"]
    assert 50_000 < body["value"] < 900_000


# Different neighbourhoods as well as sizes, so the probes span the valuation bands.
CHEAP = {**VALID, "neighborhood": "MeadowV", "gr_liv_area": 800, "overall_qual": 3}
DEAR = {**VALID, "neighborhood": "NridgHt", "gr_liv_area": 2600, "overall_qual": 9}


def test_the_interval_is_multiplicative_not_additive(client):
    """Log-space intervals mean a dearer house carries a wider dollar band.  An additive
    band would under-cover expensive collateral."""
    cheap = client.post("/value", json=CHEAP).json()
    dear = client.post("/value", json=DEAR).json()
    assert dear["high"] - dear["low"] > cheap["high"] - cheap["low"]
    # Symmetric in log space: value is the geometric midpoint of the range, to rounding.
    for body in (cheap, dear):
        assert (body["low"] * body["high"]) ** 0.5 == pytest.approx(body["value"], rel=1e-4)


def test_each_valuation_carries_the_width_of_its_own_band(client):
    """Mondrian: the half-width is the one calibrated for the valuation band the property
    falls in, as published on the model card -- not one width for every house."""
    bands = client.get("/model-card").json()["interval"]["bands"]
    seen = set()
    for request in (CHEAP, VALID, DEAR):
        body = client.post("/value", json=request).json()
        band = bands[body["interval_band"] - 1]
        lo_edge, hi_edge = band["valuation_from"] or 0, band["valuation_to"] or float("inf")
        assert lo_edge <= body["value"] < hi_edge
        rel = (body["high"] - body["low"]) / body["value"]
        assert rel == pytest.approx(band["relative_width"], abs=1e-4)
        seen.add(body["interval_band"])
    assert len(seen) == 3, "probe properties should land in three different bands"


def test_the_card_says_how_the_served_interval_was_validated(client):
    """The served bands are recalibrated on the latest sales, so no holdout remains to
    score them on. The card must say its coverage is a backtest, and show what each
    earlier interval achieved, rather than presenting the number as a holdout result."""
    interval = client.get("/model-card").json()["interval"]
    assert "backtest" in interval["measured_coverage_basis"]
    assert interval["calibrated_on"]["sales"] > 100
    once = interval["previous_calibrated_once"]["measured_coverage"]
    assert once < interval["measured_coverage"] < interval["confidence"]
    assert all(0 < b["backtest_coverage"] <= 1 for b in interval["bands"])


def test_the_card_publishes_contiguous_bands_with_the_cheapest_widest(client):
    """The bands tile the whole valuation axis, and the cheapest is wider than the
    middle -- the heterogeneity the constant-width interval ignored."""
    interval = client.get("/model-card").json()["interval"]
    bands = interval["bands"]
    assert bands[0]["valuation_from"] is None and bands[-1]["valuation_to"] is None
    for below, above in zip(bands, bands[1:]):
        assert below["valuation_to"] == above["valuation_from"]
    middle = bands[len(bands) // 2]
    assert bands[0]["relative_width"] > middle["relative_width"]
    assert interval["previous_constant_width"]["q"] > 0


def test_the_response_admits_how_much_was_imputed(client):
    """A caller supplying ten fields is getting a thinner valuation than one supplying
    eighty, and has a right to know it."""
    body = client.post("/value", json=VALID).json()
    assert body["fields_supplied"] == 10
    assert body["fields_imputed"] > 50


def test_measured_coverage_is_reported_even_though_it_undershoots(client):
    """The service promises 80%; the recalibrated interval backtests at about 79%.
    Reporting the gap is the point; hiding it would be the defect."""
    body = client.post("/value", json=VALID).json()
    assert body["confidence"] == 0.80
    assert 0.70 < body["measured_coverage"] < body["confidence"]


def test_better_quality_is_worth_more(client):
    """A monotonicity the model must not violate, whatever else it does."""
    values = [client.post("/value", json={**VALID, "overall_qual": q}).json()["value"]
              for q in (4, 6, 8)]
    assert values == sorted(values)


def test_derived_features_are_computed_from_what_the_caller_supplied():
    """Model Risk Log defect #12. The service fills unsupplied fields from neighbourhood
    defaults; the derived features (quality x area, total square feet, age...) must then
    be recomputed from the completed row, not left at the neighbourhood's median, or the
    model never sees most of what the caller told it."""
    from ames.service import value_property

    supplied = {"Neighborhood": "NAmes", "Gr Liv Area": 2600, "Overall Qual": 9,
                "Overall Cond": 5, "Year Built": 2005, "Year Remod/Add": 2005,
                "Total Bsmt SF": 1500, "Garage Cars": 3, "Full Bath": 3, "Lot Area": 12000}
    row, _ = value_property(supplied)
    assert row["qual_x_sf"] == 9 * 2600
    assert row["house_age"] == row["Yr Sold"] - 2005
    assert row["has_basement"] == 1 and row["has_garage"] == 1
    # The imputed floor areas are rescaled so the house is internally consistent.
    assert row["1st Flr SF"] + row["2nd Flr SF"] + row["Low Qual Fin SF"] == pytest.approx(2600)
    assert row["total_sf"] == pytest.approx(1500 + 2600 - row["Low Qual Fin SF"])


def test_serving_derivation_reproduces_the_training_features():
    """For every sold property, recomputing the derived features the way the service does
    gives exactly the values the model was trained on -- and DERIVED_FEATURES names
    precisely the columns the derivation creates."""
    import numpy as np
    import pandas as pd

    from ames.features import DERIVED_FEATURES
    from ames.service import _derive

    universe = pd.read_parquet(ARTIFACTS / "universe.parquet")
    derived_in_model = [c for c in DERIVED_FEATURES if c in universe.columns]
    raw = universe.drop(columns=derived_in_model)
    rebuilt = pd.DataFrame([_derive(r) for r in raw.to_dict("records")])
    assert set(rebuilt.columns) - set(raw.columns) == set(DERIVED_FEATURES)
    for col in derived_in_model:
        assert np.allclose(rebuilt[col], universe[col], equal_nan=True), col


def test_size_and_quality_move_the_value_within_a_neighbourhood(client):
    """Same neighbourhood, a 2,600 sq ft quality-9 house against a 900 sq ft quality-4 one.
    Before defect #12 was fixed the service valued them $132k and $143k."""
    small = client.post("/value", json={**VALID, "gr_liv_area": 900, "overall_qual": 4,
                                        "total_bsmt_sf": 600, "garage_cars": 1,
                                        "full_bath": 1}).json()
    large = client.post("/value", json={**VALID, "gr_liv_area": 2600, "overall_qual": 9,
                                        "total_bsmt_sf": 1500, "garage_cars": 3,
                                        "full_bath": 3, "year_built": 2005,
                                        "year_remodeled": 2005}).json()
    assert large["value"] > 1.8 * small["value"]


def test_an_omitted_field_is_imputed_not_zeroed(client):
    """Leaving out the garage means "unknown", filled from the neighbourhood -- not "no
    garage". Before this was fixed the README's own four-field example lost 27% of its
    value to basement and garage defaults of zero, and still reported ten fields supplied."""
    minimal = {"neighborhood": "NAmes", "gr_liv_area": 1456, "overall_qual": 6,
               "year_built": 1977}
    short = client.post("/value", json=minimal).json()
    full = client.post("/value", json=VALID).json()
    assert short["fields_supplied"] == 5          # four, plus remodel year = build year
    assert short["fields_imputed"] > full["fields_imputed"]
    assert short["value"] == pytest.approx(full["value"], rel=0.10)

    no_garage = client.post("/value", json={**minimal, "garage_cars": 0}).json()
    assert no_garage["value"] < short["value"]


def test_supplying_no_garage_means_no_garage():
    from ames.service import value_property

    row, _ = value_property({"Neighborhood": "NAmes", "Gr Liv Area": 1200,
                             "Overall Qual": 5, "Garage Cars": 0, "Total Bsmt SF": 0})
    assert row["Garage Area"] == 0 and row["has_garage"] == 0
    assert row["has_basement"] == 0


def test_every_valuation_carries_the_disclaimer(client):
    assert "Not for lending" in client.post("/value", json=VALID).json()["disclaimer"]


# --------------------------------------------------------------------------
# Failing loudly
# --------------------------------------------------------------------------

def test_an_unknown_neighbourhood_is_rejected_not_guessed(client):
    r = client.post("/value", json={**VALID, "neighborhood": "Atlantis"})
    assert r.status_code == 422
    assert "known_neighborhoods" in r.json()["detail"]


@pytest.mark.parametrize("field,value", [
    ("overall_qual", 99), ("gr_liv_area", 12), ("lot_area", -1), ("year_built", 1600),
])
def test_out_of_range_inputs_are_refused(client, field, value):
    assert client.post("/value", json={**VALID, field: value}).status_code == 422


def test_a_missing_property_is_a_404(client):
    assert client.get("/properties/0000000000/report").status_code == 404


# --------------------------------------------------------------------------
# Report and governance endpoints
# --------------------------------------------------------------------------

def test_the_report_endpoint_serves_the_client_deliverable(client):
    import pandas as pd
    pid = str(pd.read_parquet(ARTIFACTS / "universe.parquet")["PID"].iloc[0])
    r = client.get(f"/properties/{pid}/report")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    # Self-contained: no network calls, no scripts.  It has to render from a file.
    assert "<script" not in r.text.lower()
    assert "http://" not in r.text.replace("http://www.w3.org", "")


def test_a_property_is_not_its_own_comparable(client):
    import pandas as pd
    u = pd.read_parquet(ARTIFACTS / "universe.parquet")
    pid = str(u["PID"].iloc[0])
    # The subject's own parcel ID belongs in the header exactly once.  A second
    # occurrence would mean it had been selected as a comparable to itself.
    assert client.get(f"/properties/{pid}/report").text.count(pid) == 1


def test_the_model_card_describes_the_model_that_is_loaded(client):
    """The card is served from the promoted artifacts, so it cannot drift from them."""
    card = client.get("/model-card").json()
    health = client.get("/health").json()
    assert health["model"].lower() in card["model"].lower()
    assert card["accuracy"]["mdape"] < 0.10
    assert card["known_weaknesses"] and card["not_for"]


def test_the_card_quotes_the_temporal_regime_not_the_flattering_one(client):
    """Random-split coverage is higher. A deployed AVM predicts forward, so the
    honest number is the temporal one."""
    card = client.get("/model-card").json()
    assert card["evaluation_regime"].startswith("temporal")
    assert card["interval"]["measured_coverage"] < card["interval"]["confidence"]


def test_health_reports_which_model_answered(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["comparables_pool"] > 2_000
    assert body["neighborhoods"] > 20


def test_the_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    assert "/value" in schema["paths"]
    assert "/model-card" in schema["paths"]


# --------------------------------------------------------------------------
# Container-safety
# --------------------------------------------------------------------------

def test_importing_the_service_never_creates_a_directory():
    """Import must not touch the filesystem.

    The serving image runs as a non-root user on a tree it does not own, so an
    import-time ``mkdir`` -- for directories the service never writes to -- turns
    ``import ames.service`` into a ``PermissionError``. That is exactly how this
    failed the first time the container was run, and it is invisible from a dev
    machine where every path is writable.
    """
    import subprocess
    import sys

    code = (
        "import pathlib\n"
        "def _forbidden(self, *a, **k):\n"
        "    raise AssertionError(f'import-time mkdir of {self}')\n"
        "pathlib.Path.mkdir = _forbidden\n"
        "import ames.service\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
