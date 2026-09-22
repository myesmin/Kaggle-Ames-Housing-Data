"""Tests for the national metro panel.

The drawdown calculation is four lines and carries a lot of weight: it is what turns
"Ames is unusually calm" from an assertion into a measurement, and what tells notebook
08 that its -30% scenario is not severe. So the four lines are checked against frames
where the answer is known by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ames.config import EXTERNAL
from ames.metro import (CRISIS_END, CRISIS_START, drawdowns, percentile_of,
                        severity_of, shock_at_quantile)


def _panel(series: dict[int, list[float]], start: float = 2005.0) -> pd.DataFrame:
    rows = []
    for cbsa, values in series.items():
        for i, v in enumerate(values):
            t = start + i / 4
            rows.append({"cbsa": cbsa, "metro": f"Metro {cbsa}",
                         "year": int(t), "quarter": int(round((t % 1) * 4)) + 1,
                         "index_nsa": v, "t": t})
    return pd.DataFrame(rows)


def _ramp(peak_at: int, peak: float, trough: float, n: int = 32) -> list[float]:
    """Rise to `peak` at quarter `peak_at`, fall to `trough`, then recover a little."""
    up = list(np.linspace(100.0, peak, peak_at + 1))
    down = list(np.linspace(peak, trough, n - peak_at - 1))
    return up + down


# --------------------------------------------------------------------------

def test_a_simple_drawdown_is_computed_exactly():
    table = drawdowns(_panel({1: _ramp(peak_at=8, peak=200.0, trough=100.0)}),
                      min_quarters=10)
    assert table["drawdown"].iloc[0] == pytest.approx(-0.5)


def test_the_peak_is_each_metro_s_own_not_a_common_date():
    """One metro peaks in 2007, one in 2009. Measuring both from a fixed date would
    score the late peaker as milder than it was."""
    early = _ramp(peak_at=8, peak=200.0, trough=140.0)     # -30% from its own peak
    late = _ramp(peak_at=16, peak=200.0, trough=140.0)     # also -30%, later
    table = drawdowns(_panel({1: early, 2: late}), min_quarters=10).set_index("cbsa")
    assert table.loc[1, "drawdown"] == pytest.approx(table.loc[2, "drawdown"], abs=1e-9)
    assert table.loc[2, "peak_t"] > table.loc[1, "peak_t"]


def test_a_market_that_only_rises_has_a_drawdown_of_zero():
    table = drawdowns(_panel({1: list(np.linspace(100, 200, 32))}), min_quarters=10)
    assert table["drawdown"].iloc[0] == pytest.approx(0.0)


def test_drawdowns_are_never_positive():
    rng = np.random.default_rng(0)
    panel = _panel({i: list(100 + rng.normal(0, 10, 32).cumsum()) for i in range(1, 12)})
    assert (drawdowns(panel, min_quarters=10)["drawdown"] <= 0).all()


def test_metros_with_too_little_history_are_dropped():
    table = drawdowns(_panel({1: _ramp(8, 200.0, 100.0), 2: [100.0] * 5}),
                      min_quarters=10)
    assert set(table["cbsa"]) == {1}


def test_the_window_bounds_are_respected():
    """An index that crashes in 2015 did not crash in the crisis."""
    panel = _panel({1: _ramp(8, 200.0, 100.0)}, start=2014.0)
    assert drawdowns(panel, start=CRISIS_START, end=CRISIS_END,
                     min_quarters=10).empty


def test_percentile_counts_metros_that_fell_further():
    table = pd.DataFrame({"cbsa": [1, 2, 3, 4], "drawdown": [-0.5, -0.4, -0.1, -0.05]})
    assert percentile_of(table, 3) == pytest.approx(0.5)      # 1 and 2 fell further
    assert percentile_of(table, 1) == pytest.approx(0.0)      # nothing fell further


def test_an_unknown_metro_raises_rather_than_returning_nan():
    table = pd.DataFrame({"cbsa": [1], "drawdown": [-0.2]})
    with pytest.raises(KeyError):
        percentile_of(table, 999)


def test_severity_and_quantile_are_inverses():
    rng = np.random.default_rng(1)
    table = pd.DataFrame({"cbsa": range(400), "drawdown": -rng.uniform(0, 0.6, 400)})
    for q in (0.05, 0.10, 0.25):
        shock = shock_at_quantile(table, q)
        assert severity_of(table, shock) == pytest.approx(q, abs=0.01)


def test_a_deeper_shock_is_exceeded_by_fewer_metros():
    table = pd.DataFrame({"cbsa": range(100),
                          "drawdown": np.linspace(-0.6, 0.0, 100)})
    assert severity_of(table, -0.5) < severity_of(table, -0.3) < severity_of(table, -0.1)


# --------------------------------------------------------------------------
# Contracts against the published file
# --------------------------------------------------------------------------

real = pytest.mark.skipif(not (EXTERNAL / "hpi_at_metro.csv").exists(),
                          reason="run `make data` first")


@pytest.fixture(scope="module")
def table():
    from ames.data import load_metro_hpi
    return drawdowns(load_metro_hpi())


@real
def test_the_panel_covers_the_documented_number_of_metros(table):
    assert 380 < len(table) < 440


@real
def test_ames_is_among_the_calmest_metros(table):
    """The fact every limitation in this project traces back to."""
    from ames.data import AMES_CBSA
    ames = table.loc[table["cbsa"] == AMES_CBSA, "drawdown"].iloc[0]
    assert -0.08 < ames < -0.01
    assert percentile_of(table, AMES_CBSA) > 0.75


@real
def test_the_imposed_thirty_percent_shock_is_not_severe(table):
    """Quoted in notebook 08 and the README. If FHFA revises history enough to move
    this, the claim needs rewriting rather than quietly drifting."""
    assert 0.12 < severity_of(table, -0.30) < 0.26
    assert shock_at_quantile(table, 0.10) < -0.35
