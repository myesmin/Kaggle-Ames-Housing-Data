"""The client-facing valuation report: comparable selection, flags, and rendering."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from ames.config import PROCESSED
from ames.report import Flag, build_flags, build_valuation_report, find_comparables

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "properties.parquet").exists(), reason="run `make data` first")


@pytest.fixture(scope="module")
def properties():
    from ames.data import load_properties
    return load_properties()


@pytest.fixture(scope="module")
def subject(properties):
    return properties.iloc[100]


# --------------------------------------------------------------------------
# Comparable selection
# --------------------------------------------------------------------------

def test_comparables_exclude_the_subject_itself(properties, subject):
    comps = find_comparables(subject, properties, n=5)
    assert subject.name not in comps.index


def test_comparables_are_similar_in_size(properties, subject):
    comps = find_comparables(subject, properties, n=5, size_tolerance=0.25)
    ratio = comps["Gr Liv Area"] / subject["Gr Liv Area"]
    assert ratio.between(0.75, 1.25).all()


def test_comparables_are_nearby(properties, subject):
    comps = find_comparables(subject, properties, n=5, max_miles=1.5)
    same_area = comps["Neighborhood"] == subject["Neighborhood"]
    close = comps["miles_away"] <= 1.5
    assert (same_area | close).all()


def test_requesting_more_comparables_returns_more(properties, subject):
    assert len(find_comparables(subject, properties, n=3)) == 3
    assert len(find_comparables(subject, properties, n=8)) == 8


def test_filters_relax_and_the_relaxation_is_recorded(properties):
    """An unusual property should still get comparables, with the compromise declared."""
    odd = properties.loc[properties["Gr Liv Area"].idxmax()]
    comps = find_comparables(odd, properties, n=5)
    assert len(comps) == 5
    assert isinstance(comps.attrs.get("relaxations"), list)


def test_an_empty_pool_returns_empty_rather_than_raising(subject):
    assert find_comparables(subject, subject.to_frame().T).empty


# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------

def test_a_typical_valuation_gets_a_good_flag(subject):
    flags = build_flags(subject, 200_000, 180_000, 222_000, n_comparables=5)
    assert any(f.level == "good" and "range is typical" in f.title for f in flags)


def test_an_unusually_wide_interval_is_flagged_as_a_caution(subject):
    flags = build_flags(subject, 200_000, 140_000, 280_000, n_comparables=5)
    assert any(f.level == "caution" and "wide" in f.title.lower() for f in flags)


def test_thin_comparable_evidence_is_flagged(subject):
    flags = build_flags(subject, 200_000, 180_000, 222_000, n_comparables=1)
    assert any("Few comparable sales" in f.title for f in flags)


@pytest.mark.parametrize("bias,word", [(-0.06, "under"), (0.06, "over")])
def test_neighborhood_bias_is_surfaced_in_the_right_direction(subject, bias, word):
    flags = build_flags(subject, 200_000, 180_000, 222_000,
                        neighborhood_bias=bias, n_comparables=5)
    assert any(f"{word}-value" in f.title for f in flags)


def test_small_neighborhood_bias_is_not_flagged(subject):
    flags = build_flags(subject, 200_000, 180_000, 222_000,
                        neighborhood_bias=0.005, n_comparables=5)
    assert not any("value this neighbourhood" in f.title for f in flags)


def test_price_extremes_carry_no_directional_warning(subject):
    """Regression test for Model Risk Log defect #11.

    The removed flag keyed on the valuation but quoted bias measured by sale price.
    Conditional on the valuation there is no price-band bias to warn about, so a cheap
    or expensive valuation alone must not tell the client which way to shade it.
    """
    for value, lo, hi in [(90_000, 80_000, 100_000), (400_000, 360_000, 445_000)]:
        flags = build_flags(subject, value, lo, hi, n_comparables=5)
        assert not any("over-valued" in f.title or "under-valued" in f.title for f in flags)


def test_a_high_screening_score_raises_a_caution(subject):
    flags = build_flags(subject, 200_000, 180_000, 222_000,
                        screening_score=0.85, n_comparables=5)
    assert any(f.level == "caution" and "non-market" in f.title for f in flags)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rendered(properties, subject):
    comps = find_comparables(subject, properties, n=5)
    flags = build_flags(subject, 185_000, 168_000, 205_000, n_comparables=len(comps))
    return build_valuation_report(subject, 185_000, 168_000, 205_000, comps, flags)


def test_report_is_self_contained(rendered):
    """No network calls: it has to survive being emailed or filed offline."""
    assert "<!doctype html>" in rendered.lower()
    assert "<script" not in rendered.lower()
    for pattern in ("http://", "https://", "src=", "@import"):
        assert pattern not in rendered.lower()


def test_report_leads_with_the_range_in_plain_language(rendered):
    assert "80% range" in rendered
    assert "$185,000" in rendered and "$168,000" in rendered and "$205,000" in rendered


def test_report_states_its_own_limitations(rendered):
    lowered = rendered.lower()
    assert "limitations" in lowered
    assert "not an appraisal" in lowered


def test_report_shows_every_comparable(properties, subject, rendered):
    comps = find_comparables(subject, properties, n=5)
    assert rendered.count("<tr>") >= len(comps) + 2        # comps + header + subject row


def test_report_escapes_user_supplied_text(properties, subject):
    """A neighbourhood name is data, not markup."""
    hostile = subject.copy()
    hostile["Neighborhood"] = '<script>alert("x")</script>'
    comps = find_comparables(subject, properties, n=3)
    out = build_valuation_report(hostile, 185_000, 168_000, 205_000, comps, [])
    assert "<script>alert" not in out
    assert "&lt;script&gt;" in out


def test_report_renders_with_no_comparables_and_no_flags(subject):
    out = build_valuation_report(subject, 185_000, 168_000, 205_000,
                                 pd.DataFrame(), [])
    assert "No comparable sales" in out


def test_interval_marker_sits_inside_the_band(subject):
    """The point estimate must be drawn between the two endpoints, not outside them."""
    out = build_valuation_report(subject, 170_000, 168_000, 205_000, pd.DataFrame(), [])
    x = float(re.search(r'<line x1="([\d.]+)"', out).group(1))
    assert 0 <= x <= 600
    # 170k of the way from 168k to 205k is ~5% along, so near the left end.
    assert x < 100
