"""AVM metrics, retransformation, conformal coverage, and spatial diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from ames.avm import (
    ConformalizedQuantileRegression, ConformalValuation, MondrianConformal,
    duan_smearing_factor, error_parity, fsd, mdape,
    mean_signed_error, morans_i, parity_gap, percentage_error, ppe, predict_dollars,
    scorecard,
)

RNG = np.random.default_rng(20240601)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_perfect_prediction_scores_perfectly():
    a = np.array([100_000.0, 250_000.0, 400_000.0])
    card = scorecard(a, a)
    assert card["MdAPE"] == 0.0
    assert card["PPE5"] == card["PPE10"] == 1.0
    assert card["FSD"] == pytest.approx(0.0)
    assert card["mean_signed_error"] == pytest.approx(0.0)


def test_metrics_against_hand_computed_values():
    actual = np.array([100.0, 100.0, 100.0, 100.0])
    predicted = np.array([90.0, 104.0, 112.0, 130.0])   # errors -10%, +4%, +12%, +30%
    assert np.allclose(percentage_error(actual, predicted), [-0.10, 0.04, 0.12, 0.30])
    assert mdape(actual, predicted) == pytest.approx(0.11)   # median of .10 .04 .12 .30
    assert ppe(actual, predicted, 0.05) == pytest.approx(0.25)
    assert ppe(actual, predicted, 0.10) == pytest.approx(0.50)
    assert ppe(actual, predicted, 0.25) == pytest.approx(0.75)
    assert mean_signed_error(actual, predicted) == pytest.approx(0.09)


def test_mdape_is_robust_to_a_single_extreme_error():
    """The reason MdAPE and not MAPE is the industry headline metric."""
    actual = np.full(101, 200_000.0)
    predicted = actual * 1.03
    predicted[0] = 20_000_000.0
    assert mdape(actual, predicted) == pytest.approx(0.03)


def test_ppe_is_monotone_in_tolerance():
    a = RNG.lognormal(12, 0.3, 400)
    p = a * np.exp(RNG.normal(0, 0.15, 400))
    tolerances = [0.05, 0.10, 0.20, 0.25]
    values = [ppe(a, p, t) for t in tolerances]
    assert values == sorted(values)


def test_fsd_recovers_the_known_log_error_dispersion():
    a = np.full(20_000, 200_000.0)
    p = a * np.exp(RNG.normal(0, 0.18, 20_000))
    assert fsd(a, p) == pytest.approx(0.18, abs=0.01)


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        mdape([1, 2, 3], [1, 2])


# --------------------------------------------------------------------------
# Duan smearing
# --------------------------------------------------------------------------

def test_smearing_factor_is_one_for_zero_residuals():
    assert duan_smearing_factor(np.zeros(100)) == pytest.approx(1.0)


def test_smearing_factor_always_exceeds_one_for_nonzero_residuals():
    """Jensen's inequality: mean(exp(e)) > exp(mean(e)) = 1 whenever e varies."""
    assert duan_smearing_factor(RNG.normal(0, 0.2, 5000)) > 1.0


def test_smearing_corrects_the_retransformation_bias():
    """Naive exp() of a log-model prediction under-states the conditional mean.

    Construct a case with the answer known in closed form: y = exp(mu + e) with e
    normal, so E[y] = exp(mu + sigma^2/2).  Naive exp(mu) is low by exp(sigma^2/2);
    smearing must close essentially all of that gap.
    """
    mu, sigma, n = np.log(200_000), 0.25, 200_000
    resid = RNG.normal(0, sigma, n)
    y = np.exp(mu + resid)
    naive = np.exp(mu)
    corrected = predict_dollars(mu, duan_smearing_factor(resid))
    truth = y.mean()
    assert naive < truth
    assert abs(corrected - truth) / truth < 0.01
    assert abs(corrected - truth) < abs(naive - truth)


# --------------------------------------------------------------------------
# Conformal intervals -- the coverage guarantee
# --------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.10, 0.20, 0.32])
def test_empirical_coverage_matches_nominal_on_held_out_data(alpha):
    """Split conformal guarantees coverage >= 1 - alpha and should not be far above it."""
    n = 6000
    log_actual = RNG.normal(np.log(190_000), 0.45, n)
    log_pred = log_actual + RNG.normal(0, 0.16, n)
    calib, holdout = slice(0, n // 2), slice(n // 2, n)

    cv = ConformalValuation(alpha).calibrate(log_actual[calib], log_pred[calib])
    coverage = cv.coverage(log_actual[holdout], log_pred[holdout])
    assert coverage >= 1 - alpha - 0.02        # the guarantee, within sampling noise
    assert coverage <= 1 - alpha + 0.03        # and not wildly conservative


def test_coverage_holds_under_heteroskedastic_errors():
    """The case that breaks the textbook Gaussian prediction interval."""
    n = 8000
    log_actual = RNG.normal(np.log(190_000), 0.5, n)
    scale = 0.08 + 0.25 * (log_actual - log_actual.min()) / np.ptp(log_actual)
    log_pred = log_actual + RNG.normal(0, scale)
    cv = ConformalValuation(0.20).calibrate(log_actual[: n // 2], log_pred[: n // 2])
    assert cv.coverage(log_actual[n // 2:], log_pred[n // 2:]) >= 0.78


def test_tighter_alpha_gives_a_wider_interval():
    log_actual = RNG.normal(12, 0.4, 3000)
    log_pred = log_actual + RNG.normal(0, 0.15, 3000)
    widths = [ConformalValuation(a).calibrate(log_actual, log_pred).relative_width
              for a in (0.32, 0.20, 0.10, 0.05)]
    assert widths == sorted(widths)


def test_dollar_interval_is_multiplicative_and_brackets_the_point_estimate():
    log_actual = RNG.normal(12, 0.4, 2000)
    cv = ConformalValuation(0.20).calibrate(log_actual, log_actual + RNG.normal(0, 0.15, 2000))
    log_pred = np.log([120_000.0, 480_000.0])
    lo, hi = cv.interval_dollars(log_pred)
    assert np.all(lo < np.exp(log_pred)) and np.all(hi > np.exp(log_pred))
    # A constant ratio, not a constant dollar amount: the band scales with price.
    assert (hi[0] / lo[0]) == pytest.approx(hi[1] / lo[1], rel=1e-12)
    assert (hi[1] - lo[1]) > (hi[0] - lo[0])


def test_uncalibrated_interval_raises():
    with pytest.raises(RuntimeError):
        ConformalValuation(0.2).interval_log([12.0])


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        ConformalValuation(1.5)


# --------------------------------------------------------------------------
# Adaptive intervals -- Mondrian and CQR
# --------------------------------------------------------------------------

def _heteroskedastic(n: int, seed: int):
    """Error scale rises steeply with value: the shape that breaks a single half-width."""
    rng = np.random.default_rng(seed)
    log_pred = rng.normal(np.log(190_000), 0.45, n)
    rank = (log_pred - log_pred.min()) / np.ptp(log_pred)
    scale = 0.04 + 0.30 * rank
    return log_pred + rng.normal(0, scale), log_pred, scale


def _coverage_by_bin(log_actual, lo, hi, key, n_bins=5):
    edges = np.quantile(key, np.linspace(0, 1, n_bins + 1))
    bins = np.digitize(key, edges[1:-1])
    inside = (log_actual >= lo) & (log_actual <= hi)
    return np.array([inside[bins == b].mean() for b in range(n_bins)])


def test_mondrian_restores_within_group_coverage_where_the_global_interval_fails():
    """The global interval over-covers easy rows and under-covers hard ones; Mondrian doesn't."""
    y, p, _ = _heteroskedastic(12_000, 1)
    cal, test = slice(0, 6000), slice(6000, None)

    glob = ConformalValuation(0.20).calibrate(y[cal], p[cal])
    g_lo, g_hi = glob.interval_log(p[test])
    global_bins = _coverage_by_bin(y[test], g_lo, g_hi, p[test])
    assert global_bins.min() < 0.70 and global_bins.max() > 0.88   # the planted failure

    mond = MondrianConformal(0.20, n_bins=5).calibrate(y[cal], p[cal])
    m_lo, m_hi = mond.interval_log(p[test])
    mondrian_bins = _coverage_by_bin(y[test], m_lo, m_hi, p[test])
    assert np.all(np.abs(mondrian_bins - 0.80) < 0.04)


def test_mondrian_width_tracks_the_error_scale():
    y, p, _ = _heteroskedastic(8000, 2)
    mond = MondrianConformal(0.20, n_bins=5).calibrate(y, p)
    widths = [mond.q_by_group_[g] for g in sorted(mond.q_by_group_)]
    assert np.all(np.diff(widths) > 0)


def test_mondrian_small_groups_fall_back_to_the_global_half_width():
    y, p, _ = _heteroskedastic(2000, 3)
    groups = np.where(np.arange(2000) < 10, "tiny", "big")
    mond = MondrianConformal(0.20, min_group=30).calibrate(y, p, groups=groups)
    assert mond.q_by_group_["tiny"] == mond.q_global_
    assert mond.half_width(p[:1], groups=np.array(["never_seen"]))[0] == mond.q_global_


def test_mondrian_with_one_bin_equals_split_conformal():
    y, p, _ = _heteroskedastic(4000, 4)
    one = MondrianConformal(0.20, n_bins=1).calibrate(y, p)
    base = ConformalValuation(0.20).calibrate(y, p)
    assert np.allclose(one.half_width(p), base.q_)


def test_mondrian_survives_a_round_trip_through_json():
    """Serving rebuilds the calibrator from the model card; it must give identical bands,
    including for valuations beyond the calibration range."""
    import json

    y, p, _ = _heteroskedastic(3000, 9)
    mond = MondrianConformal(0.20, n_bins=5).calibrate(y, p)
    back = MondrianConformal.from_dict(json.loads(json.dumps(mond.to_dict())))
    probe = np.concatenate([p[:200], [p.min() - 1.0, p.max() + 1.0]])
    assert np.array_equal(back.half_width(probe), mond.half_width(probe))


def test_only_a_value_binned_mondrian_can_be_saved():
    y, p, _ = _heteroskedastic(500, 10)
    by_group = MondrianConformal(0.20).calibrate(y, p, groups=np.arange(500) % 3)
    with pytest.raises(RuntimeError):
        by_group.to_dict()


def test_cqr_restores_marginal_coverage_of_miscalibrated_quantile_models():
    """Quantile bands that are too narrow get widened back to the nominal level."""
    y, p, scale = _heteroskedastic(12_000, 5)
    lo, hi = p - 0.6 * 1.2816 * scale, p + 0.6 * 1.2816 * scale   # 60% of the true width
    cal, test = slice(0, 6000), slice(6000, None)
    raw = np.mean((y[test] >= lo[test]) & (y[test] <= hi[test]))
    assert raw < 0.65

    cqr = ConformalizedQuantileRegression(0.20).calibrate(y[cal], lo[cal], hi[cal])
    assert abs(cqr.coverage(y[test], lo[test], hi[test]) - 0.80) < 0.03


def test_cqr_narrows_bands_that_are_too_cautious():
    y, p, scale = _heteroskedastic(8000, 6)
    lo, hi = p - 3 * scale, p + 3 * scale
    cqr = ConformalizedQuantileRegression(0.20).calibrate(y, lo, hi)
    assert cqr.q_ < 0


def test_cqr_repairs_crossed_quantiles():
    y, p, scale = _heteroskedastic(4000, 7)
    lo, hi = p - scale, p + scale
    a = ConformalizedQuantileRegression(0.20).calibrate(y, lo, hi)
    b = ConformalizedQuantileRegression(0.20).calibrate(y, hi, lo)     # swapped
    assert a.q_ == b.q_


def test_cqr_dollar_interval_needs_no_smearing():
    y, p, scale = _heteroskedastic(4000, 8)
    cqr = ConformalizedQuantileRegression(0.20).calibrate(y, p - scale, p + scale)
    lo_d, hi_d = cqr.interval_dollars(p - scale, p + scale)
    lo_l, hi_l = cqr.interval_log(p - scale, p + scale)
    assert np.allclose(np.log(lo_d), lo_l) and np.allclose(np.log(hi_d), hi_l)


@pytest.mark.parametrize("cls", ["mondrian", "cqr"])
def test_adaptive_intervals_raise_before_calibration(cls):
    with pytest.raises(RuntimeError):
        if cls == "mondrian":
            MondrianConformal(0.2).interval_log([12.0])
        else:
            ConformalizedQuantileRegression(0.2).interval_log([11.9], [12.1])


# --------------------------------------------------------------------------
# Error parity
# --------------------------------------------------------------------------

def test_error_parity_detects_a_planted_group_bias():
    groups = np.repeat(["A", "B", "C"], 200)
    actual = RNG.lognormal(12, 0.3, 600)
    predicted = actual * np.exp(RNG.normal(0, 0.05, 600))
    predicted[groups == "C"] *= 0.85            # C systematically under-valued by 15%
    parity = error_parity(actual, predicted, groups)
    assert set(parity["group"]) == {"A", "B", "C"}
    worst = parity.iloc[0]                       # sorted by mean signed error ascending
    assert worst["group"] == "C"
    assert worst["mean_signed_error"] < -0.10
    assert parity_gap(parity) > 0.10


def test_error_parity_drops_groups_too_small_to_measure():
    groups = np.array(["big"] * 100 + ["tiny"] * 5)
    actual = RNG.lognormal(12, 0.2, 105)
    parity = error_parity(actual, actual * 1.01, groups, min_n=20)
    assert list(parity["group"]) == ["big"]


# --------------------------------------------------------------------------
# Moran's I
# --------------------------------------------------------------------------

def test_morans_i_is_near_zero_for_a_spatially_random_field():
    n = 1200
    lat, lon = RNG.uniform(42.0, 42.07, n), RNG.uniform(-93.69, -93.58, n)
    res = morans_i(RNG.normal(0, 1, n), lat, lon)
    assert abs(res["morans_i"]) < 0.10
    assert res["p_value"] > 0.01


def test_morans_i_detects_a_strong_spatial_gradient():
    n = 1200
    lat, lon = RNG.uniform(42.0, 42.07, n), RNG.uniform(-93.69, -93.58, n)
    values = 1000 * lat + RNG.normal(0, 0.05, n)
    res = morans_i(values, lat, lon)
    assert res["morans_i"] > 0.8
    assert res["p_value"] < 1e-6


def test_morans_i_ignores_missing_coordinates():
    n = 400
    lat, lon = RNG.uniform(42.0, 42.07, n), RNG.uniform(-93.69, -93.58, n)
    lat[:20] = np.nan
    assert morans_i(RNG.normal(0, 1, n), lat, lon)["n"] == n - 20
