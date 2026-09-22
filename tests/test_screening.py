"""Imbalanced-classification metrics for the transaction screen."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ames.screening import (best_threshold, calibration_table, cost_optimal_threshold,
                            discrimination_scorecard, gains_table, ks_statistic,
                            threshold_sweep)

RNG = np.random.default_rng(7)


def _separable(n=4000, base_rate=0.176, separation=0.45):
    y = (RNG.random(n) < base_rate).astype(int)
    score = np.clip(RNG.beta(2, 6, n) + separation * y, 0, 1)
    return y, score


def _random_scores(n=4000, base_rate=0.176):
    y = (RNG.random(n) < base_rate).astype(int)
    return y, RNG.random(n)


# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------

def test_perfect_ranking_scores_perfectly():
    y = np.array([0] * 80 + [1] * 20)
    score = y.astype(float)
    card = discrimination_scorecard(y, score)
    assert card["ROC_AUC"] == pytest.approx(1.0)
    assert card["PR_AUC"] == pytest.approx(1.0)
    assert card["KS"] == pytest.approx(1.0)


def test_random_scores_land_at_the_no_skill_baselines():
    y, score = _random_scores()
    card = discrimination_scorecard(y, score)
    assert card["ROC_AUC"] == pytest.approx(0.5, abs=0.05)
    # The PR-AUC no-skill baseline is the positive rate, not 0.5.
    assert card["PR_AUC"] == pytest.approx(card["positive_rate"], abs=0.05)
    assert card["PR_AUC_lift"] == pytest.approx(1.0, abs=0.3)
    assert card["KS"] < 0.10


def test_pr_auc_is_the_stricter_metric_under_imbalance():
    """ROC-AUC flatters a model on an imbalanced problem; PR-AUC is why we report both."""
    y, score = _separable(separation=0.18)
    card = discrimination_scorecard(y, score)
    assert card["ROC_AUC"] > card["PR_AUC"]


def test_ks_equals_the_maximum_of_tpr_minus_fpr():
    from sklearn.metrics import roc_curve

    y, score = _separable()
    fpr, tpr, _ = roc_curve(y, score)
    assert ks_statistic(y, score)["ks"] == pytest.approx(np.max(tpr - fpr))


def test_ks_names_a_usable_threshold():
    y, score = _separable()
    ks = ks_statistic(y, score)
    assert 0 < ks["threshold"] < 1
    assert ks["tpr_at_ks"] > ks["fpr_at_ks"]


def test_brier_rewards_calibration_not_ranking():
    """Two models with identical ranking but different calibration score differently."""
    y = (RNG.random(3000) < 0.2).astype(int)
    calibrated = np.where(y == 1, 0.8, 0.1)
    overconfident = np.where(y == 1, 0.99, 0.5)   # same ranking, worse calibration
    a = discrimination_scorecard(y, calibrated)
    b = discrimination_scorecard(y, overconfident)
    assert a["ROC_AUC"] == pytest.approx(b["ROC_AUC"])
    assert a["Brier"] < b["Brier"]


# --------------------------------------------------------------------------
# Gains and lift
# --------------------------------------------------------------------------

def test_gains_table_captures_everything_by_the_last_decile():
    y, score = _separable()
    gains = gains_table(y, score)
    assert len(gains) == 10
    assert gains["cumulative_capture"].iloc[-1] == pytest.approx(1.0)
    assert gains["cumulative_volume"].iloc[-1] == pytest.approx(1.0)


def test_cumulative_capture_and_volume_are_monotone():
    y, score = _separable()
    gains = gains_table(y, score)
    assert gains["cumulative_capture"].is_monotonic_increasing
    assert gains["cumulative_volume"].is_monotonic_increasing


def test_a_useful_screen_has_top_decile_lift_above_one():
    y, score = _separable()
    assert gains_table(y, score)["lift"].iloc[0] > 2.0


def test_random_scores_have_no_lift():
    y, score = _random_scores()
    assert gains_table(y, score)["lift"].iloc[0] == pytest.approx(1.0, abs=0.5)


def test_cumulative_lift_converges_to_one():
    y, score = _separable()
    assert gains_table(y, score)["cumulative_lift"].iloc[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------

def test_recall_falls_and_precision_rises_as_the_threshold_tightens():
    y, score = _separable()
    sweep = threshold_sweep(y, score)
    assert sweep["recall"].is_monotonic_decreasing
    assert sweep["flagged_share"].is_monotonic_decreasing
    # Precision is noisy at the extreme tail, so compare the ends rather than every step.
    assert sweep["precision"].iloc[0] < sweep["precision"].dropna().iloc[-1]


def test_cost_optimal_threshold_moves_with_the_cost_ratio():
    """Expensive false negatives should widen the net; expensive false positives narrow it."""
    y, score = _separable()
    catch_everything = best_threshold(cost_optimal_threshold(y, score, 50, 1))
    be_precise = best_threshold(cost_optimal_threshold(y, score, 1, 50))
    assert catch_everything["threshold"] < be_precise["threshold"]
    assert catch_everything["recall"] > be_precise["recall"]
    assert catch_everything["flagged_share"] > be_precise["flagged_share"]


def test_expected_cost_is_the_weighted_error_count():
    y, score = _separable()
    sweep = cost_optimal_threshold(y, score, cost_false_negative=10,
                                   cost_false_positive=3)
    assert np.allclose(sweep["expected_cost"],
                       sweep["false_negatives"] * 10 + sweep["false_positives"] * 3)
    assert (sweep["true_positives"] + sweep["false_negatives"]).nunique() == 1


def test_best_threshold_actually_minimises_the_objective():
    y, score = _separable()
    sweep = cost_optimal_threshold(y, score, 10, 1)
    assert best_threshold(sweep)["expected_cost"] == sweep["expected_cost"].min()


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_a_well_calibrated_score_has_small_gaps():
    n = 20_000
    p = RNG.uniform(0.02, 0.9, n)
    y = (RNG.random(n) < p).astype(int)
    table = calibration_table(y, p)
    assert table["gap"].abs().max() < 0.05


def test_a_systematically_overconfident_score_shows_a_negative_gap():
    n = 10_000
    p_true = RNG.uniform(0.05, 0.5, n)
    y = (RNG.random(n) < p_true).astype(int)
    inflated = np.clip(p_true * 1.8, 0, 1)     # predicts ~2x too high
    assert calibration_table(y, inflated)["gap"].mean() < -0.05


def test_calibration_table_survives_a_degenerate_score():
    y = (RNG.random(500) < 0.2).astype(int)
    table = calibration_table(y, np.full(500, 0.2))
    assert len(table) >= 1
    assert table["n"].sum() == 500
