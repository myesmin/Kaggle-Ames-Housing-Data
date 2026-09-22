"""Drift metrics, monitoring triggers, and champion/challenger promotion rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ames.monitoring import (MonitoringThresholds, PSI_SIGNIFICANT, PSI_STABLE,
                             accuracy_by_period, any_breach, categorical_stability_index,
                             champion_challenger, evaluate_triggers, feature_drift,
                             population_stability_index, promotion_decision, psi_label)

RNG = np.random.default_rng(11)


# --------------------------------------------------------------------------
# Population stability index
# --------------------------------------------------------------------------

def test_psi_of_a_distribution_against_itself_is_zero():
    x = RNG.normal(size=5000)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_small_for_two_draws_from_the_same_distribution():
    a, b = RNG.normal(size=6000), RNG.normal(size=6000)
    assert population_stability_index(a, b) < PSI_STABLE


def test_psi_increases_monotonically_with_the_size_of_the_shift():
    base = RNG.normal(size=8000)
    shifts = [0.1, 0.3, 0.6, 1.0, 2.0]
    values = [population_stability_index(base, RNG.normal(loc=s, size=8000)) for s in shifts]
    assert values == sorted(values)
    assert values[0] < PSI_STABLE
    assert values[-1] > PSI_SIGNIFICANT


def test_psi_detects_a_variance_shift_not_only_a_mean_shift():
    base = RNG.normal(size=8000)
    assert population_stability_index(base, RNG.normal(scale=2.5, size=8000)) > PSI_SIGNIFICANT


def test_psi_is_finite_when_a_bin_is_empty():
    """Without the epsilon floor an unpopulated bin sends PSI to infinity."""
    base = RNG.normal(size=4000)
    disjoint = RNG.normal(loc=12, size=4000)     # no overlap at all
    value = population_stability_index(base, disjoint)
    assert np.isfinite(value)
    assert value > PSI_SIGNIFICANT


def test_psi_of_a_constant_feature_is_zero():
    """A feature with no distribution cannot have a distributional shift."""
    assert population_stability_index(np.ones(500), np.ones(500)) == 0.0


def test_psi_handles_empty_input():
    assert np.isnan(population_stability_index([], [1, 2, 3]))


def test_categorical_stability_index_matches_a_hand_computation():
    # expected 50/50, actual 25/75 -> (0.25-0.5)ln(0.25/0.5) + (0.75-0.5)ln(0.75/0.5)
    expected = ["a"] * 500 + ["b"] * 500
    actual = ["a"] * 250 + ["b"] * 750
    hand = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert categorical_stability_index(expected, actual) == pytest.approx(hand, rel=1e-6)


def test_categorical_index_counts_a_level_that_appears_only_in_production():
    stable = categorical_stability_index(["a"] * 500 + ["b"] * 500,
                                         ["a"] * 500 + ["b"] * 500)
    novel = categorical_stability_index(["a"] * 500 + ["b"] * 500,
                                        ["a"] * 400 + ["b"] * 400 + ["c"] * 200)
    assert novel > stable


def test_psi_labels_follow_the_scorecard_convention():
    assert psi_label(0.05) == "stable"
    assert psi_label(0.15) == "moderate shift"
    assert psi_label(0.40) == "significant shift"
    assert psi_label(np.nan) == "undefined"


def test_feature_drift_ranks_the_most_shifted_feature_first():
    n = 4000
    baseline = pd.DataFrame({"steady": RNG.normal(size=n), "moved": RNG.normal(size=n),
                             "cat": RNG.choice(list("abc"), n)})
    current = pd.DataFrame({"steady": RNG.normal(size=n),
                            "moved": RNG.normal(loc=1.5, size=n),
                            "cat": RNG.choice(list("abc"), n)})
    drift = feature_drift(baseline, current, numeric=["steady", "moved"], nominal=["cat"])
    assert drift.iloc[0]["feature"] == "moved"
    assert drift.iloc[0]["status"] == "significant shift"
    assert drift.set_index("feature").loc["steady", "status"] == "stable"


# --------------------------------------------------------------------------
# Accuracy tracking
# --------------------------------------------------------------------------

def test_accuracy_by_period_tracks_a_planted_degradation():
    periods, actual, predicted = [], [], []
    for i, err in enumerate([0.02, 0.04, 0.08, 0.16]):
        a = RNG.lognormal(12, 0.3, 300)
        actual.append(a)
        predicted.append(a * np.exp(RNG.normal(0, err, 300)))
        periods.append(np.full(300, f"Q{i + 1}"))
    table = accuracy_by_period(np.concatenate(actual), np.concatenate(predicted),
                               np.concatenate(periods))
    assert len(table) == 4
    assert list(table["MdAPE"]) == sorted(table["MdAPE"])       # getting worse
    assert list(table["PPE10"]) == sorted(table["PPE10"], reverse=True)


def test_accuracy_by_period_drops_periods_too_small_to_measure():
    actual = RNG.lognormal(12, 0.2, 110)
    period = np.array(["big"] * 100 + ["tiny"] * 10)
    table = accuracy_by_period(actual, actual * 1.01, period, min_n=30)
    assert list(table["period"]) == ["big"]


# --------------------------------------------------------------------------
# Triggers
# --------------------------------------------------------------------------

def test_a_healthy_snapshot_breaches_nothing():
    healthy = {"MdAPE": 0.052, "PPE10": 0.78, "mean_signed_error": 0.003,
               "score_psi": 0.04, "max_feature_psi": 0.06, "coverage": 0.80}
    triggers = evaluate_triggers(healthy)
    assert not any_breach(triggers)
    assert len(triggers) == 6


@pytest.mark.parametrize("metric,value", [
    ("MdAPE", 0.09), ("PPE10", 0.55), ("mean_signed_error", -0.08),
    ("score_psi", 0.60), ("max_feature_psi", 0.60), ("coverage", 0.55),
])
def test_each_control_fires_independently(metric, value):
    healthy = {"MdAPE": 0.052, "PPE10": 0.78, "mean_signed_error": 0.003,
               "score_psi": 0.04, "max_feature_psi": 0.06, "coverage": 0.80}
    triggers = evaluate_triggers({**healthy, metric: value})
    assert any_breach(triggers)
    assert int(triggers["breached"].fillna(False).sum()) == 1


def test_bias_control_is_two_sided():
    """A model that reads 8% low is as broken as one that reads 8% high."""
    base = {"MdAPE": 0.05, "PPE10": 0.78, "score_psi": 0.04,
            "max_feature_psi": 0.06, "coverage": 0.80}
    assert any_breach(evaluate_triggers({**base, "mean_signed_error": 0.08}))
    assert any_breach(evaluate_triggers({**base, "mean_signed_error": -0.08}))


def test_missing_metrics_are_reported_as_unknown_not_as_passing():
    """Silence must not be read as a pass -- an un-run control is not a clean control."""
    triggers = evaluate_triggers({"MdAPE": 0.05})
    assert triggers["breached"].isna().sum() == 5
    assert not any_breach(triggers)


def test_thresholds_are_configurable():
    strict = MonitoringThresholds(mdape_limit=0.04)
    assert any_breach(evaluate_triggers({"MdAPE": 0.055}, strict))
    assert not any_breach(evaluate_triggers({"MdAPE": 0.055}))


# --------------------------------------------------------------------------
# Champion / challenger
# --------------------------------------------------------------------------

def _frames(champ, chall):
    periods = [f"P{i}" for i in range(len(champ))]
    return (pd.DataFrame({"period": periods, "MdAPE": champ}),
            pd.DataFrame({"period": periods, "MdAPE": chall}))


def test_challenger_that_wins_everywhere_is_promoted():
    champion, challenger = _frames([0.06, 0.07, 0.065, 0.062], [0.05, 0.055, 0.05, 0.051])
    comparison = champion_challenger(champion, challenger)
    assert comparison["challenger_wins"].all()
    assert promotion_decision(comparison)["promote"] is True


def test_challenger_that_is_merely_less_bad_on_average_is_not_promoted():
    """Wins big once, loses three times: better mean, worse consistency, no promotion."""
    champion, challenger = _frames([0.06, 0.06, 0.06, 0.20], [0.061, 0.062, 0.063, 0.05])
    comparison = champion_challenger(champion, challenger)
    decision = promotion_decision(comparison)
    assert decision["mean_improvement"] > 0        # better on average
    assert decision["challenger_win_rate"] == 0.25  # but wins only a quarter of the time
    assert decision["promote"] is False


def test_promotion_requires_a_positive_mean_improvement_too():
    champion, challenger = _frames([0.06, 0.06, 0.06, 0.06], [0.059, 0.059, 0.059, 0.30])
    decision = promotion_decision(champion_challenger(champion, challenger))
    assert decision["challenger_win_rate"] == 0.75
    assert decision["mean_improvement"] < 0
    assert decision["promote"] is False


def test_higher_is_better_metrics_are_handled():
    champion = pd.DataFrame({"period": ["P0", "P1"], "PPE10": [0.70, 0.72]})
    challenger = pd.DataFrame({"period": ["P0", "P1"], "PPE10": [0.78, 0.75]})
    comparison = champion_challenger(champion, challenger, metric="PPE10",
                                     lower_is_better=False)
    assert comparison["challenger_wins"].all()


# --------------------------------------------------------------------------
# Guards against manufacturing alarms
# --------------------------------------------------------------------------

def test_psi_refuses_to_answer_on_a_sample_too_small_to_measure():
    """Ten bins and eight observations is noise, and noise dressed as a number is worse
    than a gap -- most bins are empty and the epsilon floor reports a huge false shift."""
    base = RNG.normal(size=4000)
    tiny = RNG.normal(size=8)
    naive = population_stability_index(base, tiny, min_sample=0)
    assert naive > PSI_SIGNIFICANT                 # what an unguarded PSI would report
    assert np.isnan(population_stability_index(base, tiny))   # what it reports instead


def test_psi_is_available_once_the_sample_is_large_enough():
    base = RNG.normal(size=4000)
    assert np.isfinite(population_stability_index(base, RNG.normal(size=200)))


def test_the_minimum_sample_scales_with_the_bin_count():
    """Fewer bins means less resolution, not a fabricated alarm -- which is the right
    lever for a small monitoring window."""
    base = RNG.normal(size=4000)
    sample = RNG.normal(size=80)
    assert np.isnan(population_stability_index(base, sample, bins=10))   # needs 100
    assert np.isfinite(population_stability_index(base, sample, bins=5))  # needs 50


def test_feature_drift_excludes_deterministic_time_features():
    """Under a temporal split a sale date differs by construction, so monitoring it
    guarantees a permanent red light and teaches everyone to ignore the dashboard."""
    from ames.monitoring import DETERMINISTIC_TIME_FEATURES

    n = 1000
    baseline = pd.DataFrame({"sale_time": np.linspace(2006, 2008, n),
                             "Gr Liv Area": RNG.normal(1500, 400, n)})
    current = pd.DataFrame({"sale_time": np.linspace(2009, 2010, n),
                            "Gr Liv Area": RNG.normal(1500, 400, n)})

    unguarded = population_stability_index(baseline["sale_time"], current["sale_time"])
    assert unguarded > 5                            # no overlap at all -> enormous PSI

    drift = feature_drift(baseline, current, numeric=["sale_time", "Gr Liv Area"])
    assert "sale_time" in DETERMINISTIC_TIME_FEATURES
    assert list(drift["feature"]) == ["Gr Liv Area"]
    assert drift.iloc[0]["status"] == "stable"


def test_the_exclusion_list_is_overridable():
    n = 1000
    baseline = pd.DataFrame({"sale_time": np.linspace(2006, 2008, n)})
    current = pd.DataFrame({"sale_time": np.linspace(2009, 2010, n)})
    drift = feature_drift(baseline, current, numeric=["sale_time"], exclude=frozenset())
    assert list(drift["feature"]) == ["sale_time"]


def test_categorical_drift_also_respects_the_minimum_sample():
    baseline = pd.DataFrame({"cat": RNG.choice(list("abcd"), 2000)})
    tiny = pd.DataFrame({"cat": RNG.choice(list("abcd"), 5)})
    drift = feature_drift(baseline, tiny, numeric=[], nominal=["cat"])
    assert np.isnan(drift.iloc[0]["psi"])
    assert drift.iloc[0]["status"] == "undefined"
