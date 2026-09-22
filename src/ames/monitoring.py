"""Ongoing model monitoring: drift, accuracy decay, and recalibration triggers.

The Federal Reserve / OCC supervisory guidance on model risk management (SR 11-7)
defines validation as three activities:

  1. Conceptual soundness: is the design defensible?        (docs/METHODOLOGY.md)
  2. Outcomes analysis: do outputs match reality?            (notebooks 03, 04)
  3. Ongoing monitoring: is it still performing?             (this module, notebook 05)

SR 11-7 says performance changes should themselves trigger validation activity.

Contents: PSI on inputs, score stability on outputs, accuracy by vintage, and a
pre-registered set of thresholds that fire a recalibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .avm import mdape, mean_signed_error, ppe

# --------------------------------------------------------------------------
# Population stability
# --------------------------------------------------------------------------

#: Conventional PSI action thresholds from credit-risk scorecard practice.
#: < 0.10 stable · 0.10-0.25 moderate shift, investigate · > 0.25 significant, act.
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


#: PSI is not scale-free in sample size.  With ten bins and a handful of observations
#: most bins are empty, and the epsilon floor then reports a large shift that is pure
#: sampling noise.  The usual guard is roughly this many expected observations per bin;
#: below it PSI is reported as not measurable (NaN).
#:
#: Tied to the bin count so small windows can use fewer bins: a quarter with 80 sales
#: cannot support ten bins but can support five (less resolution, no false alarm).
MIN_OBSERVATIONS_PER_BIN = 10


def _min_sample(bins: int, override: int | None) -> int:
    return MIN_OBSERVATIONS_PER_BIN * bins if override is None else override


def population_stability_index(expected, actual, bins: int = 10,
                               epsilon: float = 1e-6,
                               min_sample: int | None = None) -> float:
    """Population Stability Index between a baseline and a current sample.

        PSI = sum_i (A_i - E_i) * ln(A_i / E_i)

    where ``E_i`` and ``A_i`` are the *proportions* of the expected (development) and
    actual (production) samples falling in bin ``i``.

    Bin edges come from the expected sample's deciles, so bins are frozen at
    development time and the statistic measures the new population against the old
    one rather than re-describing each sample in its own terms.

    ``epsilon`` floors empty bins; otherwise one empty bin sends PSI to infinity (a
    numerical artefact).

    Interpretation follows the credit-scorecard convention: below 0.10 the populations are
    comparable, 0.10-0.25 warrants investigation, above 0.25 is a significant shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    floor = _min_sample(bins, min_sample)
    if len(actual) < floor or len(expected) < floor:
        # Too few observations to tell a shift from noise. NaN keeps the gap visible;
        # a number here would be a false alarm from a small sample.
        return np.nan

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if len(edges) < 3:
        # A near-constant feature has no distribution to shift.
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    e_counts = np.histogram(expected, bins=edges)[0] / len(expected)
    a_counts = np.histogram(actual, bins=edges)[0] / len(actual)
    e_counts = np.clip(e_counts, epsilon, None)
    a_counts = np.clip(a_counts, epsilon, None)
    return float(np.sum((a_counts - e_counts) * np.log(a_counts / e_counts)))


def categorical_stability_index(expected, actual, epsilon: float = 1e-6) -> float:
    """PSI for a categorical feature: bins are the levels themselves.

    Levels present in only one sample still contribute, floored at ``epsilon``. A
    category seen in production but never in development is a real shift.
    """
    e = pd.Series(expected).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    levels = e.index.union(a.index)
    e = e.reindex(levels).fillna(0.0).clip(lower=epsilon)
    a = a.reindex(levels).fillna(0.0).clip(lower=epsilon)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    """Traffic-light reading of a PSI value."""
    if not np.isfinite(value):
        return "undefined"
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate shift"
    return "significant shift"


#: Features whose distribution changes *by construction* under a temporal split.
#: A sale date cannot help but differ between a 2006-2008 training window and a
#: 2009-2010 scoring window by definition of the split. Monitoring them would give a
#: permanent red light that people learn to ignore. Seasonality features are excluded
#: for the same reason: a quarter holds three months, so against a full-year baseline
#: it always looks like a shift.
DETERMINISTIC_TIME_FEATURES = frozenset({
    "sale_time", "Yr Sold", "Mo Sold", "month_sin", "month_cos",
})


def feature_drift(baseline: pd.DataFrame, current: pd.DataFrame,
                  numeric: list[str], nominal: list[str] | None = None,
                  bins: int = 10, exclude: frozenset[str] | None = None,
                  min_sample: int | None = None) -> pd.DataFrame:
    """Per-feature stability, sorted worst first.

    Called the *characteristic* stability index in scorecard practice when applied to
    inputs, versus *population* stability index on the score. Same arithmetic. Both
    are needed: the score can look stable while its inputs move in offsetting
    directions.
    """
    exclude = DETERMINISTIC_TIME_FEATURES if exclude is None else exclude
    measurable = len(current) >= _min_sample(bins, min_sample)

    rows = []
    for col in numeric:
        if col not in baseline or col not in current or col in exclude:
            continue
        psi = population_stability_index(baseline[col], current[col], bins=bins,
                                         min_sample=min_sample)
        rows.append({"feature": col, "type": "numeric", "psi": psi,
                     "status": psi_label(psi),
                     "baseline_mean": float(np.nanmean(baseline[col])),
                     "current_mean": float(np.nanmean(current[col]))})
    for col in nominal or []:
        if col not in baseline or col not in current or col in exclude:
            continue
        psi = categorical_stability_index(baseline[col], current[col]) if measurable else np.nan
        rows.append({"feature": col, "type": "nominal", "psi": psi,
                     "status": psi_label(psi),
                     "baseline_mean": np.nan, "current_mean": np.nan})
    if not rows:
        return pd.DataFrame(columns=["feature", "type", "psi", "status"])
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Accuracy over time
# --------------------------------------------------------------------------

def accuracy_by_period(actual, predicted, period, min_n: int = 30) -> pd.DataFrame:
    """AVM accuracy per production vintage.

    A single headline MdAPE hides a model that was accurate in its first live quarter
    and has degraded since.
    """
    df = pd.DataFrame({"actual": np.asarray(actual, float),
                       "predicted": np.asarray(predicted, float),
                       "period": np.asarray(period)})
    rows = []
    for name, g in df.groupby("period", sort=True):
        if len(g) < min_n:
            continue
        rows.append({
            "period": name,
            "n": len(g),
            "MdAPE": mdape(g.actual, g.predicted),
            "PPE10": ppe(g.actual, g.predicted, 0.10),
            "mean_signed_error": mean_signed_error(g.actual, g.predicted),
            "median_actual": float(g.actual.median()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Recalibration triggers
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitoringThresholds:
    """Pre-registered limits that fire a recalibration.

    Fixed in code before notebook 05 runs. Thresholds chosen after seeing monitoring
    output have the same problem as picking a model after looking at the test set.
    """

    mdape_limit: float = 0.065          # ~20% worse than the validated 5.5%
    ppe10_floor: float = 0.72           # a few points below the validated 76.6%
    bias_limit: float = 0.02            # |mean signed error| above 2% is systematic
    score_psi_limit: float = PSI_SIGNIFICANT
    feature_psi_limit: float = PSI_SIGNIFICANT
    coverage_floor: float = 0.72        # 80% intervals covering below this are broken


def evaluate_triggers(metrics: dict[str, float],
                      thresholds: MonitoringThresholds | None = None) -> pd.DataFrame:
    """Check one monitoring snapshot against every threshold.

    Returns one row per control with its observed value, its limit, and whether it
    breached. Every control is shown, not only the failures.
    """
    t = thresholds or MonitoringThresholds()
    checks = [
        ("Accuracy: MdAPE", metrics.get("MdAPE"), t.mdape_limit, "above"),
        ("Accuracy: PPE10", metrics.get("PPE10"), t.ppe10_floor, "below"),
        ("Bias: |mean signed error|",
         abs(metrics["mean_signed_error"]) if "mean_signed_error" in metrics else None,
         t.bias_limit, "above"),
        ("Score stability: PSI", metrics.get("score_psi"), t.score_psi_limit, "above"),
        ("Input stability: worst feature PSI", metrics.get("max_feature_psi"),
         t.feature_psi_limit, "above"),
        ("Interval coverage", metrics.get("coverage"), t.coverage_floor, "below"),
    ]
    rows = []
    for name, observed, limit, direction in checks:
        if observed is None or not np.isfinite(observed):
            breached = None
        else:
            breached = observed > limit if direction == "above" else observed < limit
        rows.append({"control": name, "observed": observed, "limit": limit,
                     "breach_if": direction, "breached": breached})
    return pd.DataFrame(rows)


def any_breach(triggers: pd.DataFrame) -> bool:
    """True if any control breached.  Unknown (un-run) controls do not count as breaches,
    nor as passes: ``evaluate_triggers`` reports them as NA so a missing control stays
    visible."""
    return bool(triggers["breached"].eq(True).any())


# --------------------------------------------------------------------------
# Champion / challenger
# --------------------------------------------------------------------------

def champion_challenger(champion_metrics: pd.DataFrame,
                        challenger_metrics: pd.DataFrame,
                        on: str = "period", metric: str = "MdAPE",
                        lower_is_better: bool = True) -> pd.DataFrame:
    """Compare a frozen production model against a periodically-retrained one.

    Promotion depends on whether the challenger wins consistently across vintages, not
    on one headline number. A challenger that wins on average by losing badly in three
    quarters and winning big in one is more volatile, not safer.
    """
    merged = champion_metrics.merge(challenger_metrics, on=on,
                                    suffixes=("_champion", "_challenger"))
    c, ch = f"{metric}_champion", f"{metric}_challenger"
    merged["improvement"] = (merged[c] - merged[ch]) if lower_is_better else (merged[ch] - merged[c])
    merged["challenger_wins"] = merged["improvement"] > 0
    return merged


def promotion_decision(comparison: pd.DataFrame, min_win_rate: float = 0.6,
                       min_mean_improvement: float = 0.0) -> dict[str, object]:
    """Would the challenger be promoted?  Stated as a rule, evaluated against it."""
    win_rate = float(comparison["challenger_wins"].mean())
    mean_improvement = float(comparison["improvement"].mean())
    return {
        "periods_compared": int(len(comparison)),
        "challenger_win_rate": win_rate,
        "mean_improvement": mean_improvement,
        "promote": bool(win_rate >= min_win_rate
                        and mean_improvement > min_mean_improvement),
        "rule": f"promote if the challenger wins in >= {min_win_rate:.0%} of periods "
                f"and improves the metric on average",
    }
