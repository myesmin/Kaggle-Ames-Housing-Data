"""Transaction screening: flagging sales that are not open-market transactions.

An AVM models the open market. Notebook 01 measured the effect of pooling in
non-arm's-length sales: foreclosures and short sales transact 15% below
comparable normal sales per square foot, family transfers 7% below, and partial sales
of unfinished construction 26% above. A model fit across all of them averages several
different price-formation processes.

Notebook 03 filters on ``Sale Condition``, an assessor field that in production is
often late, missing, or wrong. This module flags non-market transactions from their
own attributes before any label exists.

Imbalanced binary classification (17.6% positive), so plain accuracy is not used:
predicting "market sale" for everything scores 82.4%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

# --------------------------------------------------------------------------
# Discrimination
# --------------------------------------------------------------------------

def ks_statistic(y_true, y_score) -> dict[str, float]:
    """Kolmogorov-Smirnov separation, the standard credit-scorecard headline metric.

    The largest vertical gap between the cumulative distributions of scores for
    positives and negatives, and the score at which it occurs. Unlike AUC it gives a
    threshold, which an operating policy needs.

    Rough scorecard convention: below 0.20 is weak, 0.20-0.40 is usable, above 0.40 is
    strong. KS is the maximum of TPR - FPR on the ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gaps = tpr - fpr
    i = int(np.argmax(gaps))
    return {"ks": float(gaps[i]), "threshold": float(thresholds[i]),
            "tpr_at_ks": float(tpr[i]), "fpr_at_ks": float(fpr[i])}


def discrimination_scorecard(y_true, y_score, label: str = "") -> pd.Series:
    """The metrics that survive class imbalance.

    ``PR-AUC`` (average precision) is reported alongside ROC-AUC because ROC-AUC is
    optimistic under imbalance: it rewards ranking the large negative class well,
    which is easy.  The no-skill baseline for PR-AUC is the positive rate itself, so it
    is quoted here for comparison.

    ``Brier score`` measures calibration, not ranking: whether a predicted 30% actually
    happens 30% of the time.  A screen used as a *gate* only needs ranking; one whose
    output feeds another model needs calibration too.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    ks = ks_statistic(y_true, y_score)
    base_rate = float(y_true.mean())
    return pd.Series({
        "n": len(y_true),
        "positive_rate": base_rate,
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "PR_AUC": average_precision_score(y_true, y_score),
        "PR_AUC_no_skill": base_rate,
        "PR_AUC_lift": average_precision_score(y_true, y_score) / base_rate,
        "KS": ks["ks"],
        "KS_threshold": ks["threshold"],
        "Brier": brier_score_loss(y_true, y_score),
    }, name=label or "model")


# --------------------------------------------------------------------------
# Gains / lift
# --------------------------------------------------------------------------

def gains_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Decile gains table, the view a manual review queue works from.

    Rank every transaction by score, cut into deciles, and report what share of all
    true positives each decile captures.  *Lift* is that share divided by the decile's
    share of volume: a lift of 4 in the top decile means reviewing 10% of transactions
    finds 40% of the non-market ones.

    Answers "how much review capacity is needed and what will it catch?".
    """
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                       "score": np.asarray(y_score, dtype=float)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = np.minimum((np.arange(len(df)) * bins) // len(df), bins - 1) + 1

    total_pos = df["y"].sum()
    out = df.groupby("decile").agg(n=("y", "size"), positives=("y", "sum"),
                                   min_score=("score", "min"),
                                   max_score=("score", "max"))
    out["precision"] = out["positives"] / out["n"]
    out["capture_rate"] = out["positives"] / total_pos
    out["cumulative_capture"] = out["capture_rate"].cumsum()
    out["cumulative_volume"] = (out["n"].cumsum() / len(df))
    out["lift"] = out["capture_rate"] / (out["n"] / len(df))
    out["cumulative_lift"] = out["cumulative_capture"] / out["cumulative_volume"]
    return out


# --------------------------------------------------------------------------
# Operating threshold
# --------------------------------------------------------------------------

def threshold_sweep(y_true, y_score, thresholds=None) -> pd.DataFrame:
    """Precision, recall and volume across candidate cut-offs."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 97)
    rows = []
    positives = y_true.sum()
    for t in thresholds:
        flagged = y_score >= t
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        rows.append({
            "threshold": float(t),
            "flagged": int(flagged.sum()),
            "flagged_share": float(flagged.mean()),
            "true_positives": tp, "false_positives": fp,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall": tp / positives if positives else np.nan,
        })
    return pd.DataFrame(rows)


def cost_optimal_threshold(y_true, y_score, cost_false_negative: float,
                           cost_false_positive: float,
                           thresholds=None) -> pd.DataFrame:
    """Choose the cut-off by expected cost, not by maximising F1.

    Picks the threshold minimising expected cost given false-negative and
    false-positive costs. F1 assumes the two errors cost the same; here they do not:

    * A false negative lets a non-market sale into the AVM's training set, biasing
      later valuations in that neighbourhood.
    * A false positive excludes a real market sale, losing one observation out of
      thousands.

    Notebook 06 derives the costs from the measured price distortion.
    """
    sweep = threshold_sweep(y_true, y_score, thresholds)
    positives = int(np.asarray(y_true).astype(int).sum())
    sweep["false_negatives"] = positives - sweep["true_positives"]
    sweep["expected_cost"] = (sweep["false_negatives"] * cost_false_negative
                              + sweep["false_positives"] * cost_false_positive)
    return sweep


def best_threshold(sweep: pd.DataFrame, column: str = "expected_cost",
                   minimise: bool = True) -> pd.Series:
    idx = sweep[column].idxmin() if minimise else sweep[column].idxmax()
    return sweep.loc[idx]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def calibration_table(y_true, y_score, bins: int = 10) -> pd.DataFrame:
    """Predicted probability versus realised frequency, by score band."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                       "p": np.asarray(y_score, dtype=float)})
    if df["p"].nunique() <= 1:
        # A constant score has one band, not zero.  qcut with duplicates="drop" would
        # return an empty interval index and silently produce an empty table.
        df["band"] = f"[{df['p'].iloc[0]:.4f}]"
    else:
        try:
            df["band"] = pd.qcut(df["p"], bins, duplicates="drop")
        except ValueError:
            df["band"] = pd.cut(df["p"], bins)
    out = df.groupby("band", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    out["gap"] = out["observed"] - out["predicted"]
    return out.reset_index()
