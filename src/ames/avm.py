"""AVM accuracy measurement, retransformation, conformal intervals, and fairness testing.

Metrics are the industry AVM ones: median absolute percentage error (MdAPE) and
percentage-of-predictions-within-N-percent (PPE10 etc.), not R-squared. MdAPE and PPE10
measure how far a single valuation is likely to be off, which is what lenders,
appraisers and regulators ask about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import CONFORMAL_ALPHA, MDAPE_THRESHOLD, PPE10_THRESHOLD, SEED

# --------------------------------------------------------------------------
# Retransformation
# --------------------------------------------------------------------------

def duan_smearing_factor(log_residuals: np.ndarray) -> float:
    """Duan (1983) smearing estimate.

    A model fit on log(price) predicts the *conditional mean of the log*, which
    exponentiates to the conditional *median* of the price, not its mean.  For
    right-skewed prices that biases every dollar figure low, more so at the top of the
    distribution.  Duan's non-parametric correction multiplies
    exp(fitted log) by the sample mean of exp(residual):

        E[y | x] = exp(x'b) * (1/n) * sum_i exp(e_i)

    Unlike the lognormal correction exp(sigma^2 / 2) it makes no distributional
    assumption about the residuals, which matters here because they are not Gaussian.

    Reference: Duan, N. (1983), "Smearing Estimate: A Nonparametric Retransformation
    Method", *JASA* 78(383), 605-610.
    """
    return float(np.mean(np.exp(np.asarray(log_residuals, dtype=float))))


def predict_dollars(log_pred: np.ndarray, smearing: float = 1.0) -> np.ndarray:
    """Convert log-space predictions to dollars, applying the smearing factor."""
    return np.exp(np.asarray(log_pred, dtype=float)) * smearing


# --------------------------------------------------------------------------
# Accuracy metrics
# --------------------------------------------------------------------------

def _arrays(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {p.shape}")
    return a, p


def percentage_error(actual, predicted) -> np.ndarray:
    """Signed error as a fraction of the sale price.  Positive = over-valued."""
    a, p = _arrays(actual, predicted)
    return (p - a) / a


def mdape(actual, predicted) -> float:
    """Median absolute percentage error, the headline AVM accuracy number."""
    return float(np.median(np.abs(percentage_error(actual, predicted))))


def ppe(actual, predicted, tolerance: float = 0.10) -> float:
    """Percentage of predictions within +/- ``tolerance`` of the sale price.

    PPE10 is the standard: the share of valuations within 10%.  Institutional AVMs are
    typically expected to clear 75%.
    """
    return float(np.mean(np.abs(percentage_error(actual, predicted)) <= tolerance))


def fsd(actual, predicted) -> float:
    """Forecast Standard Deviation proxy.

    The dispersion of the prediction-to-sale ratio, reported by AVM vendors as a
    per-valuation confidence measure.  Computed here as the standard deviation of
    log(predicted / actual), which is scale-free and symmetric in over/under-valuation.
    """
    a, p = _arrays(actual, predicted)
    return float(np.std(np.log(p / a), ddof=1))


def mean_signed_error(actual, predicted) -> float:
    """Average signed percentage error.  Non-zero means systematic over/under-valuation."""
    return float(np.mean(percentage_error(actual, predicted)))


def rmse(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def scorecard(actual, predicted, label: str = "") -> pd.Series:
    """The full AVM scorecard, as a single labelled row."""
    a, p = _arrays(actual, predicted)
    return pd.Series({
        "n": len(a),
        "MdAPE": mdape(a, p),
        "PPE5": ppe(a, p, 0.05),
        "PPE10": ppe(a, p, 0.10),
        "PPE20": ppe(a, p, 0.20),
        "PPE25": ppe(a, p, 0.25),
        "FSD": fsd(a, p),
        "mean_signed_error": mean_signed_error(a, p),
        "RMSE": rmse(a, p),
        "RMSE_log": rmse(np.log(a), np.log(p)),
        "R2": 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2),
    }, name=label or "model")


def passes_industry_thresholds(card: pd.Series) -> dict[str, bool]:
    """Check a scorecard against the institutional-grade thresholds in config."""
    return {
        f"MdAPE < {MDAPE_THRESHOLD:.0%}": bool(card["MdAPE"] < MDAPE_THRESHOLD),
        f"PPE10 > {PPE10_THRESHOLD:.0%}": bool(card["PPE10"] > PPE10_THRESHOLD),
    }


# --------------------------------------------------------------------------
# Split-conformal prediction intervals
# --------------------------------------------------------------------------

class ConformalValuation:
    """Split-conformal prediction intervals around a log-price AVM.

    Used instead of the OLS prediction interval, which requires homoskedastic Gaussian
    residuals; these are not (Model Risk Log, defect #5).  Split conformal makes no distributional
    assumption at all.  It only requires that the calibration rows be exchangeable with
    the rows being predicted, and in exchange it guarantees marginal coverage of at
    least 1 - alpha in finite samples.

    Method (Lei et al. 2018, "Distribution-Free Predictive Inference for Regression"):
      1. Fit the model on a proper training subset.
      2. On a held-out calibration subset, compute absolute residuals in log space.
      3. Take q = the ceil((n+1)(1-alpha))/n empirical quantile of those residuals.
      4. The interval for a new row is exp(pred +/- q), i.e. a *multiplicative* band
         in dollars, since valuation error scales with price.

    The finite-sample correction in step 3 makes the guarantee exact rather than
    asymptotic; without it small calibration sets under-cover.
    """

    def __init__(self, alpha: float = CONFORMAL_ALPHA):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.q_: float | None = None
        self.n_calibration_: int | None = None

    def calibrate(self, log_actual, log_pred) -> "ConformalValuation":
        """Fit the interval half-width from calibration-set residuals."""
        resid = np.abs(np.asarray(log_actual, float) - np.asarray(log_pred, float))
        n = len(resid)
        if n < 2:
            raise ValueError("need at least 2 calibration points")
        self.q_ = _conformal_quantile(resid, self.alpha)
        self.n_calibration_ = n
        return self

    def _check(self) -> float:
        if self.q_ is None:
            raise RuntimeError("call calibrate() first")
        return self.q_

    def interval_log(self, log_pred) -> tuple[np.ndarray, np.ndarray]:
        q = self._check()
        lp = np.asarray(log_pred, float)
        return lp - q, lp + q

    def interval_dollars(self, log_pred, smearing: float = 1.0):
        """Lower and upper dollar bounds.  Width is a constant *ratio*, not a constant $."""
        lo, hi = self.interval_log(log_pred)
        return predict_dollars(lo, smearing), predict_dollars(hi, smearing)

    @property
    def relative_width(self) -> float:
        """Interval width as a fraction of the point estimate: exp(q) - exp(-q)."""
        q = self._check()
        return float(np.exp(q) - np.exp(-q))

    def coverage(self, log_actual, log_pred) -> float:
        """Empirical share of held-out rows falling inside the interval."""
        lo, hi = self.interval_log(log_pred)
        a = np.asarray(log_actual, float)
        return float(np.mean((a >= lo) & (a <= hi)))


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected conformal quantile, shared by every variant below."""
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


class MondrianConformal:
    """Group-conditional split conformal: one half-width per group instead of one overall.

    ``ConformalValuation`` guarantees marginal coverage (80% across the whole book) and
    nothing about any slice. On this data the global interval covers the middle of the
    price distribution at 80-90% and the cheapest decile at under 50%: one quantile
    cannot fit both where the model is sharp and where it is not.

    Mondrian conformal (Vovk et al. 2003; Vovk, Gammerman & Shafer, *Algorithmic Learning
    in a Random World*, ch. 4) calibrates each group separately, which extends the
    finite-sample guarantee to hold within every group, under the same exchangeability
    assumption.

    The grouping must be computable when a valuation is issued, so it cannot use sale
    price (the quantity being estimated). The default bins by the model's predicted
    value, with edges learned from the calibration set.  An explicit ``groups`` array (neighbourhood,
    say) can be passed instead.

    Groups with fewer than ``min_group`` calibration rows fall back to the global
    half-width. A 90th-percentile residual from twelve houses is noise and would give
    an overconfident interval.
    """

    def __init__(self, alpha: float = CONFORMAL_ALPHA, n_bins: int = 5,
                 min_group: int = 30):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if n_bins < 1:
            raise ValueError("n_bins must be at least 1")
        self.alpha = alpha
        self.n_bins = n_bins
        self.min_group = min_group
        self.edges_: np.ndarray | None = None
        self.q_by_group_: dict | None = None
        self.q_global_: float | None = None
        self.n_by_group_: dict | None = None

    def _groups(self, log_pred, groups) -> np.ndarray:
        if groups is not None:
            return np.asarray(groups)
        if self.edges_ is None:
            raise RuntimeError("call calibrate() first")
        # Interior edges only, so values beyond the calibration range land in the end bins.
        return np.digitize(np.asarray(log_pred, float), self.edges_[1:-1])

    def calibrate(self, log_actual, log_pred, groups=None) -> "MondrianConformal":
        log_actual = np.asarray(log_actual, float)
        log_pred = np.asarray(log_pred, float)
        if len(log_actual) < 2:
            raise ValueError("need at least 2 calibration points")
        if groups is None:
            self.edges_ = np.quantile(log_pred, np.linspace(0, 1, self.n_bins + 1))
        g = self._groups(log_pred, groups)
        resid = np.abs(log_actual - log_pred)

        self.q_global_ = _conformal_quantile(resid, self.alpha)
        self.q_by_group_, self.n_by_group_ = {}, {}
        for name in np.unique(g):
            mask = g == name
            self.n_by_group_[name] = int(mask.sum())
            self.q_by_group_[name] = (_conformal_quantile(resid[mask], self.alpha)
                                      if mask.sum() >= self.min_group else self.q_global_)
        return self

    def band(self, log_pred) -> np.ndarray:
        """0-based valuation band of each prediction (value-binned calibrators only)."""
        if self.edges_ is None:
            raise RuntimeError("band() needs a calibrator binned on the valuation")
        return self._groups(log_pred, None)

    def half_width(self, log_pred, groups=None) -> np.ndarray:
        """Per-row half-width in log space; unseen groups get the global value."""
        if self.q_by_group_ is None:
            raise RuntimeError("call calibrate() first")
        g = self._groups(log_pred, groups)
        return np.array([self.q_by_group_.get(name, self.q_global_) for name in g])

    def interval_log(self, log_pred, groups=None) -> tuple[np.ndarray, np.ndarray]:
        lp = np.asarray(log_pred, float)
        q = self.half_width(lp, groups)
        return lp - q, lp + q

    def interval_dollars(self, log_pred, groups=None, smearing: float = 1.0):
        lo, hi = self.interval_log(log_pred, groups)
        return predict_dollars(lo, smearing), predict_dollars(hi, smearing)

    def coverage(self, log_actual, log_pred, groups=None) -> float:
        lo, hi = self.interval_log(log_pred, groups)
        a = np.asarray(log_actual, float)
        return float(np.mean((a >= lo) & (a <= hi)))

    # -- persistence -------------------------------------------------------
    # Serving reads the calibrated bands from the model card rather than re-deriving
    # them, and rebuilds this object from it, so the API assigns bands with the same
    # code the notebook validated.

    def to_dict(self) -> dict:
        """JSON-safe state of a value-binned calibrator."""
        if self.edges_ is None or self.q_by_group_ is None:
            raise RuntimeError("only a calibrated, value-binned instance can be saved")
        bands = range(len(self.edges_) - 1)
        return {
            "alpha": self.alpha,
            "min_group": self.min_group,
            "edges_log": [float(e) for e in self.edges_],
            "q_by_band": [float(self.q_by_group_.get(b, self.q_global_)) for b in bands],
            "n_by_band": [int(self.n_by_group_.get(b, 0)) for b in bands],
            "q_global": float(self.q_global_),
        }

    @classmethod
    def from_dict(cls, state: dict) -> "MondrianConformal":
        edges = np.asarray(state["edges_log"], float)
        obj = cls(alpha=state["alpha"], n_bins=len(edges) - 1, min_group=state["min_group"])
        obj.edges_ = edges
        obj.q_by_group_ = dict(enumerate(state["q_by_band"]))
        obj.n_by_group_ = dict(enumerate(state["n_by_band"]))
        obj.q_global_ = state["q_global"]
        return obj


class ConformalizedQuantileRegression:
    """CQR: conformal calibration of a pair of quantile models (Romano, Patterson & Candès,
    "Conformalized Quantile Regression", NeurIPS 2019).

    Mondrian conformal lets width vary between a handful of groups.  CQR lets it vary per
    property: two models estimate the conditional alpha/2 and 1-alpha/2 quantiles of log
    price directly, so a house the model finds hard to place gets a wide band and a
    typical one gets a narrow band. Quantile models alone carry no coverage guarantee
    (they are fitted, not calibrated, and often under-cover out of sample), so a
    conformal step corrects them:

      1. Score each calibration row by how far it falls outside its quantile band,
         E = max(lo - y, y - hi).  Negative when the row is inside.
      2. Take the conformal quantile of E.
      3. Widen (or, if the band was too cautious, narrow) every band by that amount.

    This restores the same marginal guarantee as ``ConformalValuation`` while keeping the
    per-property shape the quantile models learned.

    Working in log space needs no smearing correction.  Quantiles are equivariant under
    monotone transforms, so exp(a quantile of log price) is that quantile of price. The
    mean is not, which is why ``duan_smearing_factor`` exists.
    """

    def __init__(self, alpha: float = CONFORMAL_ALPHA):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.q_: float | None = None

    @staticmethod
    def _ordered(lo, hi) -> tuple[np.ndarray, np.ndarray]:
        # Independently fitted quantile models can cross.  Ordering the pair is the
        # standard repair and cannot reduce coverage.
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return np.minimum(lo, hi), np.maximum(lo, hi)

    def calibrate(self, log_actual, log_lo, log_hi) -> "ConformalizedQuantileRegression":
        y = np.asarray(log_actual, float)
        lo, hi = self._ordered(log_lo, log_hi)
        if len(y) < 2:
            raise ValueError("need at least 2 calibration points")
        self.q_ = _conformal_quantile(np.maximum(lo - y, y - hi), self.alpha)
        return self

    def interval_log(self, log_lo, log_hi) -> tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("call calibrate() first")
        lo, hi = self._ordered(log_lo, log_hi)
        return lo - self.q_, hi + self.q_

    def interval_dollars(self, log_lo, log_hi):
        lo, hi = self.interval_log(log_lo, log_hi)
        return np.exp(lo), np.exp(hi)

    def coverage(self, log_actual, log_lo, log_hi) -> float:
        lo, hi = self.interval_log(log_lo, log_hi)
        a = np.asarray(log_actual, float)
        return float(np.mean((a >= lo) & (a <= hi)))


def split_calibration(index, calib_fraction: float = 0.5, seed: int = SEED):
    """Split training rows into proper-train and calibration halves for conformal."""
    proper, calib = train_test_split(np.asarray(index), test_size=calib_fraction,
                                     random_state=seed)
    return proper, calib


# --------------------------------------------------------------------------
# Fairness / error-parity testing
# --------------------------------------------------------------------------

def error_parity(actual, predicted, group, min_n: int = 20) -> pd.DataFrame:
    """Per-group AVM accuracy, for the nondiscrimination factor of the AVM rule.

    The interagency AVM quality-control rule requires institutions to design AVM
    programmes to comply with applicable nondiscrimination law.  There is no protected-
    class attribute in the Ames data, so this is not a fair-lending test. It checks
    whether accuracy is uniform across neighborhoods and price segments: an AVM that is
    systematically pessimistic in some geographies passes that bias into every
    downstream LTV and credit decision, which is what the rule targets.

    Groups with fewer than ``min_n`` observations are dropped: a MdAPE on eight houses is
    noise (same problem as the original project's 103-row test set).
    """
    df = pd.DataFrame({
        "actual": np.asarray(actual, float),
        "predicted": np.asarray(predicted, float),
        "group": np.asarray(group),
    })
    rows = []
    for name, g in df.groupby("group"):
        if len(g) < min_n:
            continue
        rows.append({
            "group": name,
            "n": len(g),
            "MdAPE": mdape(g.actual, g.predicted),
            "PPE10": ppe(g.actual, g.predicted, 0.10),
            "mean_signed_error": mean_signed_error(g.actual, g.predicted),
            "median_price": float(g.actual.median()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("mean_signed_error").reset_index(drop=True)


def parity_gap(parity: pd.DataFrame, column: str = "mean_signed_error") -> float:
    """Spread between the best- and worst-served group (headline disparity number)."""
    return float(parity[column].max() - parity[column].min())


# --------------------------------------------------------------------------
# Spatial diagnostics
# --------------------------------------------------------------------------

def morans_i(values, lat, lon, k: int = 8) -> dict[str, float]:
    """Global Moran's I on a k-nearest-neighbour spatial weights matrix.

    Tests whether residuals cluster in space.  A hedonic model is supposed to have
    absorbed location through Neighborhood and the distance features; if the residuals
    still cluster, the model is missing location value. That matters for a lender
    because the errors are then correlated where the collateral is correlated.

    Returns the statistic, its expectation under the null of no spatial association,
    and a normal-approximation z-score and two-sided p-value.
    """
    v = np.asarray(values, float)
    coords = np.column_stack([np.asarray(lat, float), np.asarray(lon, float)])
    ok = np.isfinite(v) & np.isfinite(coords).all(axis=1)
    v, coords = v[ok], coords[ok]
    n = len(v)
    if n <= k + 1:
        raise ValueError(f"need more than k+1={k + 1} points, got {n}")

    # Row-standardised k-NN weights.  Equirectangular projection is accurate at the
    # scale of one city and lets a KD-tree do the neighbour search.
    from scipy.spatial import cKDTree

    lat0 = np.radians(coords[:, 0].mean())
    xy = np.column_stack([coords[:, 1] * np.cos(lat0), coords[:, 0]])
    # query k+1 because the first neighbour of every point is itself.
    nn = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]

    W = np.zeros((n, n))
    np.put_along_axis(W, nn, 1.0 / k, axis=1)

    z = v - v.mean()
    # NumPy on macOS Accelerate raises a spurious divide-by-zero FP warning for large
    # matmuls on entirely finite input; W is dense, finite and row-stochastic here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        num = z @ W @ z
        s1 = 0.5 * ((W + W.T) ** 2).sum()
    den = (z ** 2).sum()
    I = float(n / W.sum() * num / den)

    expected = -1.0 / (n - 1)
    # Variance of I under the normality assumption (Cliff & Ord).
    s0 = W.sum()
    s2 = ((W.sum(1) + W.sum(0)) ** 2).sum()
    var = ((n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
            - (z ** 4).sum() / ((z ** 2).sum() / n) ** 2 / n
            * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * s0 ** 2))
           / ((n - 1) * (n - 2) * (n - 3) * s0 ** 2) - expected ** 2)

    from scipy import stats
    zscore = (I - expected) / np.sqrt(var) if var > 0 else np.nan
    return {
        "morans_i": I,
        "expected": expected,
        "z_score": float(zscore),
        "p_value": float(2 * (1 - stats.norm.cdf(abs(zscore)))) if np.isfinite(zscore) else np.nan,
        "n": n,
        "k": k,
    }
