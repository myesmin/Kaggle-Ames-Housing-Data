"""Fair-lending analysis of HMDA filings.

HMDA data allows lending outcomes to be examined by race, ethnicity and sex. Steps,
following standard compliance-analytics practice:

1. Raw disparity: denial rates by group with interval estimates (a rate on 174
   applications is far less certain than one on 20,000).
2. Adjusted disparity: logistic regression controlling for the underwriting factors
   HMDA records (income, loan size, LTV, DTI, purpose, lien, year), to see how much of
   the raw gap remains.
3. Caveat: HMDA has no credit score, assets, reserves or employment history. Credit
   score is the strongest predictor of denial and correlates with race in the US. An
   adjusted disparity here is therefore an upper bound on unexplained disparity, not a
   measurement of discrimination. Regulators fit these models with the credit file;
   the public file lacks it.

Always report step 3 alongside step 2. See ``docs/METHODOLOGY.md`` for the literature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HMDA_DECISIONED, HMDA_DENIED

#: ``denial_reason`` codes.  10 is "not applicable" and 1111 is the exemption
#: sentinel; neither is a reason.
DENIAL_REASONS: dict[int, str] = {
    1: "Debt-to-income ratio",
    2: "Employment history",
    3: "Credit history",
    4: "Collateral",
    5: "Insufficient cash",
    6: "Unverifiable information",
    7: "Credit application incomplete",
    8: "Mortgage insurance denied",
    9: "Other",
}
DENIAL_REASON_COLUMNS = ("denial_reason-1", "denial_reason-2",
                         "denial_reason-3", "denial_reason-4")

#: Ordered DTI bands.  HMDA reports some applications as a number and others as a
#: band, so everything is collapsed to the coarser representation. A model fit on the
#: mixture would pick up the reporting convention, not the borrower.
DTI_BANDS = ("<20%", "20%-<30%", "30%-<36%", "36%-<40%", "40%-<45%",
             "45%-<50%", "50%-60%", ">60%")

#: Groups too small to support an adjusted estimate are still reported in the raw
#: table (omitting them would hide results). In the model they are pooled: a category
#: with a dozen observations gives a singular design matrix or an odds ratio with a
#: 40-fold confidence interval.
MIN_GROUP_FOR_MODEL = 100
POOLED_LABEL = "Other minority (pooled)"

#: Not protected classes; these encode missingness. An applicant who declined to
#: report race is not a racial group, and including them would pull every estimate
#: toward the population mean.  They stay in the raw table,
#: flagged, and out of the model.
NOT_A_GROUP = ("Race Not Available", "Ethnicity Not Available", "Sex Not Available",
               "Free Form Text Only", "Joint")


def decisioned(hmda: pd.DataFrame) -> pd.DataFrame:
    """Applications on which the institution made a credit decision.

    Excludes withdrawn, incomplete, and ``action_taken == 6`` (loans purchased on the
    secondary market). Those are another lender's decisions; counting them pulls every
    denial rate towards zero.
    """
    return hmda[hmda["action_taken"].isin(HMDA_DECISIONED)].copy()


def dti_band(value) -> str | float:
    """Collapse HMDA's mixed numeric / banded DTI onto one ordered scale."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in DTI_BANDS:
        return text
    if text in ("Exempt", "NA", "nan", ""):
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return np.nan
    if number < 20:
        return "<20%"
    if number < 30:
        return "20%-<30%"
    if number < 36:
        return "30%-<36%"
    if number < 40:
        return "36%-<40%"
    if number < 45:
        return "40%-<45%"
    if number < 50:
        return "45%-<50%"
    if number <= 60:
        return "50%-60%"
    return ">60%"


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Used instead of the normal approximation, which is already poor at a 25% rate on
    174 observations; some groups here are much smaller.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def denial_rates(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Denial rate per group with a Wilson interval, largest group first.

    Read the rate with its interval: e.g. 24.7% with a +/- 6 point band is a signal
    that needs more data to size.
    """
    rows = []
    for name, group in df.groupby(by, observed=True):
        n = len(group)
        denials = int(group["action_taken"].eq(HMDA_DENIED).sum())
        lo, hi = _wilson(denials, n)
        rows.append({by: name, "n": n, "denials": denials,
                     "denial_rate": denials / n if n else np.nan,
                     "ci_low": lo, "ci_high": hi})
    return (pd.DataFrame(rows)
            .sort_values("n", ascending=False)
            .reset_index(drop=True))


def build_model_frame(df: pd.DataFrame, protected: str, reference: str,
                      min_group: int = MIN_GROUP_FOR_MODEL
                      ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble the design matrix for the adjusted model.

    Controls are the underwriting factors HMDA records and a lender could defend in
    an examination: ability to repay (income, DTI), exposure (loan amount, LTV), and
    product (purpose, lien, occupancy).  Year is included because credit conditions
    moved sharply across 2018-2024 and the applicant mix moved with them.

    Exclusions:

    * Rows missing any control are dropped, not imputed. Imputing a borrower's LTV
      inside a discrimination model would fabricate evidence.
    * ``NOT_A_GROUP`` categories are removed. "Race Not Available" is missing data
      with a category label.
    * Groups smaller than ``min_group`` are pooled into one bucket, which is fitted
      (so those rows still inform the controls) but not interpreted.
    """
    work = df.copy()
    work["dti"] = work["debt_to_income_ratio"].map(dti_band)
    work["log_income"] = np.log(work["income"].where(work["income"] > 0))
    work["log_loan"] = np.log(work["loan_amount"].where(work["loan_amount"] > 0))
    work["ltv"] = work["loan_to_value_ratio"].where(
        work["loan_to_value_ratio"].between(1, 200))

    work = work[~work[protected].isin(NOT_A_GROUP)]
    controls = ["log_income", "log_loan", "ltv", "dti", "loan_purpose",
                "lien_status", "occupancy_type", "activity_year"]
    work = work.dropna(subset=["log_income", "log_loan", "ltv", "dti", protected])

    # Pool after the dropna, because what matters is the group's size in the sample
    # the model sees, not in the raw file.
    sizes = work[protected].value_counts()
    small = {g for g, n in sizes.items() if n < min_group and g != reference}
    work[protected] = work[protected].where(~work[protected].isin(small), POOLED_LABEL)

    for col in ["dti", "loan_purpose", "lien_status", "occupancy_type",
                "activity_year", protected]:
        work[col] = work[col].astype(str)
    # The reference group is the model's baseline, so every odds ratio reads as
    # "relative to an otherwise identical file from the reference group".
    work[protected] = pd.Categorical(
        work[protected],
        categories=[reference] + sorted(set(work[protected]) - {reference}))

    design = pd.get_dummies(work[controls + [protected]], drop_first=True, dtype=float)
    # A dummy that is constant across the fitted rows carries no information and
    # makes the Hessian singular.  Drop it rather than let the fit fail obscurely.
    design = design.loc[:, design.nunique() > 1]
    return design, work["action_taken"].eq(HMDA_DENIED).astype(int), work[protected]


def adjusted_disparity(df: pd.DataFrame, protected: str, reference: str,
                       min_group: int = MIN_GROUP_FOR_MODEL) -> pd.DataFrame:
    """Odds of denial relative to ``reference``, holding the recorded file constant.

    Returns one row per non-reference group: the adjusted odds ratio, its 95%
    interval, the p-value, and the group's size in the fitted sample.  Groups below
    ``min_group`` are returned with ``reliable=False`` rather than dropped: the
    estimate exists but does not support inference.

    This is an upper bound on unexplained disparity, not a finding of discrimination.
    See the module docstring.
    """
    import statsmodels.api as sm

    design, denied, groups = build_model_frame(df, protected, reference, min_group)
    exog = sm.add_constant(design, has_constant="add")

    # GLM/IRLS rather than Logit/Newton.  With ~25 dummy columns and a protected group
    # of 150 the Newton Hessian is near-singular and the fit silently fails to
    # converge while still returning numbers. IRLS is the standard, numerically
    # stable way to fit a binomial GLM.
    fit = sm.GLM(denied, exog, family=sm.families.Binomial()).fit(maxiter=200)
    null = sm.GLM(denied, np.ones((len(denied), 1)),
                  family=sm.families.Binomial()).fit()
    pseudo_r2 = 1 - fit.llf / null.llf          # McFadden

    counts = groups.value_counts()
    rows = []
    for term in design.columns:
        if not term.startswith(f"{protected}_"):
            continue
        group = term[len(protected) + 1:]
        coef, se = fit.params[term], fit.bse[term]
        n = int(counts.get(group, 0))
        rows.append({
            "group": group, "n_total": n,
            "odds_ratio": float(np.exp(coef)),
            "ci_low": float(np.exp(coef - 1.96 * se)),
            "ci_high": float(np.exp(coef + 1.96 * se)),
            "p_value": float(fit.pvalues[term]),
            "reliable": n >= min_group and group != POOLED_LABEL,
        })
    out = pd.DataFrame(rows).sort_values("n_total", ascending=False)
    out.attrs["n_fitted"] = int(fit.nobs)
    out.attrs["reference"] = reference
    out.attrs["pseudo_r2"] = float(pseudo_r2)
    out.attrs["controls"] = ["log income", "log loan amount", "LTV", "DTI band",
                             "loan purpose", "lien status", "occupancy", "year"]
    out.attrs["omitted"] = ["credit score", "assets and reserves",
                            "employment history", "property condition"]
    out.attrs["converged"] = bool(getattr(fit, "converged", True))
    out["e_value"] = out["odds_ratio"].map(e_value)
    out["e_value_ci"] = [e_value_for_interval(lo, hi)
                         for lo, hi in zip(out["ci_low"], out["ci_high"])]
    return out.reset_index(drop=True)


def denial_reason_mix(df: pd.DataFrame) -> pd.DataFrame:
    """How often each reason is cited among denials.

    A denial can carry up to four reasons, so the shares sum to more than one. The
    relevant row here is Collateral: a denial on collateral is a valuation that stopped
    a loan, the mechanism notebook 08 prices in basis points, observed in real data.
    """
    denials = df[df["action_taken"].eq(HMDA_DENIED)]
    counts: dict[str, int] = {}
    for column in DENIAL_REASON_COLUMNS:
        if column not in denials.columns:
            continue
        codes = pd.to_numeric(denials[column], errors="coerce")
        for code, label in DENIAL_REASONS.items():
            counts[label] = counts.get(label, 0) + int(codes.eq(code).sum())

    total = len(denials)
    return (pd.DataFrame({"reason": list(counts), "denials": list(counts.values())})
            .assign(share=lambda d: d["denials"] / total)
            .sort_values("denials", ascending=False)
            .reset_index(drop=True))


def ltv_distribution(df: pd.DataFrame, bins: tuple[float, ...] =
                     (0, 60, 75, 80, 90, 95, 97, 200)) -> pd.DataFrame:
    """Distribution of reported LTVs.

    Notebook 08 writes every loan at 80% and lets AVM error spread true LTV around it.
    This checks that premise: in a real market the contracted LTV is already dispersed
    by product and down payment, before any valuation error.
    """
    ltv = df["loan_to_value_ratio"].where(df["loan_to_value_ratio"].between(1, 200))
    ltv = ltv.dropna()
    banded = pd.cut(ltv, bins=list(bins), right=True, include_lowest=True)
    out = (banded.value_counts().sort_index().rename("n").to_frame()
           .assign(share=lambda d: d["n"] / len(ltv)))
    out.index = out.index.astype(str)
    out.attrs["n"] = len(ltv)
    out.attrs["median"] = float(ltv.median())
    out.attrs["share_exactly_80"] = float((ltv == 80.0).mean())
    return out.reset_index(names="ltv_band")


def e_value(odds_ratio: float) -> float:
    """How strong an unmeasured confounder would have to be to explain a result away.

    HMDA has no credit score, and credit score both predicts denial and correlates
    with race in the US, which is the main objection to any adjusted disparity from
    public HMDA. The E-value quantifies how plausible that objection is.

    It is the minimum association (risk-ratio scale, with both the exposure and the
    outcome, beyond every measured control) that an unmeasured confounder would need
    to reduce the observed association to nothing. 1.3 means a weak confounder
    suffices and the result is fragile; 3.4 means the missing variable would have to be
    very strong.

    Computed on the risk-ratio approximation of the odds ratio, the standard treatment
    for outcomes this common. Apply it to the confidence bound nearest the null for the
    conservative version (the one to quote).

    VanderWeele, T.J. & Ding, P. (2017). "Sensitivity Analysis in Observational
    Research: Introducing the E-Value." *Annals of Internal Medicine* 167(4), 268-274.
    """
    rr = float(odds_ratio)
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1:                      # protective side: reflect, compute, report
        rr = 1 / rr
    if rr == 1:
        return 1.0
    return rr + np.sqrt(rr * (rr - 1))


def e_value_for_interval(ci_low: float, ci_high: float) -> float:
    """E-value for the confidence limit closest to the null.

    Two traps. Use the limit nearest 1, not the lower one (for a protective estimate
    that is the upper limit). If the interval contains 1, the answer is 1: no
    confounding is needed to explain away a result already consistent with no effect.
    Mapping ``e_value`` over ``ci_low`` would report e.g. 1.65 there, claiming
    robustness for a null finding.
    """
    lo, hi = float(ci_low), float(ci_high)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    if lo <= 1 <= hi:
        return 1.0
    return e_value(lo if lo > 1 else hi)
