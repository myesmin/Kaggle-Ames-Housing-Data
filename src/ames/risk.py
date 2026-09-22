"""Credit risk: originate a loan book off the AVM, mark it to market, stress it.

Loans are originated against AVM values, not sale prices, so valuation error flows
into LTV, then PD, then expected loss. An AVM 8% too generous on a segment writes 80% LTV
loans that are really 87% LTV. This is the mechanism behind the interagency AVM rule.
See notebook 08.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CCAR_SEVERELY_ADVERSE_HPI_SHOCK
from .finance import remaining_balance

# --------------------------------------------------------------------------
# Origination
# --------------------------------------------------------------------------

def originate(avm_value, sale_price, target_ltv: float = 0.80) -> pd.DataFrame:
    """Write loans at ``target_ltv`` of the *AVM value*, then measure true LTV.

    ``ltv_underwritten`` is what the loan file says.  ``ltv_true`` is the loan against
    the price actually paid (the best available proxy for market value). The gap
    between them is AVM error expressed as leverage.
    """
    v = np.asarray(avm_value, float)
    p = np.asarray(sale_price, float)
    loan = v * target_ltv
    return pd.DataFrame({
        "avm_value": v,
        "sale_price": p,
        "loan_amount": loan,
        "ltv_underwritten": np.full_like(v, target_ltv),
        "ltv_true": loan / p,
        "valuation_error": (v - p) / p,
    })


# --------------------------------------------------------------------------
# Mark to market
# --------------------------------------------------------------------------

def mark_to_market(book: pd.DataFrame, hpi_factor, months_seasoned,
                   mortgage_rate=0.06, amortization_years: int = 30) -> pd.DataFrame:
    """Roll the book forward: amortise the debt, index the collateral.

    Parameters
    ----------
    hpi_factor
        Cumulative house-price factor from origination to the valuation date.
    months_seasoned
        Months elapsed since origination.
    mortgage_rate
        Note rate.

    All three accept a scalar or a per-loan array. Per-loan is the realistic case:
    sales span 2006-2010, so each vintage originated at a different point on the FHFA
    path and at a different PMMS rate, and both affect its current LTV.
    """
    out = book.copy()
    n = len(out)
    factor = np.broadcast_to(np.asarray(hpi_factor, float), (n,))
    months = np.broadcast_to(np.asarray(months_seasoned, float), (n,))
    rates = np.broadcast_to(np.asarray(mortgage_rate, float), (n,))

    out["collateral_value"] = out["sale_price"] * factor
    out["balance"] = [
        remaining_balance(loan, rate, amortization_years, int(round(m)))
        for loan, rate, m in zip(out["loan_amount"], rates, months)
    ]
    out["mtm_ltv"] = out["balance"] / out["collateral_value"]
    out["equity"] = out["collateral_value"] - out["balance"]
    out["negative_equity"] = out["equity"] < 0
    return out


def negative_equity_share(book: pd.DataFrame) -> float:
    """Share of loans underwater (MTM LTV > 1); a strong leading indicator of default."""
    return float(book["negative_equity"].mean())


# --------------------------------------------------------------------------
# PD / LGD / EAD
# --------------------------------------------------------------------------

def pd_from_ltv(mtm_ltv, base_pd: float = 0.005, steepness: float = 9.0,
                midpoint: float = 1.00, ceiling: float = 0.45) -> np.ndarray:
    """Probability of default as a logistic function of mark-to-market LTV.

    A shape, not an estimate: the Ames data has no default outcomes, so this is not
    calibrated. It encodes the empirical shape of the mortgage default curve: PD sits
    near a floor while the borrower has equity, rises steeply as MTM LTV crosses 100%
    (the ruthless-default option moves into the money), and saturates well below 100%
    because many underwater borrowers keep paying.

    Parameters
    ----------
    base_pd    floor PD for a well-collateralised loan
    midpoint   LTV at which the curve is halfway to its ceiling (1.00 = zero equity)
    ceiling    maximum PD, for deeply underwater loans
    """
    ltv = np.asarray(mtm_ltv, float)
    logistic = 1.0 / (1.0 + np.exp(-steepness * (ltv - midpoint)))
    return base_pd + (ceiling - base_pd) * logistic


def lgd_from_ltv(mtm_ltv, foreclosure_cost_rate: float = 0.25,
                 floor: float = 0.0, cap: float = 0.75) -> np.ndarray:
    """Loss given default from a foreclosure-cost haircut on the collateral.

    On default the lender recovers the collateral less the cost of taking and selling
    it: legal fees, carrying costs, deferred maintenance, and the distressed-sale
    discount.  Writing the haircut as ``h``, recovery per dollar of debt is
    ``(1 - h) / LTV`` and so

        LGD = 1 - (1 - h) / MTM_LTV

    which is zero whenever the loan is sufficiently over-collateralised and rises as the
    loan approaches and passes the net collateral value.  The 25% default haircut is
    within the range typically observed for single-family REO dispositions.
    """
    ltv = np.asarray(mtm_ltv, float)
    lgd = 1.0 - (1.0 - foreclosure_cost_rate) / np.maximum(ltv, 1e-9)
    return np.clip(lgd, floor, cap)


def expected_loss(book: pd.DataFrame, foreclosure_cost_rate: float = 0.25,
                  **pd_kwargs) -> pd.DataFrame:
    """EL = PD x LGD x EAD, per loan.

    EAD is the outstanding balance. A fully-drawn amortising mortgage has no undrawn
    commitment, so no credit-conversion factor applies.
    """
    out = book.copy()
    out["pd"] = pd_from_ltv(out["mtm_ltv"], **pd_kwargs)
    out["lgd"] = lgd_from_ltv(out["mtm_ltv"], foreclosure_cost_rate)
    out["ead"] = out["balance"]
    out["expected_loss"] = out["pd"] * out["lgd"] * out["ead"]
    return out


def portfolio_summary(book: pd.DataFrame, label: str = "") -> pd.Series:
    """Book-level roll-up, in the units a credit committee uses (bps of exposure)."""
    ead = book["ead"].sum()
    el = book["expected_loss"].sum()
    return pd.Series({
        "n_loans": len(book),
        "total_ead": ead,
        "total_collateral": book["collateral_value"].sum(),
        "weighted_mtm_ltv": float(np.average(book["mtm_ltv"], weights=book["ead"])),
        "negative_equity_share": negative_equity_share(book),
        "weighted_pd": float(np.average(book["pd"], weights=book["ead"])),
        "weighted_lgd": float(np.average(book["lgd"], weights=book["ead"])),
        "expected_loss": el,
        "el_rate_bps": 1e4 * el / ead if ead else np.nan,
    }, name=label or "portfolio")


# --------------------------------------------------------------------------
# Stress
# --------------------------------------------------------------------------

def apply_hpi_shock(book: pd.DataFrame, shock: float = CCAR_SEVERELY_ADVERSE_HPI_SHOCK,
                    foreclosure_cost_rate: float = 0.25, **pd_kwargs) -> pd.DataFrame:
    """Shock collateral values and re-run the loss stack.

    The default shock is the ~30% house-price decline in the Federal Reserve's
    supervisory severely adverse scenario.  Balances are unchanged: a stress scenario
    hits the asset side, not the borrower's amortisation schedule.
    """
    out = book.copy()
    out["collateral_value"] = out["collateral_value"] * (1 + shock)
    out["mtm_ltv"] = out["balance"] / out["collateral_value"]
    out["equity"] = out["collateral_value"] - out["balance"]
    out["negative_equity"] = out["equity"] < 0
    return expected_loss(out, foreclosure_cost_rate, **pd_kwargs)


def stress_grid(book: pd.DataFrame, shocks, **kwargs) -> pd.DataFrame:
    """Loss profile across a range of house-price shocks (one row per shock)."""
    rows = []
    for s in shocks:
        stressed = apply_hpi_shock(book, s, **kwargs)
        row = portfolio_summary(stressed, label=f"{s:+.0%}")
        row["hpi_shock"] = s
        rows.append(row)
    return pd.DataFrame(rows).set_index("hpi_shock")


# --------------------------------------------------------------------------
# Valuation uncertainty as a credit input
# --------------------------------------------------------------------------

def lgd_with_valuation_uncertainty(book: pd.DataFrame, conformal_half_width,
                                   foreclosure_cost_rate: float = 0.25,
                                   confidence: str = "lower") -> np.ndarray:
    """Recompute LGD marking collateral at the *conservative end* of the AVM's own interval.

    The conformal interval is multiplicative in log space, so the lower bound of an
    80% valuation range is ``value * exp(-q)``. Marking collateral at the point estimate
    assumes the AVM has no error; marking at the lower bound is a prudent haircut.

    The difference between the two LGDs is the dollar cost of AVM uncertainty, which is
    why AVM quality-control rules are effectively credit-risk rules.

    ``conformal_half_width`` is a scalar for a constant-width interval, or one value per
    loan for an adaptive one (Mondrian), where cheap collateral carries a wider band.
    """
    q = np.asarray(conformal_half_width, float)
    if np.any(q < 0):
        raise ValueError("half-width must be non-negative")
    if q.ndim and len(q) != len(book):
        raise ValueError(f"{len(q)} half-widths for {len(book)} loans")
    sign = -1.0 if confidence == "lower" else 1.0
    adjusted_value = book["collateral_value"].to_numpy() * np.exp(sign * q)
    adjusted_ltv = book["balance"] / adjusted_value
    return lgd_from_ltv(adjusted_ltv, foreclosure_cost_rate)


def uncertainty_cost(book: pd.DataFrame, conformal_half_width,
                     foreclosure_cost_rate: float = 0.25) -> pd.Series:
    """Expected loss at the point estimate vs. at the conservative end of the AVM interval.

    With per-loan half-widths, ``conformal_half_width`` and ``interval_ratio`` report the
    exposure-weighted mean, so the row stays comparable with a constant-width one.
    """
    base_el = (book["pd"] * book["lgd"] * book["ead"]).sum()
    lgd_adj = lgd_with_valuation_uncertainty(book, conformal_half_width,
                                             foreclosure_cost_rate)
    adj_el = (book["pd"] * lgd_adj * book["ead"]).sum()
    ead = book["ead"].sum()
    q = np.broadcast_to(np.asarray(conformal_half_width, float), (len(book),))
    q_mean = float(np.average(q, weights=book["ead"])) if ead else float(q.mean())
    return pd.Series({
        "conformal_half_width": q_mean,
        "interval_ratio": float(np.exp(q_mean)),
        "el_point_estimate": base_el,
        "el_conservative": adj_el,
        "el_uplift": adj_el - base_el,
        "el_uplift_bps": 1e4 * (adj_el - base_el) / ead if ead else np.nan,
    }, name="valuation_uncertainty")


# --------------------------------------------------------------------------
# Concentration
# --------------------------------------------------------------------------

def neighborhood_risk_return(prices: pd.DataFrame, group_col: str = "Neighborhood",
                             price_col: str = "SalePrice", year_col: str = "Yr Sold",
                             min_n: int = 30) -> pd.DataFrame:
    """Annual median price growth and its volatility, per neighborhood.

    Caveat: five annual observations per neighborhood give four growth rates. A standard
    deviation of four numbers is a very noisy volatility estimate; treat the ranking as
    a hypothesis, not a risk measurement.
    """
    rows = []
    for name, g in prices.groupby(group_col):
        if len(g) < min_n:
            continue
        yearly = g.groupby(year_col)[price_col].median().sort_index()
        if len(yearly) < 3:
            continue
        growth = yearly.pct_change().dropna()
        rows.append({
            group_col: name,
            "n_sales": len(g),
            "n_years": len(yearly),
            "median_price": float(g[price_col].median()),
            "mean_annual_growth": float(growth.mean()),
            "growth_volatility": float(growth.std(ddof=1)),
            "total_growth": float(yearly.iloc[-1] / yearly.iloc[0] - 1),
        })
    return pd.DataFrame(rows).sort_values("mean_annual_growth",
                                          ascending=False).reset_index(drop=True)
