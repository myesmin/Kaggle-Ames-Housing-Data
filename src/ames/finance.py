"""Investment underwriting primitives: debt, NOI, returns, and Monte Carlo.

Everything here is closed-form or a short deterministic loop, and every function is
checked in ``tests/test_finance.py`` against an independent oracle (the standard
annuity formula, ``numpy_financial``, or an algebraic identity), so no DCF number
comes from untested notebook code.

All assumption defaults live in :class:`UnderwritingAssumptions` and are sourced in
``docs/ASSUMPTIONS.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .config import PROPERTY_TAX_RATE, SEED

# --------------------------------------------------------------------------
# Debt
# --------------------------------------------------------------------------

def monthly_payment(principal: float, annual_rate: float, years: int = 30) -> float:
    """Level monthly payment on a fully-amortising fixed-rate loan.

        P * i / (1 - (1 + i)^-n),    i = annual_rate / 12,  n = 12 * years

    The zero-rate case degenerates to straight-line repayment.
    """
    n = int(round(years * 12))
    if n <= 0:
        raise ValueError("years must be positive")
    if principal == 0:
        return 0.0
    i = annual_rate / 12.0
    if abs(i) < 1e-12:
        return principal / n
    return principal * i / (1 - (1 + i) ** -n)


def amortization_schedule(principal: float, annual_rate: float,
                          years: int = 30) -> pd.DataFrame:
    """Full monthly schedule: payment, interest, principal, closing balance.

    Balance is carried forward row by row rather than recomputed from a formula, so the
    schedule is self-consistent; the final balance lands on zero to within float noise.
    """
    n = int(round(years * 12))
    pmt = monthly_payment(principal, annual_rate, years)
    i = annual_rate / 12.0

    balance = principal
    rows = []
    for month in range(1, n + 1):
        interest = balance * i
        principal_paid = pmt - interest
        balance = balance - principal_paid
        rows.append((month, pmt, interest, principal_paid, max(balance, 0.0)))

    return pd.DataFrame(rows, columns=["month", "payment", "interest",
                                       "principal", "balance"])


def remaining_balance(principal: float, annual_rate: float, years: int,
                      months_elapsed: int) -> float:
    """Closed-form outstanding balance after ``months_elapsed`` payments.

        B_m = P (1+i)^m - PMT * ((1+i)^m - 1) / i
    """
    n = int(round(years * 12))
    m = int(round(months_elapsed))
    if m <= 0:
        return float(principal)
    if m >= n:
        return 0.0
    i = annual_rate / 12.0
    pmt = monthly_payment(principal, annual_rate, years)
    if abs(i) < 1e-12:
        return float(principal - pmt * m)
    growth = (1 + i) ** m
    return float(principal * growth - pmt * (growth - 1) / i)


def annual_debt_service(principal: float, annual_rate: float, years: int = 30) -> float:
    return 12.0 * monthly_payment(principal, annual_rate, years)


# --------------------------------------------------------------------------
# Underwriting assumptions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class UnderwritingAssumptions:
    """Every input to the DCF, in one auditable place.

    Sources and sensitivity ranges: ``docs/ASSUMPTIONS.md``.  Defaults are mid-range,
    not optimistic.
    """

    # --- operations (shares of effective gross income unless noted) ---
    vacancy_rate: float = 0.07          # Ames rental vacancy; college-town turnover
    management_fee: float = 0.08        # third-party SFR property management
    maintenance_rate: float = 0.08      # repairs and turnover
    capex_reserve: float = 0.05         # roof/HVAC/appliance sinking fund
    insurance_annual: float = 2800.0    # Iowa average annual premium, dollars
    property_tax_rate: float = PROPERTY_TAX_RATE   # of market value; see config

    # --- capital structure ---
    ltv: float = 0.75                   # investor (non-owner-occupied) leverage
    mortgage_rate: float = 0.0725       # PMMS 30-yr + investor spread
    amortization_years: int = 30
    closing_cost_rate: float = 0.02     # of purchase price, buyer side

    # --- exit ---
    hold_years: int = 10
    exit_cap_rate: float = 0.065
    selling_cost_rate: float = 0.06     # brokerage + transfer at disposition

    # --- growth ---
    rent_growth: float = 0.03
    expense_growth: float = 0.03
    hpa: float = 0.03                   # house price appreciation

    def with_(self, **kwargs) -> "UnderwritingAssumptions":
        """Return a copy with fields overridden (for sensitivity and Monte Carlo)."""
        return replace(self, **kwargs)


# --------------------------------------------------------------------------
# Net operating income
# --------------------------------------------------------------------------

def net_operating_income(gross_rent_annual: float, value: float,
                         a: UnderwritingAssumptions) -> dict[str, float]:
    """Year-one NOI, itemised.

    NOI is before debt service and before income tax, by definition. Including debt in
    NOI is a common cause of wrong cap rates.

    Vacancy is applied to gross scheduled rent to get effective gross income; the
    percentage operating expenses are then taken on EGI, while insurance is a fixed
    dollar amount and property tax scales with assessed value rather than with rent.
    """
    egi = gross_rent_annual * (1 - a.vacancy_rate)
    management = egi * a.management_fee
    maintenance = egi * a.maintenance_rate
    capex = egi * a.capex_reserve
    taxes = value * a.property_tax_rate
    opex = management + maintenance + capex + taxes + a.insurance_annual
    return {
        "gross_rent": gross_rent_annual,
        "vacancy_loss": gross_rent_annual * a.vacancy_rate,
        "effective_gross_income": egi,
        "management": management,
        "maintenance": maintenance,
        "capex_reserve": capex,
        "property_tax": taxes,
        "insurance": a.insurance_annual,
        "total_opex": opex,
        "noi": egi - opex,
    }


def cap_rate(noi: float, value: float) -> float:
    """Going-in capitalisation rate: NOI / price."""
    if value <= 0:
        raise ValueError("value must be positive")
    return noi / value


def value_from_cap(noi: float, cap: float) -> float:
    """Direct-capitalisation value: NOI / cap rate.  Inverse of :func:`cap_rate`."""
    if cap <= 0:
        raise ValueError("cap rate must be positive")
    return noi / cap


def dscr(noi: float, debt_service_annual: float) -> float:
    """Debt service coverage ratio.

    Below 1.0 the property cannot pay its own mortgage from operations.  Most lenders
    underwrite investor loans to a 1.20-1.25 floor.
    """
    if debt_service_annual <= 0:
        return np.inf
    return noi / debt_service_annual


def cash_on_cash(noi: float, debt_service_annual: float, equity: float) -> float:
    """Year-one levered cash yield on invested equity."""
    if equity <= 0:
        raise ValueError("equity must be positive")
    return (noi - debt_service_annual) / equity


# --------------------------------------------------------------------------
# Internal rate of return
# --------------------------------------------------------------------------

def irr(cashflows, lo: float = -0.9999, hi: float = 10.0, tol: float = 1e-10) -> float:
    """IRR by bisection on the NPV function.

    Bisection rather than Newton because the NPV of a levered real-estate cash-flow
    stream is not always well-behaved near the root, and a failed Newton iteration
    returns a plausible-looking wrong number where bisection returns NaN.  Returns NaN
    when the stream has no sign change and therefore no IRR.
    """
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) < 2 or not (np.any(cf > 0) and np.any(cf < 0)):
        return np.nan

    def npv(rate: float) -> float:
        return float(np.sum(cf / (1 + rate) ** np.arange(len(cf))))

    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) < tol or hi - lo < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def npv(rate: float, cashflows) -> float:
    cf = np.asarray(cashflows, dtype=float)
    return float(np.sum(cf / (1 + rate) ** np.arange(len(cf))))


# --------------------------------------------------------------------------
# The full levered pro forma
# --------------------------------------------------------------------------

def underwrite(price: float, gross_rent_annual: float,
               a: UnderwritingAssumptions) -> dict[str, float]:
    """Acquire, hold for ``hold_years``, sell.  Returns the full underwriting summary.

    Cash-flow convention (annual, end-of-period, year 0 = acquisition):
      t=0   -(equity + closing costs)
      t=1..N   NOI_t - debt service
      t=N   plus net sale proceeds = exit value - selling costs - loan balance

    Exit value is taken as the *greater* of direct capitalisation of forward NOI and the
    HPA-grown price.  Cap-rate exit alone makes the return depend mostly on the exit-cap
    assumption; HPA alone ignores that a buyer prices off income.  The max is
    stress-tested in the tornado.
    """
    loan = price * a.ltv
    equity = price * (1 - a.ltv) + price * a.closing_cost_rate
    ds = annual_debt_service(loan, a.mortgage_rate, a.amortization_years)

    year1 = net_operating_income(gross_rent_annual, price, a)
    going_in_cap = cap_rate(year1["noi"], price)

    flows = [-equity]
    noi_t = year1["noi"]
    noi_by_year = []
    for t in range(1, a.hold_years + 1):
        if t > 1:
            rent_t = gross_rent_annual * (1 + a.rent_growth) ** (t - 1)
            value_t = price * (1 + a.hpa) ** (t - 1)
            grown = a.with_(insurance_annual=a.insurance_annual
                            * (1 + a.expense_growth) ** (t - 1))
            noi_t = net_operating_income(rent_t, value_t, grown)["noi"]
        noi_by_year.append(noi_t)
        flows.append(noi_t - ds)

    # Exit, at the end of the final year.
    forward_rent = gross_rent_annual * (1 + a.rent_growth) ** a.hold_years
    forward_value = price * (1 + a.hpa) ** a.hold_years
    forward_assumptions = a.with_(insurance_annual=a.insurance_annual
                                  * (1 + a.expense_growth) ** a.hold_years)
    forward_noi = net_operating_income(forward_rent, forward_value,
                                       forward_assumptions)["noi"]
    exit_value = max(value_from_cap(forward_noi, a.exit_cap_rate), forward_value)
    payoff = remaining_balance(loan, a.mortgage_rate, a.amortization_years,
                               a.hold_years * 12)
    net_proceeds = exit_value * (1 - a.selling_cost_rate) - payoff
    flows[-1] += net_proceeds

    return {
        "price": price,
        "loan": loan,
        "equity": equity,
        "annual_debt_service": ds,
        "noi_year1": year1["noi"],
        "going_in_cap_rate": going_in_cap,
        "dscr": dscr(year1["noi"], ds),
        "cash_on_cash": cash_on_cash(year1["noi"], ds, equity),
        "exit_value": exit_value,
        "loan_payoff": payoff,
        "net_sale_proceeds": net_proceeds,
        "equity_multiple": sum(f for f in flows[1:]) / equity,
        "irr": irr(flows),
        "cashflows": flows,
        "noi_by_year": noi_by_year,
    }


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------

def tornado(price: float, gross_rent_annual: float, base: UnderwritingAssumptions,
            ranges: dict[str, tuple[float, float]], metric: str = "irr") -> pd.DataFrame:
    """One-at-a-time sensitivity: how far does ``metric`` move when each input is
    swung to its low and high bound with everything else held at base?

    Sorted by absolute swing (tornado order). The ordering shows which assumptions are
    worth diligence spend.
    """
    base_value = underwrite(price, gross_rent_annual, base)[metric]
    rows = []
    for name, (low, high) in ranges.items():
        lo_v = underwrite(price, gross_rent_annual, base.with_(**{name: low}))[metric]
        hi_v = underwrite(price, gross_rent_annual, base.with_(**{name: high}))[metric]
        rows.append({
            "assumption": name, "low_input": low, "high_input": high,
            "low_output": lo_v, "high_output": hi_v, "base_output": base_value,
            "swing": abs(hi_v - lo_v),
        })
    return (pd.DataFrame(rows).sort_values("swing", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------

#: Correlation structure for the simulated drivers.  These are judgemental, not
#: estimated: five years of one city's data cannot identify a 4x4 correlation matrix.
#: They encode three economic facts: rent growth and house-price appreciation move
#: together; higher mortgage rates depress prices and push exit caps up; exit cap and
#: HPA are therefore negatively related.  Stated explicitly so each number can be
#: challenged.
DEFAULT_CORRELATION = pd.DataFrame(
    [[1.00, 0.60, -0.20, 0.10],
     [0.60, 1.00, -0.50, -0.10],
     [-0.20, -0.50, 1.00, 0.55],
     [0.10, -0.10, 0.55, 1.00]],
    index=["rent_growth", "hpa", "exit_cap_rate", "mortgage_rate"],
    columns=["rent_growth", "hpa", "exit_cap_rate", "mortgage_rate"],
)


def monte_carlo(price: float, gross_rent_annual: float, base: UnderwritingAssumptions,
                n: int = 2000, sigmas: dict[str, float] | None = None,
                correlation: pd.DataFrame | None = None,
                seed: int = SEED) -> pd.DataFrame:
    """Simulate the levered IRR under correlated shocks to the four key drivers.

    Draws are multivariate normal on the correlation matrix above, then applied as
    additive shocks to the base assumptions.  Exit cap and mortgage rate are floored at
    1% so a tail draw cannot produce a negative discount rate and an infinite exit value.
    """
    sigmas = sigmas or {"rent_growth": 0.015, "hpa": 0.025,
                        "exit_cap_rate": 0.010, "mortgage_rate": 0.0125}
    corr = correlation if correlation is not None else DEFAULT_CORRELATION
    names = list(corr.index)

    sd = np.array([sigmas[k] for k in names])
    cov = corr.to_numpy() * np.outer(sd, sd)
    rng = np.random.default_rng(seed)
    # See the note in avm.morans_i: macOS Accelerate emits spurious FP warnings from the
    # Cholesky matmul inside multivariate_normal even though cov is finite and PSD.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        draws = rng.multivariate_normal(np.zeros(len(names)), cov, size=n)

    rows = []
    for draw in draws:
        shocked = {k: getattr(base, k) + d for k, d in zip(names, draw)}
        shocked["exit_cap_rate"] = max(shocked["exit_cap_rate"], 0.01)
        shocked["mortgage_rate"] = max(shocked["mortgage_rate"], 0.01)
        res = underwrite(price, gross_rent_annual, base.with_(**shocked))
        rows.append({**shocked, "irr": res["irr"], "dscr": res["dscr"],
                     "cash_on_cash": res["cash_on_cash"],
                     "equity_multiple": res["equity_multiple"]})
    return pd.DataFrame(rows)


def summarise_simulation(sim: pd.DataFrame, metric: str = "irr") -> pd.Series:
    """Distribution summary plus the two numbers an investment committee asks for:
    probability of a negative return, and probability the property cannot cover debt."""
    s = sim[metric].dropna()
    return pd.Series({
        "n": len(s),
        "mean": s.mean(),
        "p5": s.quantile(0.05),
        "p25": s.quantile(0.25),
        "median": s.median(),
        "p75": s.quantile(0.75),
        "p95": s.quantile(0.95),
        f"P({metric} < 0)": float((s < 0).mean()),
        "P(DSCR < 1.0)": float((sim["dscr"] < 1.0).mean()),
        "P(DSCR < 1.25)": float((sim["dscr"] < 1.25).mean()),
    }, name=metric)
