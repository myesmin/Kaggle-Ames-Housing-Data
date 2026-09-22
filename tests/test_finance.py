"""Finance primitives, each checked against an independent closed-form oracle.

The oracle is either ``numpy_financial`` (a separate implementation), the textbook
annuity formula written out by hand, or an algebraic identity that must hold exactly.
A test that only checks a function against itself proves nothing.
"""
from __future__ import annotations

import numpy as np
import numpy_financial as npf
import pytest

from ames.finance import (
    UnderwritingAssumptions, amortization_schedule, annual_debt_service, cap_rate,
    cash_on_cash, dscr, irr, monte_carlo, monthly_payment, net_operating_income, npv,
    remaining_balance, tornado, underwrite, value_from_cap,
)

LOANS = [(100_000, 0.05, 30), (240_000, 0.0725, 30), (75_500, 0.0399, 15),
         (1_000_000, 0.09, 40)]


# --------------------------------------------------------------------------
# Amortisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("principal,rate,years", LOANS)
def test_monthly_payment_matches_numpy_financial(principal, rate, years):
    assert monthly_payment(principal, rate, years) == pytest.approx(
        -npf.pmt(rate / 12, years * 12, principal), rel=1e-12)


@pytest.mark.parametrize("principal,rate,years", LOANS)
def test_monthly_payment_matches_hand_written_annuity_formula(principal, rate, years):
    i, n = rate / 12, years * 12
    expected = principal * (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    assert monthly_payment(principal, rate, years) == pytest.approx(expected, rel=1e-12)


def test_zero_rate_loan_is_straight_line():
    assert monthly_payment(360_000, 0.0, 30) == pytest.approx(1000.0)


def test_schedule_amortises_to_zero_and_conserves_principal():
    p, r, y = 240_000, 0.0725, 30
    sched = amortization_schedule(p, r, y)
    assert len(sched) == y * 12
    assert sched["balance"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert sched["principal"].sum() == pytest.approx(p, rel=1e-9)
    # Every payment splits exactly into interest plus principal.
    assert np.allclose(sched["interest"] + sched["principal"], sched["payment"])
    # Interest falls monotonically, principal rises -- the defining shape of a level loan.
    assert (sched["interest"].diff().dropna() < 0).all()
    assert (sched["principal"].diff().dropna() > 0).all()


def test_total_interest_matches_numpy_financial():
    p, r, y = 240_000, 0.0725, 30
    sched = amortization_schedule(p, r, y)
    periods = np.arange(1, y * 12 + 1)
    assert sched["interest"].sum() == pytest.approx(
        -npf.ipmt(r / 12, periods, y * 12, p).sum(), rel=1e-9)


@pytest.mark.parametrize("months", [0, 1, 12, 60, 120, 359, 360, 400])
def test_closed_form_balance_matches_the_iterated_schedule(months):
    p, r, y = 240_000, 0.0725, 30
    sched = amortization_schedule(p, r, y)
    expected = p if months == 0 else sched["balance"].iloc[min(months, y * 12) - 1]
    assert remaining_balance(p, r, y, months) == pytest.approx(expected, abs=1e-6)


def test_annual_debt_service_is_twelve_payments():
    assert annual_debt_service(240_000, 0.0725, 30) == pytest.approx(
        12 * monthly_payment(240_000, 0.0725, 30))


# --------------------------------------------------------------------------
# IRR / NPV
# --------------------------------------------------------------------------

STREAMS = [
    [-1000, 300, 300, 300, 300],
    [-250_000, 12_000, 12_500, 13_000, 13_500, 310_000],
    [-50_000, -5_000, 20_000, 30_000, 40_000],
    [-100, 110],
]


@pytest.mark.parametrize("cf", STREAMS)
def test_irr_matches_numpy_financial(cf):
    assert irr(cf) == pytest.approx(npf.irr(cf), abs=1e-7)


@pytest.mark.parametrize("cf", STREAMS)
def test_irr_is_the_rate_that_zeroes_npv(cf):
    """The defining property, independent of any reference implementation."""
    assert npv(irr(cf), cf) == pytest.approx(0.0, abs=1e-5)


def test_irr_returns_nan_without_a_sign_change():
    assert np.isnan(irr([-100, -50, -25]))
    assert np.isnan(irr([100, 50, 25]))


def test_npv_at_zero_rate_is_the_plain_sum():
    cf = [-1000, 300, 300, 300, 300]
    assert npv(0.0, cf) == pytest.approx(sum(cf))


# --------------------------------------------------------------------------
# NOI, cap rate, DSCR
# --------------------------------------------------------------------------

def test_cap_rate_identity_round_trips():
    """value_from_cap is the exact inverse of cap_rate."""
    noi, value = 18_500.0, 285_000.0
    assert value_from_cap(noi, cap_rate(noi, value)) == pytest.approx(value, rel=1e-12)


def test_noi_components_sum_to_noi():
    a = UnderwritingAssumptions()
    r = net_operating_income(21_600, 300_000, a)
    assert r["effective_gross_income"] == pytest.approx(r["gross_rent"] - r["vacancy_loss"])
    assert r["total_opex"] == pytest.approx(
        r["management"] + r["maintenance"] + r["capex_reserve"]
        + r["property_tax"] + r["insurance"])
    assert r["noi"] == pytest.approx(r["effective_gross_income"] - r["total_opex"])


def test_noi_excludes_debt_service():
    """NOI must be invariant to capital structure -- the classic cap-rate error."""
    base = UnderwritingAssumptions()
    levered = base.with_(ltv=0.90, mortgage_rate=0.12)
    assert (net_operating_income(21_600, 300_000, base)["noi"]
            == pytest.approx(net_operating_income(21_600, 300_000, levered)["noi"]))


def test_property_tax_uses_the_cited_story_county_rate():
    """30.58245 per $1,000 x 47.4316% rollback = 1.4505% of market value."""
    a = UnderwritingAssumptions()
    assert net_operating_income(0, 300_000, a)["property_tax"] == pytest.approx(
        300_000 * 30.58245 / 1000 * 0.474316, rel=1e-9)


def test_dscr_and_cash_on_cash_are_consistent():
    noi, ds, equity = 18_000.0, 15_000.0, 75_000.0
    assert dscr(noi, ds) == pytest.approx(1.2)
    assert cash_on_cash(noi, ds, equity) == pytest.approx(0.04)
    # DSCR of exactly 1.0 means zero levered cash flow.
    assert cash_on_cash(15_000.0, 15_000.0, equity) == pytest.approx(0.0)


def test_dscr_is_infinite_with_no_debt():
    assert np.isinf(dscr(10_000, 0))


# --------------------------------------------------------------------------
# The full pro forma
# --------------------------------------------------------------------------

def test_underwrite_cashflow_stream_is_internally_consistent():
    a = UnderwritingAssumptions()
    r = underwrite(250_000, 24_000, a)
    cf = r["cashflows"]
    assert len(cf) == a.hold_years + 1
    assert cf[0] == pytest.approx(-r["equity"])
    # Year 1 levered cash flow is NOI less debt service, by construction.
    assert cf[1] == pytest.approx(r["noi_year1"] - r["annual_debt_service"])
    # The reported IRR really does zero this stream.
    assert npv(r["irr"], cf) == pytest.approx(0.0, abs=1e-4)


def test_higher_price_at_constant_rent_lowers_the_cap_rate():
    a = UnderwritingAssumptions()
    low = underwrite(200_000, 24_000, a)["going_in_cap_rate"]
    high = underwrite(400_000, 24_000, a)["going_in_cap_rate"]
    assert low > high


def test_loan_payoff_is_less_than_original_balance_after_holding():
    a = UnderwritingAssumptions()
    r = underwrite(250_000, 24_000, a)
    assert 0 < r["loan_payoff"] < r["loan"]


# --------------------------------------------------------------------------
# Sensitivity and simulation
# --------------------------------------------------------------------------

def test_tornado_is_sorted_by_swing_and_brackets_the_base_case():
    a = UnderwritingAssumptions()
    t = tornado(250_000, 24_000, a,
                {"rent_growth": (0.01, 0.05), "exit_cap_rate": (0.05, 0.08),
                 "mortgage_rate": (0.05, 0.09), "vacancy_rate": (0.03, 0.12)})
    assert list(t["swing"]) == sorted(t["swing"], reverse=True)
    for _, row in t.iterrows():
        lo, hi = sorted([row["low_output"], row["high_output"]])
        assert lo <= row["base_output"] + 1e-9
        assert hi >= row["base_output"] - 1e-9


def test_monte_carlo_is_reproducible_and_draws_are_correlated_as_specified():
    a = UnderwritingAssumptions()
    s1 = monte_carlo(250_000, 24_000, a, n=400, seed=7)
    s2 = monte_carlo(250_000, 24_000, a, n=400, seed=7)
    assert np.allclose(s1["irr"].fillna(-99), s2["irr"].fillna(-99))
    # Rent growth and HPA are specified as positively correlated (+0.60).
    assert s1[["rent_growth", "hpa"]].corr().iloc[0, 1] > 0.4
    # Exit cap and HPA are specified as negatively correlated (-0.50).
    assert s1[["exit_cap_rate", "hpa"]].corr().iloc[0, 1] < -0.3
    assert s1["exit_cap_rate"].min() >= 0.01


# --------------------------------------------------------------------------
# Credit-risk primitives
# --------------------------------------------------------------------------

def test_mark_to_market_accepts_per_loan_vintages():
    """Loans originated in different years sit at different points on the HPI path."""
    from ames.risk import mark_to_market, originate

    book = originate(avm_value=[200_000.0, 200_000.0], sale_price=[200_000.0, 200_000.0])
    rolled = mark_to_market(book, hpi_factor=[1.10, 0.90],
                            months_seasoned=[12, 120], mortgage_rate=[0.06, 0.05])
    assert rolled["collateral_value"].tolist() == pytest.approx([220_000.0, 180_000.0])
    # The older loan has amortised further, so it carries the smaller balance.
    assert rolled["balance"].iloc[1] < rolled["balance"].iloc[0]
    # But its collateral fell 10% while the younger loan's rose 10%, and that outweighs
    # ten years of amortisation -- which is the whole reason MTM LTV is worth computing
    # rather than inferring from seasoning.
    assert rolled["mtm_ltv"].iloc[1] > rolled["mtm_ltv"].iloc[0]
    assert np.allclose(rolled["mtm_ltv"],
                       rolled["balance"] / rolled["collateral_value"])
    assert np.allclose(rolled["equity"],
                       rolled["collateral_value"] - rolled["balance"])


def test_originate_converts_avm_error_into_ltv_error():
    """An AVM 10% high writes an 80% LTV loan that is really 88.9% LTV."""
    from ames.risk import originate

    book = originate(avm_value=[110_000.0], sale_price=[100_000.0], target_ltv=0.80)
    assert book["loan_amount"].iloc[0] == pytest.approx(88_000.0)
    assert book["ltv_true"].iloc[0] == pytest.approx(0.88)
    assert book["valuation_error"].iloc[0] == pytest.approx(0.10)


def test_lgd_is_zero_when_well_collateralised_and_rises_with_ltv():
    from ames.risk import lgd_from_ltv

    # At 25% foreclosure cost, recovery covers the loan up to LTV = 0.75.
    assert lgd_from_ltv(0.50, 0.25) == pytest.approx(0.0)
    assert lgd_from_ltv(0.75, 0.25) == pytest.approx(0.0)
    assert lgd_from_ltv(1.00, 0.25) == pytest.approx(0.25)
    assert lgd_from_ltv(1.50, 0.25) == pytest.approx(0.50)
    assert np.all(np.diff(lgd_from_ltv([0.8, 1.0, 1.2, 1.5], 0.25)) > 0)


def test_pd_curve_has_the_expected_shape():
    from ames.risk import pd_from_ltv

    ltvs = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.6])
    pds = pd_from_ltv(ltvs)
    assert np.all(np.diff(pds) > 0)                  # monotone in LTV
    assert pds[0] < 0.02                             # near the floor with equity
    assert pd_from_ltv(1.00) == pytest.approx((0.005 + 0.45) / 2, abs=1e-9)  # midpoint
    assert pds[-1] < 0.45                            # saturates below the ceiling


def test_hpi_shock_leaves_balances_untouched():
    """A stress hits the asset side, not the borrower's amortisation schedule."""
    from ames.risk import apply_hpi_shock, expected_loss, mark_to_market, originate

    book = expected_loss(mark_to_market(
        originate([250_000.0] * 5, [250_000.0] * 5), 1.05, 36))
    stressed = apply_hpi_shock(book, -0.30)
    assert np.allclose(stressed["balance"], book["balance"])
    assert np.allclose(stressed["collateral_value"], book["collateral_value"] * 0.70)
    assert stressed["expected_loss"].sum() > book["expected_loss"].sum()


def test_valuation_uncertainty_only_ever_increases_expected_loss():
    from ames.risk import expected_loss, mark_to_market, originate, uncertainty_cost

    book = expected_loss(mark_to_market(
        originate([250_000.0] * 20, [250_000.0] * 20), 0.80, 60))
    cost = uncertainty_cost(book, conformal_half_width=0.10)
    assert cost["el_conservative"] >= cost["el_point_estimate"]
    assert cost["el_uplift"] >= 0
    # A zero-width interval must be a no-op.
    assert uncertainty_cost(book, 0.0)["el_uplift"] == pytest.approx(0.0)


def test_per_loan_half_widths_match_the_constant_case_and_reject_a_mismatch():
    """An adaptive (Mondrian) interval passes one half-width per loan.  Equal widths must
    reproduce the scalar result exactly, and a wrong-length array is an error, not a
    silent broadcast."""
    import numpy as np

    from ames.risk import expected_loss, mark_to_market, originate, uncertainty_cost

    book = expected_loss(mark_to_market(
        originate([150_000.0] * 10 + [400_000.0] * 10, [140_000.0] * 10 + [410_000.0] * 10),
        0.80, 60))
    scalar = uncertainty_cost(book, 0.10)
    per_loan = uncertainty_cost(book, np.full(len(book), 0.10))
    assert per_loan["el_uplift"] == pytest.approx(scalar["el_uplift"])
    assert per_loan["conformal_half_width"] == pytest.approx(0.10)

    wider_cheap = np.r_[np.full(10, 0.15), np.full(10, 0.10)]
    assert uncertainty_cost(book, wider_cheap)["el_uplift"] >= scalar["el_uplift"]
    with pytest.raises(ValueError):
        uncertainty_cost(book, np.full(3, 0.10))
