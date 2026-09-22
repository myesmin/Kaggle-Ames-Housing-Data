"""Build the static site that stands in for the live API.

Hugging Face moved Docker and Gradio Spaces behind a PRO subscription; only *static*
Spaces remain free. Instead of a paid Space or a dead badge, the site publishes the
numbers, the figures, and a few worked valuation reports.

The service is unchanged and still in the repository (``make serve`` and ``docker run``
both work); only the hosted copy is gone.

Every number on the page is read from ``data/processed/*.json``, the same summaries the
notebooks write and ``make promote`` reads. Nothing is typed in by hand, so re-running
``make site`` picks up any change.

Writes ``site/``:

    index.html        the page
    README.md         Hugging Face Space frontmatter (sdk: static)
    figures/          the subset of reports/figures/ the page embeds
    valuations/       the worked client reports from reports/valuations/
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from ames.config import (
    MDAPE_THRESHOLD,
    PPE10_THRESHOLD,
    PROCESSED,
    REPORTS,
    ROOT,
)

SITE = ROOT / "site"
ARTIFACTS = ROOT / "serving" / "artifacts"
REPO = "https://github.com/myesmin/real-estate-valuation-risk"

#: Figures the page embeds, in the order they appear.  A curated subset of the 44 in
#: reports/figures/.
FIGURES = (
    "03_first_pass_rescore.png",
    "03_regime_comparison.png",
    "04_coverage_validation.png",
    "04_adaptive_intervals.png",
    "04_decile_bias_artefact.png",
    "08_stress_curve.png",
    "08_uncertainty_cost.png",
    "07_noi_waterfall.png",
    "09_raw_denial_disparity.png",
)


# --------------------------------------------------------------------------
# Formatting.  Every number on the page goes through one of these, so units and
# precision are decided once rather than per call site.
# --------------------------------------------------------------------------
def pct(x: float | None, dp: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{dp}f}%"


def usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def bps(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.0f} bps"


def mult(x: float | None, dp: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{dp}f}×"


def load(name: str) -> dict:
    path = PROCESSED / name
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)} -- run `make run` before `make site`"
        )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# Design tokens.  Lifted from src/ames/viz.py and reports/valuations/*.html so the
# page, the charts and the client reports share one style.
#
# `--paper` is the same warm off-white the matplotlib figures are
# rendered on (viz.SURFACE).  Figure cards keep it in *both* colour schemes, so a PNG
# blends into its card instead of floating as a lit rectangle on a dark page.
# --------------------------------------------------------------------------
CSS = """
:root{
  --bg:#fcfcfb; --panel:#fff; --paper:#FBFAF7;
  --ink:#0b0b0b; --secondary:#52514e; --muted:#8a8880;
  --border:#e8e7e3; --rule:#cbcac5;
  --accent:#2a78d6; --good:#1e8f5a; --warn:#c2410c;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#1a1a19; --panel:#232322;
    --ink:#fff; --secondary:#c3c2b7; --muted:#a3a29a;
    --border:#383835; --rule:#4a4a46;
    --accent:#5a9fe8; --good:#3fbd85; --warn:#f08a52;
  }
}
*{box-sizing:border-box}
body{margin:0;padding:40px 20px 64px;background:var(--bg);color:var(--ink);
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.page{max-width:860px;margin:0 auto}
a{color:var(--accent)}
.kicker{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:8px}
h1{font-size:34px;line-height:1.15;letter-spacing:-.02em;margin:0 0 12px}
h2{font-size:22px;line-height:1.25;letter-spacing:-.01em;margin:0 0 10px}
.standfirst{font-size:17px;color:var(--secondary);margin:0 0 18px}
header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:26px}
.links{display:flex;flex-wrap:wrap;gap:18px;font-size:14px;margin-top:6px}

/* The disclaimer is the first thing on the page, not a footnote. A number this easy
   to quote is one someone will eventually paste into a decision. */
.notice{border:1px solid var(--border);border-left:3px solid var(--warn);
  background:var(--panel);border-radius:8px;padding:14px 16px;margin:0 0 34px;
  font-size:14.5px;color:var(--secondary)}
.notice strong{color:var(--ink)}

/* Exactly one hero number on the page. */
.hero{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  padding:26px 28px;margin-bottom:14px}
.hero .value{font-size:60px;font-weight:650;letter-spacing:-.03em;line-height:1;
  font-variant-numeric:tabular-nums}
.hero .label{font-size:15px;color:var(--secondary);margin-top:10px}

.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;
  margin-bottom:38px}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:10px;
  padding:16px 18px}
.tile .label{font-size:12px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);margin-bottom:7px}
.tile .value{font-size:25px;font-weight:600;letter-spacing:-.015em;
  font-variant-numeric:tabular-nums}
.tile .note{font-size:12.5px;color:var(--secondary);margin-top:5px}

section{margin:0 0 46px}
section > p{color:var(--secondary)}
section > p.lead{color:var(--ink)}

figure{margin:20px 0 0}
/* Paper in both schemes -- the PNG's own surface, so the image has no visible edge. */
figure img{display:block;width:100%;height:auto;background:var(--paper);
  border:1px solid var(--border);border-radius:10px;padding:10px}
figcaption{font-size:13px;color:var(--muted);margin-top:9px}

/* A limit on the claim just made, kept adjacent to it rather than in a footnote. */
p.caveat{font-size:13.5px;color:var(--muted);border-left:2px solid var(--border);
  padding-left:14px;margin-top:20px}

table{width:100%;border-collapse:collapse;margin:18px 0 0;font-size:14.5px}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border)}
th{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--secondary);
  border-bottom:1.5px solid var(--rule)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
/* Status is a dot plus a word. Never the colour alone, and the text keeps ink tokens. */
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;
  vertical-align:middle}
.dot.good{background:var(--good)} .dot.warn{background:var(--warn)}

ul.plain{padding-left:20px;color:var(--secondary)} ul.plain li{margin:6px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;
  margin-top:18px}
.card{display:block;background:var(--panel);border:1px solid var(--border);
  border-radius:10px;padding:16px 18px;text-decoration:none;color:inherit}
.card:hover{border-color:var(--accent)}
.card .t{font-weight:600;margin-bottom:4px}
.card .d{font-size:13.5px;color:var(--secondary)}
footer{border-top:1px solid var(--border);padding-top:20px;margin-top:10px;
  font-size:13.5px;color:var(--muted)}
@media(max-width:560px){h1{font-size:27px}.hero .value{font-size:46px}}
"""


def tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="note">{note}</div>' if note else ""
    return (f'<div class="tile"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{note_html}</div>')


def figure(name: str, caption: str) -> str:
    return (f'<figure><img src="figures/{name}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def status(ok: bool, text: str) -> str:
    return f'<span class="dot {"good" if ok else "warn"}"></span>{text}'


def build_page() -> str:
    avm = load("avm_summary.json")
    gov = load("governance_summary.json")
    conf = load("conformal.json")["temporal"]
    launch = conf["mondrian"]
    mond = load("recalibration.json")            # the bands the service issues
    credit = load("credit_summary.json")
    under = load("underwriting_summary.json")
    mon = load("monitoring_summary.json")
    metro = load("metro_summary.json")
    screen = load("screening_summary.json")
    fair = load("fairlending_summary.json")
    ad = gov["adaptive_intervals"]
    fair_black = fair["adjusted"]["black"]

    mdape = avm["mdape_temporal"]
    ppe10 = avm["ppe10_temporal"]

    scorecard = f"""
<table>
  <thead><tr>
    <th>Measure</th><th class="num">This model</th>
    <th class="num">Institutional threshold</th><th>Result</th>
  </tr></thead>
  <tbody>
    <tr><td>MdAPE (median absolute percentage error)</td>
        <td class="num">{pct(mdape, 2)}</td>
        <td class="num">&lt; {pct(MDAPE_THRESHOLD, 0)}</td>
        <td>{status(mdape < MDAPE_THRESHOLD, "Just misses")}</td></tr>
    <tr><td>PPE10 (share within &plusmn;10% of sale price)</td>
        <td class="num">{pct(ppe10)}</td>
        <td class="num">&gt; {pct(PPE10_THRESHOLD, 0)}</td>
        <td>{status(ppe10 > PPE10_THRESHOLD, "Meets")}</td></tr>
    <tr><td>Interval coverage, served range (backtest)</td>
        <td class="num">{pct(mond["backtest"]["coverage_rolling"])}</td>
        <td class="num">80.0%</td>
        <td>{status(False, "Under-covers")}</td></tr>
    <tr><td>Moran&rsquo;s I (spatial clustering of errors)</td>
        <td class="num">{gov["morans_i"]:.3f}</td>
        <td class="num">&asymp; 0</td>
        <td>{status(False, f"p = {gov['morans_p']:.0e}")}</td></tr>
  </tbody>
</table>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ames AVM &mdash; valuation, governance and investment risk</title>
<meta name="description" content="An automated valuation model for Ames, Iowa, with its
 own uncertainty, governance record and credit-risk consequences measured.">
<style>{CSS}</style></head><body><div class="page">

<header>
  <div class="kicker">Ames, Iowa &middot; {avm["universe_n"]:,} arm&rsquo;s-length sales, 2006&ndash;2010</div>
  <h1>Automated valuation, model governance and credit-risk transmission</h1>
  <p class="standfirst">An automated valuation model for Ames, Iowa, tested on later
  sales it had not seen, with a measured range around every value and its errors followed
  through to loan losses.</p>
  <div class="links">
    <a href="{REPO}">Source &amp; methodology on GitHub &rarr;</a>
    <a href="{REPO}/blob/main/docs/MODEL_VALIDATION_REPORT.md">Validation report</a>
    <a href="{REPO}/blob/main/docs/MODEL_RISK_LOG.md">Model risk log</a>
  </div>
</header>

<div class="notice"><strong>Not for lending, underwriting or any real valuation
decision.</strong> Trained on 2006&ndash;2010 sales in one small city. These values are
historical and this is a demonstration of a modelling and governance pipeline, not a
valuation service.</div>

<div class="hero">
  <div class="value">{pct(mdape)}</div>
  <div class="label">Median absolute percentage error on sales the model had not seen.
    Trained on 2006&ndash;2008, tested on 2009&ndash;2010.</div>
</div>

<div class="tiles">
  {tile("Within &plusmn;10%", pct(ppe10), "PPE10, temporal holdout")}
  {tile("RMSE", usd(avm["corrected_rmse"]), "root mean squared error")}
  {tile("Interval width", pct(mond["relative_width"]), "80% range, mean, as served")}
  {tile("Stressed loss", bps(credit["stressed_el_bps"]), f"from {bps(credit['base_el_bps'])} base")}
</div>

<section>
  <div class="kicker">Baseline</div>
  <h2>The first-pass pipeline was unmeasurable, and scored worse than the sample mean</h2>
  <p class="lead">An earlier modelling pass on this dataset (three notebooks, one
  regularised regression, one reported accuracy figure) carried ten defects, four of
  which invalidated the reported results. The most consequential assigned training
  rows into the held-out test frame by index alignment, so every reported test metric
  described data the model had already seen.</p>
  <p>None of the defects raised an error. The code ran, the diagnostics looked
  plausible, and the held-out prices had been withheld, so nothing could be checked
  against them. The De Cock (2011) publication the extract comes from still has those
  prices.
  Rescored against them, the original predictions give MdAPE
  {pct(avm["original_submission_mdape"], 1)} and RMSE
  {usd(avm["original_submission_rmse"])}, against {usd(avm["corrected_rmse"])} for the
  rebuilt pipeline. Each defect and its remediation is recorded in the
  <a href="{REPO}/blob/main/docs/MODEL_RISK_LOG.md">model risk log</a>.</p>
  {figure("03_first_pass_rescore.png",
          "Baseline predictions rescored against the withheld ground truth.")}
</section>

<section>
  <div class="kicker">Accuracy</div>
  <h2>Performance against institutional acceptance thresholds</h2>
  <p>AVMs are judged on how close most valuations land, not on variance explained. The
  two standard measures have conventional thresholds used by secondary-market buyers. Both
  are shown at their measured values on the 2009&ndash;2010 test sales.</p>
  {scorecard}
  {figure("03_regime_comparison.png",
          "Identical model specification under a shuffled split and a forward-in-time split.")}
</section>

<section>
  <div class="kicker">Uncertainty</div>
  <h2>Interval estimates, and their measured coverage</h2>
  <p class="lead">Each valuation comes with an 80% range (split-conformal, proportional
  to the value). The width is set separately for five price bands and re-fitted each
  quarter on the latest twelve months of closed sales. As served, the bands run from
  {pct(min(mond["relative_width_by_band"]), 0)} to
  {pct(max(mond["relative_width_by_band"]), 0)} of value.</p>
  <p>Calibrated once at launch, the range contained the sale price
  {pct(launch["empirical_coverage"])} of the time against the {pct(0.8, 0)} target.
  Re-fitted each quarter, it contains it {pct(mond["backtest"]["coverage_rolling"])} of
  the time. Conformal intervals assume the calibration sales resemble the ones being
  valued, and a forward-in-time test breaks that. On a shuffled split the same method
  reaches {pct(gov["conformal_coverage_random_full_resample"])}, so the shortfall comes
  from the market changing, not from the method. The model card and every API response
  report the measured figure.</p>
  {figure("04_coverage_validation.png",
          "Nominal against measured interval coverage, by regime.")}
  <p>A single half-width also fails by segment: the constant interval covers the cheapest
  sale-price decile only {pct(ad["Global"]["bottom_decile"])} of the time. Mondrian
  conformal, calibrated separately per valuation quintile, lifts that to
  {pct(ad["Mondrian · value"]["bottom_decile"])} for
  {(ad["Mondrian · value"]["relative_width"] - ad["Global"]["relative_width"]) * 100:.1f}
  points of added width. Conformalized quantile regression is the only method evaluated
  that restores the marginal promise, at {pct(ad["CQR"]["coverage"])}, for
  {pct(ad["CQR"]["relative_width"] / ad["Global"]["relative_width"] - 1, 0)} more width.
  The same quantile models without the conformal step cover
  {pct(ad["Quantile models, uncalibrated"]["coverage"])}.</p>
  {figure("04_adaptive_intervals.png",
          "Coverage by sale-price decile, and coverage against width, across interval methods.")}
</section>

<section>
  <div class="kicker">Governance</div>
  <h2>Systematic error by price band, neighbourhood and location</h2>
  <p>The 2024 interagency rule on automated valuation models asks whether a model
  produces confident estimates, resists manipulation, is tested on random samples, and
  does not discriminate. The results:</p>
  <ul class="plain">
    <li><strong>The price-band bias is mostly a measurement artefact.</strong> Grouped by
      sale price, the cheapest decile shows {pct(gov["bottom_decile_bias"])} mean signed error
      and the top {pct(gov["top_decile_bias"])}; a model unbiased by construction shows
      {pct(gov["bias_artefact_bottom"])} and {pct(gov["bias_artefact_top"])} under the same
      grouping. Grouped by the valuation a lender observes, error stays within
      &plusmn;{pct(gov["bias_by_valuation_max_abs"])}. The remaining bias is about
      {pct(gov["bottom_decile_bias"] - gov["bias_artefact_bottom"], 0)} at the cheap end and
      nil at the top; a client-report flag built on the uncorrected figure was withdrawn.</li>
    <li><strong>Accuracy varies by neighbourhood</strong> across a
      {pct(gov["neighborhood_bias_spread"])} spread in mean signed error, so valuation
      quality is conditional on location.</li>
    <li><strong>Errors cluster in space</strong> (Moran&rsquo;s I
      {gov["morans_i"]:.3f}, p&nbsp;=&nbsp;{gov["morans_p"]:.0e}). The neighbourhood
      label has not absorbed location, which is a known and unfixed weakness.</li>
  </ul>
  {figure("04_decile_bias_artefact.png",
          "Mean signed error by decile, grouped by sale price and by valuation, against an unbiased model.")}
</section>

<section>
  <div class="kicker">Credit risk</div>
  <h2>Collateral is secured against a valuation, not a price</h2>
  <p class="lead">Every loan in this constructed book is written at 80% loan-to-value on
  the model&rsquo;s valuation. Measured against what the houses actually sold for, none is
  at 80%, and {pct(credit["share_true_ltv_above_90"])} are above 90%.</p>
  <p>Under the Federal Reserve&rsquo;s severely adverse scenario (a 30% fall in house
  prices), expected losses go from {bps(credit["base_el_bps"])} to
  {bps(credit["stressed_el_bps"])}, a {mult(credit["stress_multiple"])} increase, and the
  share of borrowers underwater goes from {pct(credit["base_negative_equity"])} to
  {pct(credit["stressed_negative_equity"])}. Valuing collateral at the low end of the
  model&rsquo;s range instead of its point estimate adds a further
  {bps(credit["uncertainty_uplift_base_bps"])} at base and
  {bps(credit["uncertainty_uplift_stressed_bps"])} under stress. This link between
  valuation error and loan losses is why regulators set quality-control standards for
  AVMs.</p>
  {figure("08_stress_curve.png", "Expected loss as house prices fall.")}
  {figure("08_uncertainty_cost.png",
          "Expected loss with and without valuation uncertainty priced in.")}
</section>

<section>
  <div class="kicker">Investment</div>
  <h2>The market does not support rental underwriting at current prices</h2>
  <p class="lead">Underwriting was pre-registered against a 4&ndash;8% cap-rate band, the
  range at which residential rental acquisition is conventionally financeable. Measured
  across {under["n_properties"]:,} properties, the median cap rate is
  {pct(under["median_cap_rate"], 2)}, median DSCR {under["median_dscr"]:.2f}, and
  {pct(under["mc_p_irr_negative"])} of Monte Carlo paths terminate below zero. Only
  {pct(under["share_dscr_above_125"], 2)} of properties clear a 1.25 debt-service
  floor.</p>
  <p>The mechanism is identifiable in the NOI bridge: property tax at the Story County
  consolidated levy and insurance together absorb a large share of gross rent before
  debt service. An acquisition priced to clear 1.25 DSCR requires an entry of
  {usd(under["dscr_clearing_price"])},
  {pct(abs(under["dscr_clearing_discount"]))} below the current median value. Sensitivity
  analysis identifies <strong>{under["top_tornado_driver"].upper()}</strong> as the
  dominant driver of terminal IRR. The result is reported as measured; the
  pre-registered band was not met.</p>
  {figure("07_noi_waterfall.png", "Where the rent goes.")}
</section>

<section>
  <div class="kicker">Fair lending</div>
  <h2>Observed lending outcomes in the same market</h2>
  <p class="lead">The loan book above is constructed, which bounds what it can
  establish about credit access. This section substitutes observed data:
  {fair["n_filings"]:,} HMDA mortgage applications filed in the same MSA,
  {fair["years"][0]}&ndash;{fair["years"][1]}, in which denials, their stated reasons and
  applicant demographics are all reported.</p>
  <p>Black applicants were denied at {pct(fair["raw"]["black_denial_rate"])} against
  {pct(fair["raw"]["white_denial_rate"])} for white applicants,
  {mult(fair["raw"]["black_white_ratio"], 2)} the rate. Controlling for income, loan
  size, loan-to-value, DTI band, loan purpose, lien status, occupancy and year, the
  adjusted odds of denial remain {mult(fair_black["odds_ratio"], 2)}
  (95% CI {fair_black["ci_low"]:.2f}&ndash;{fair_black["ci_high"]:.2f},
  p&nbsp;=&nbsp;{fair_black["p_value"]:.3f}), fitted on
  {fair["adjusted"]["n_fitted"]:,} applications.</p>
  <p>HMDA records no credit score, the main missing control. Its possible effect is
  bounded with an E-value of {fair_black["e_value"]:.2f}
  ({fair_black["e_value_ci"]:.2f} at the confidence limit): an unmeasured confounder
  would have to be associated with both race and denial by that much to explain the gap
  away. Female applicants show {mult(fair["adjusted"]["female"]["odds_ratio"], 2)}: the
  same model finds no gap where none is expected.</p>
  <p>{pct(fair["collateral"]["share_of_denials"])} of denials
  ({fair["collateral"]["denials"]:,}) cite <strong>collateral</strong>: the valuation did
  not support the loan. That is the same problem priced above in basis points, seen here
  as credit that was never extended.</p>
  {figure("09_raw_denial_disparity.png", "Denial rates by applicant race, Ames HMDA.")}
  <p class="caveat">{fair["caveat"]} Full tests and caveats in
  <a href="{REPO}/blob/main/notebooks/09_fair_lending.ipynb">notebook 09</a>.</p>
</section>

<section>
  <div class="kicker">Deliverable</div>
  <h2>Worked valuation reports</h2>
  <p>The one-page report returned by the service, stating the valuation, its interval
  and the model&rsquo;s known weaknesses on the document itself.</p>
  <div class="cards">
    <a class="card" href="valuations/typical_535382100.html">
      <div class="t">A typical house</div>
      <div class="d">OldTown. A mid-priced home that sold inside its range.</div></a>
    <a class="card" href="valuations/expensive_528176010.html">
      <div class="t">An expensive house</div>
      <div class="d">NridgHt. Sold above its quoted range.</div></a>
    <a class="card" href="valuations/flagged_906385020.html">
      <div class="t">A flagged transaction</div>
      <div class="d">Screened as unlikely to be arm&rsquo;s-length.</div></a>
  </div>
</section>

<section>
  <div class="kicker">Scope</div>
  <h2>Further analysis in the repository</h2>
  <ul class="plain">
    <li><strong>Monitoring.</strong> {mon["quarters_monitored"]} quarters of drift and
      accuracy control charts. Run on the real non-arm&rsquo;s-length sales as a negative
      control, {mon["negative_control_breaches"]} of the six controls fire.</li>
    <li><strong>Transaction screening.</strong> A classifier identifying
      non-arm&rsquo;s-length sales prior to training-set inclusion
      ({screen["roc_auc"]:.2f} ROC-AUC, {mult(screen["pr_auc_lift"])} PR-AUC lift over
      the base rate).</li>
    <li><strong>Macro context.</strong> Peak-to-trough drawdown of
      {pct(abs(metro["ames_drawdown"]))} in Ames against
      {pct(abs(metro["median_drawdown"]))} for the median US metro, benchmarked across
      {metro["n_metros"]} FHFA metro series.</li>
    <li><strong>The valuation service.</strong> FastAPI, containerised, returning a
      valuation, its interval and its measured coverage per request
      (<code>make serve</code> or <code>docker run</code>). Not hosted publicly:
      Hugging Face moved Docker Spaces to a paid tier.</li>
  </ul>
</section>

<footer>
  Built from De Cock (2011) Ames housing data, FHFA and Case-Shiller house price
  indices, Freddie Mac PMMS mortgage rates, Zillow ZHVI and ZORI, and CFPB HMDA
  mortgage applications. Every source and assumption is cited in
  <a href="{REPO}/blob/main/docs/DATA_SOURCES.md">DATA_SOURCES.md</a> and
  <a href="{REPO}/blob/main/docs/ASSUMPTIONS.md">ASSUMPTIONS.md</a>.
  <br><br>
  Generated from the pipeline&rsquo;s own summary outputs on {date.today():%d %B %Y}.
  Not for lending, underwriting or any real valuation decision.
</footer>

</div></body></html>
"""


SPACE_README = """---
title: Ames AVM
emoji: 🏠
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Ames AVM — valuation, governance and investment risk

An automated valuation model for Ames, Iowa, validated out of sample and forward in
time, with its uncertainty measured, its errors traced through to credit losses, and
its governance record attached.

**Not for lending, underwriting or any real valuation decision.**

Source, methodology and validation report:
https://github.com/myesmin/real-estate-valuation-risk
"""


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "figures").mkdir(parents=True)
    (SITE / "valuations").mkdir(parents=True)

    for name in FIGURES:
        src = REPORTS / "figures" / name
        if not src.exists():
            raise SystemExit(f"missing figure {name} -- run `make run` first")
        shutil.copy2(src, SITE / "figures" / name)

    reports = sorted((REPORTS / "valuations").glob("*.html"))
    if not reports:
        raise SystemExit("no valuation reports -- run `make run` first")
    for src in reports:
        shutil.copy2(src, SITE / "valuations" / src.name)

    (SITE / "index.html").write_text(build_page())
    (SITE / "README.md").write_text(SPACE_README)

    total = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"built {SITE.relative_to(ROOT)}/  ({total / 1024:,.0f} KB)")
    print(f"  {len(FIGURES)} figures, {len(reports)} valuation reports")
    print(f"  open: file://{SITE / 'index.html'}")


if __name__ == "__main__":
    main()
