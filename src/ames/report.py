"""Client-facing valuation reports.

For non-analyst readers (credit officer, homeowner, acquisitions manager). Renders one
property as a self-contained HTML page (no external CSS, JavaScript or network calls)
that can be emailed, printed, or embedded in a loan file.

Design rules:

1. Put the range next to the estimate, in dollars. A single number reads as exact, and
   notebook 04 measured how far from exact it is.
2. State the uncertainty in words. A non-analyst can act on "the 80% range is X to Y";
   "FSD 0.103" means nothing to them.
3. Show the comparable sales, so the reader can check the reasoning.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS
from .features import haversine_miles

# --------------------------------------------------------------------------
# Comparable selection
# --------------------------------------------------------------------------

def find_comparables(subject: pd.Series, candidates: pd.DataFrame,
                     n: int = 5, size_tolerance: float = 0.25,
                     max_miles: float = 1.5) -> pd.DataFrame:
    """Appraiser-style comparable selection, in the order a human would apply it.

    Not a nearest-neighbour search in scaled feature space: a client report must be able
    to explain why each comparable was chosen, which a 237-dimensional distance cannot.
    The rules are a residential appraiser's, and each can be read off the output table:

      1. Same neighbourhood, or within ``max_miles``.
      2. Living area within ``size_tolerance`` of the subject.
      3. Rank what survives by a similarity score over size, age and quality.

    Filters relax in order if too few properties survive, and the report states which
    filters were relaxed rather than silently widening.
    """
    pool = candidates[candidates.index != subject.name].copy()
    if pool.empty:
        return pool

    relaxations: list[str] = []

    if {"Latitude", "Longitude"} <= set(pool.columns) and np.isfinite(subject.get("Latitude", np.nan)):
        pool["miles_away"] = haversine_miles(subject["Latitude"], subject["Longitude"],
                                             pool["Latitude"], pool["Longitude"])
    else:
        pool["miles_away"] = np.nan

    near = pool[(pool["Neighborhood"] == subject["Neighborhood"])
                | (pool["miles_away"] <= max_miles)]
    if len(near) < n:
        relaxations.append("location")
        near = pool

    sqft = subject["Gr Liv Area"]
    sized = near[near["Gr Liv Area"].between(sqft * (1 - size_tolerance),
                                             sqft * (1 + size_tolerance))]
    if len(sized) < n:
        relaxations.append("size")
        sized = near

    # Similarity: normalised absolute differences, lower is more comparable.
    sized = sized.copy()
    sized["similarity"] = (
        (sized["Gr Liv Area"] - sqft).abs() / max(sqft, 1) * 2.0
        + (sized["Overall Qual"] - subject["Overall Qual"]).abs() / 10 * 1.5
        + (sized["Year Built"] - subject["Year Built"]).abs() / 100 * 1.0
        + sized["miles_away"].fillna(sized["miles_away"].median() if
                                     sized["miles_away"].notna().any() else 0) * 0.5
    )
    out = sized.nsmallest(n, "similarity")
    out.attrs["relaxations"] = relaxations
    return out


# --------------------------------------------------------------------------
# Risk flags
# --------------------------------------------------------------------------

@dataclass
class Flag:
    """One plain-English caution, with a severity a non-analyst can read."""

    level: str      # "good" | "watch" | "caution"
    title: str
    detail: str


def build_flags(subject: pd.Series, value: float, lo: float, hi: float,
                neighborhood_bias: float | None = None,
                screening_score: float | None = None,
                n_comparables: int = 0) -> list[Flag]:
    """Translate the model diagnostics into cautions a reader can act on.

    Each flag names the finding, the notebook that measured it, and what it means for
    this property.
    """
    flags: list[Flag] = []

    width = (hi - lo) / value if value else np.nan
    if width > 0.30:
        flags.append(Flag("caution", "Unusually wide valuation range",
                          f"The 80% range spans {width:.0%} of the estimate, wider than "
                          f"usual for this model. Treat this figure as indicative."))
    else:
        flags.append(Flag("good", "Valuation range is typical",
                          f"The 80% range spans {width:.0%} of the estimate, in line "
                          f"with this model's normal precision."))

    if n_comparables < 3:
        flags.append(Flag("caution", "Few comparable sales",
                          f"Only {n_comparables} similar properties were found nearby. "
                          f"Valuations rest on thinner evidence than usual."))

    if neighborhood_bias is not None and abs(neighborhood_bias) >= 0.03:
        direction = "under" if neighborhood_bias < 0 else "over"
        flags.append(Flag("watch", f"Model tends to {direction}-value this neighbourhood",
                          f"Across held-out sales in {subject['Neighborhood']}, this model "
                          f"{direction}-valued by {abs(neighborhood_bias):.1%} on average. "
                          f"That pattern is measured, not corrected for, in the figure above."))

    # No price-band flag.  An earlier version warned that valuations under
    # $110k read "about 10% high" and over $320k "about 4% low".  Those figures grouped
    # errors by *sale price*; this flag conditions on the *valuation*.  Grouped by
    # valuation the error stays within +/-3% with no trend (notebook 04, decile-bias
    # section), so the flag told clients to shade valuations to correct a bias absent in
    # the quantity it keyed on.  Model Risk Log, defect #11.  The neighbourhood flag above
    # is sound because it conditions on and quotes the same variable.

    if screening_score is not None and screening_score >= 0.5:
        flags.append(Flag("caution", "Property resembles non-market transactions",
                          f"A separate screening model scores this property {screening_score:.0%} "
                          f"on resemblance to foreclosures, family transfers and unfinished "
                          f"construction. Confirm the sale is arm's-length before relying "
                          f"on this valuation."))

    if subject.get("Overall Cond", 5) <= 3:
        flags.append(Flag("watch", "Below-average recorded condition",
                          "The assessor records this property in poor condition. Physical "
                          "inspection is advisable before lending against this value."))

    return flags


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_LEVEL_COLOUR = {"good": "#008300", "watch": "#eda100", "caution": "#e34948"}
_LEVEL_WORD = {"good": "OK", "watch": "Note", "caution": "Caution"}


def _money(v: float) -> str:
    return f"${v:,.0f}"


def _esc(v) -> str:
    return html.escape(str(v))


def _interval_bar(value: float, lo: float, hi: float) -> str:
    """Inline SVG showing where the point estimate sits inside its range.

    Drawn so the reader sees the valuation as a band, not a point.
    """
    span = hi - lo
    pos = 50.0 if span <= 0 else (value - lo) / span * 100
    # The bar is the interval: one band and one marker, no background track implying a
    # wider scale.  Text uses
    # `currentColor` and the band a CSS custom property so both follow the page theme.
    return f"""
<svg viewBox="0 0 600 62" width="100%" height="62" role="img"
     aria-label="Valuation range from {_money(lo)} to {_money(hi)}, estimate {_money(value)}">
  <rect x="0" y="22" width="600" height="14" rx="7" fill="var(--band)"/>
  <line x1="{pos * 6:.1f}" y1="14" x2="{pos * 6:.1f}" y2="44"
        stroke="var(--marker)" stroke-width="4" stroke-linecap="round"/>
  <text x="0" y="58" font-size="12" fill="currentColor" opacity=".72">{_money(lo)}</text>
  <text x="600" y="58" font-size="12" fill="currentColor" opacity=".72"
        text-anchor="end">{_money(hi)}</text>
  <text x="{pos * 6:.1f}" y="10" font-size="12" fill="var(--marker)" text-anchor="middle"
        font-weight="600">{_money(value)}</text>
</svg>"""


def _comparables_table(comps: pd.DataFrame, subject: pd.Series) -> str:
    if comps.empty:
        return "<p class='muted'>No comparable sales met the selection criteria.</p>"
    rows = []
    for _, c in comps.iterrows():
        d = c.get("miles_away", np.nan)
        if not np.isfinite(d):
            miles = "same area"
        elif d < 0.05:
            miles = "same block"          # parcel centroids coincide on attached homes
        else:
            miles = f"{d:.2f} mi"
        rows.append(
            f"<tr><td>{_esc(c['Neighborhood'])}</td>"
            f"<td class='num'>{c['Gr Liv Area']:,.0f}</td>"
            f"<td class='num'>{c['Year Built']:.0f}</td>"
            f"<td class='num'>{c['Overall Qual']:.0f}/10</td>"
            f"<td class='num'>{miles}</td>"
            f"<td class='num'>{_esc(int(c['Yr Sold']))}</td>"
            f"<td class='num strong'>{_money(c['SalePrice'])}</td></tr>")
    return f"""
<table>
  <thead><tr><th>Neighbourhood</th><th class='num'>Size (sq ft)</th>
  <th class='num'>Built</th><th class='num'>Quality</th><th class='num'>Distance</th>
  <th class='num'>Sold</th><th class='num'>Sale price</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
  <tfoot><tr><td><strong>This property</strong></td>
  <td class='num'>{subject['Gr Liv Area']:,.0f}</td>
  <td class='num'>{subject['Year Built']:.0f}</td>
  <td class='num'>{subject['Overall Qual']:.0f}/10</td>
  <td class='num'>&mdash;</td><td class='num'>&mdash;</td>
  <td class='num'>&mdash;</td></tr></tfoot>
</table>"""


def _drivers_table(drivers: pd.DataFrame | None) -> str:
    if drivers is None or drivers.empty:
        return ""
    rows = []
    for _, d in drivers.iterrows():
        sign = "+" if d["effect"] >= 0 else "&minus;"
        colour = "#2a78d6" if d["effect"] >= 0 else "#eb6834"
        rows.append(
            f"<tr><td>{_esc(d['label'])}</td>"
            f"<td class='num' style='color:{colour}'>{sign}{_money(abs(d['effect']))}</td></tr>")
    return f"""
<table>
  <thead><tr><th>What moves this valuation most</th>
  <th class='num'>Effect on the estimate</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>"""


def build_valuation_report(subject: pd.Series, value: float, lo: float, hi: float,
                           comparables: pd.DataFrame, flags: list[Flag],
                           drivers: pd.DataFrame | None = None,
                           confidence: float = 0.80,
                           as_of: str | None = None,
                           measured_coverage: float | None = None) -> str:
    """Render one property as a self-contained HTML page.

    ``measured_coverage`` is how often ranges built this way contained the sale price in
    testing. When given, the report quotes it next to the nominal level rather than
    letting the nominal figure stand alone.
    """
    tested = (f" In testing on later sales, ranges built this way contained the sale price "
              f"{measured_coverage:.0%} of the time." if measured_coverage is not None else "")
    as_of = as_of or date.today().strftime("%d %B %Y")
    width_pct = (hi - lo) / value if value else np.nan

    flag_html = "".join(
        f"<li><span class='pill' style='background:{_LEVEL_COLOUR[f.level]}'>"
        f"{_LEVEL_WORD[f.level]}</span><div><strong>{_esc(f.title)}</strong>"
        f"<p>{_esc(f.detail)}</p></div></li>" for f in flags)

    relaxed = comparables.attrs.get("relaxations", []) if hasattr(comparables, "attrs") else []
    relaxed_note = ("<p class='muted'>Selection filters relaxed to find enough "
                    f"comparables: {', '.join(relaxed)}.</p>" if relaxed else "")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Valuation report &mdash; {_esc(subject['Neighborhood'])}, Ames IA</title>
<style>
  :root {{ color-scheme: light dark; --band:#cde2fb; --marker:#2a78d6; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px 20px; background:#fcfcfb; color:#0b0b0b;
         font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
  .page {{ max-width:760px; margin:0 auto; }}
  header {{ border-bottom:2px solid #0b0b0b; padding-bottom:14px; margin-bottom:26px; }}
  h1 {{ font-size:21px; margin:0 0 4px; letter-spacing:-.01em; }}
  h2 {{ font-size:15px; margin:34px 0 10px; text-transform:uppercase;
        letter-spacing:.07em; color:#52514e; }}
  .muted {{ color:#8a8880; font-size:13px; }}
  .hero {{ background:#fff; border:1px solid #e8e7e3; border-radius:10px;
           padding:22px 24px; margin-bottom:8px; }}
  .value {{ font-size:42px; font-weight:700; letter-spacing:-.02em; line-height:1.1; }}
  .plain {{ font-size:15px; margin:14px 0 4px; }}
  table {{ width:100%; border-collapse:collapse; margin:10px 0 4px; font-size:14px; }}
  th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e8e7e3; }}
  th {{ font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#52514e;
        border-bottom:1.5px solid #cbcac5; }}
  td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.strong {{ font-weight:600; }}
  tfoot td {{ border-top:1.5px solid #cbcac5; border-bottom:none; background:#f7f6f3; }}
  ul.flags {{ list-style:none; padding:0; margin:8px 0; }}
  ul.flags li {{ display:flex; gap:12px; align-items:flex-start; padding:11px 0;
                 border-bottom:1px solid #e8e7e3; }}
  ul.flags p {{ margin:2px 0 0; color:#52514e; font-size:14px; }}
  .pill {{ color:#fff; font-size:11px; font-weight:700; text-transform:uppercase;
           letter-spacing:.05em; padding:3px 9px; border-radius:20px; flex:none;
           margin-top:2px; min-width:62px; text-align:center; }}
  .facts {{ display:flex; flex-wrap:wrap; gap:26px; margin-top:12px; font-size:14px; }}
  .facts div span {{ display:block; color:#8a8880; font-size:12px;
                     text-transform:uppercase; letter-spacing:.05em; }}
  footer {{ margin-top:34px; padding-top:16px; border-top:1px solid #e8e7e3;
            color:#8a8880; font-size:12.5px; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --band:#184f95; --marker:#3987e5; }}
    body {{ background:#1a1a19; color:#fff; }}
    .hero, tfoot td {{ background:#232322; border-color:#383835; }}
    th, td, ul.flags li, footer {{ border-color:#383835; }}
    h2, .muted, ul.flags p, footer, .facts div span {{ color:#c3c2b7; }}
  }}
  @media print {{ body {{ padding:0; }} .hero {{ break-inside:avoid; }} }}
</style></head><body><div class="page">

<header>
  <h1>Residential valuation report</h1>
  <div class="muted">{_esc(subject['Neighborhood'])}, Ames, Iowa &nbsp;&middot;&nbsp;
    Parcel {_esc(subject.get('PID', '&mdash;'))} &nbsp;&middot;&nbsp; Prepared {as_of}</div>
</header>

<div class="hero">
  <div class="muted">Estimated market value</div>
  <div class="value">{_money(value)}</div>
  <p class="plain"><strong>The model's {confidence:.0%} range for this home is
     {_money(lo)} to {_money(hi)}.</strong>{tested}</p>
  {_interval_bar(value, lo, hi)}
  <p class="muted">That range is {width_pct:.0%} of the estimate. It is set from how far
     this model's recent valuations fell from the prices those homes sold for, and is
     re-fitted every quarter.</p>
</div>

<div class="facts">
  <div><span>Living area</span>{subject['Gr Liv Area']:,.0f} sq ft</div>
  <div><span>Bedrooms</span>{subject.get('Bedroom AbvGr', 0):.0f}</div>
  <div><span>Year built</span>{subject['Year Built']:.0f}</div>
  <div><span>Quality</span>{subject['Overall Qual']:.0f} of 10</div>
  <div><span>Lot</span>{subject['Lot Area']:,.0f} sq ft</div>
</div>

<h2>Comparable sales</h2>
<p class="muted">Selected the way an appraiser would: nearby first, then similar in size,
   then ranked by closeness in age and quality. Every property below is a real recorded
   sale, so the reasoning can be checked rather than taken on trust.</p>
{_comparables_table(comparables, subject)}
{relaxed_note}

{'<h2>What drives this valuation</h2>' + _drivers_table(drivers) if drivers is not None and not drivers.empty else ''}

<h2>Things to be aware of</h2>
<ul class="flags">{flag_html}</ul>

<footer>
  <strong>How this was produced.</strong> A gradient-boosted model fitted to 1,550 arm's-length
  Ames sales from 2006&ndash;2008 and tested on 862 later sales it had not seen. On that test
  the median valuation error was 5.5%, and 77% of valuations were within 10% of the sale
  price. The range comes from the model's own recent errors, set separately for five price
  bands and re-fitted each quarter on the last twelve months of sales.<br><br>
  <strong>Limitations.</strong> The model has no interior photographs, no renovation permits
  and no listing history, and it does not inspect the property. Valuations of unusual
  homes, and of homes at the very top and bottom of the price range, are less reliable;
  see the notes above. This report is a decision aid, not an appraisal, and is not
  a substitute for one where a regulation requires it.
</footer>

</div></body></html>"""


def save_report(html_text: str, name: str, directory: Path | None = None) -> Path:
    directory = directory or (REPORTS / "valuations")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".html") else f"{name}.html")
    path.write_text(html_text, encoding="utf-8")
    return path
