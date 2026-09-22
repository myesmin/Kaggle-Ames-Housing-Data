"""Where Ames sits among 410 US metropolitan housing markets.

Ames barely moved in the crash: it fell 4.3% peak-to-trough against 27.4% nationally,
so the temporal split has almost no regime shift and the credit stress is imposed
rather than observed.

Against the metro distribution this cuts both ways: Ames is unusually calm, and the
-30% supervisory shock used here is not unusually severe (nearly one metro in five
exceeded it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: The crisis window.  Wide enough to contain every metro's own peak and trough (they
#: did not coincide); a fixed peak date would understate the drawdown in markets that
#: peaked late.
CRISIS_START, CRISIS_END = 2005.0, 2013.0

#: A metro needs most of the window to be measurable.  32 quarters is eight years;
#: below 30 the running maximum is being taken over too little history to trust.
MIN_QUARTERS = 30


def drawdowns(metro_hpi: pd.DataFrame, start: float = CRISIS_START,
              end: float = CRISIS_END, min_quarters: int = MIN_QUARTERS
              ) -> pd.DataFrame:
    """Peak-to-trough decline for each metro inside ``[start, end]``.

    The drawdown is measured against a **running** maximum rather than the window's
    global peak, so a metro that peaked in 2006 and one that peaked in 2008 are each
    measured from their own top. Using a common peak date would score late-peaking
    markets as milder than they were.

    Returns one row per metro: ``drawdown`` (negative), and the decimal-year peak and
    trough.
    """
    window = metro_hpi[(metro_hpi["t"] >= start) & (metro_hpi["t"] <= end)]
    rows = []
    for cbsa, group in window.groupby("cbsa"):
        if len(group) < min_quarters:
            continue
        group = group.sort_values("t")
        index = group["index_nsa"].to_numpy()
        t = group["t"].to_numpy()
        running_peak = np.maximum.accumulate(index)
        decline = index / running_peak - 1.0
        trough = int(decline.argmin())
        rows.append({
            "cbsa": int(cbsa),
            "metro": group["metro"].iloc[0],
            "drawdown": float(decline[trough]),
            "peak_t": float(t[: trough + 1][int(np.argmax(index[: trough + 1]))]),
            "trough_t": float(t[trough]),
        })
    columns = ["cbsa", "metro", "drawdown", "peak_t", "trough_t"]
    if not rows:
        # A window containing no measurable metro is a legitimate answer (ask for a
        # window before the panel starts and you get one). `pd.DataFrame([])` has no
        # columns at all, so sorting it raises KeyError instead of returning nothing.
        return pd.DataFrame(columns=columns)
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("drawdown").reset_index(drop=True))


def percentile_of(table: pd.DataFrame, cbsa: int) -> float:
    """Share of metros that fell *further* than ``cbsa``.

    Phrased this way because "82% of metros fell further than Ames" is unambiguous;
    "Ames is at the 18th percentile" leaves the reader to work out which tail.
    """
    row = table.loc[table["cbsa"] == cbsa]
    if row.empty:
        raise KeyError(f"cbsa {cbsa} not in the drawdown table")
    return float((table["drawdown"] < row["drawdown"].iloc[0]).mean())


def shock_at_quantile(table: pd.DataFrame, q: float) -> float:
    """The drawdown at the ``q`` tail of the metro distribution.

    ``shock_at_quantile(t, 0.10)`` is "what a metro at the 10th percentile of outcomes
    actually experienced", an empirically calibrated severe scenario rather than an
    imposed round number.
    """
    return float(table["drawdown"].quantile(q))


def severity_of(table: pd.DataFrame, shock: float) -> float:
    """What share of metros met or exceeded a given shock.

    A check on any imposed scenario: a "severely adverse" -30% that one metro in five
    exceeded in the 2005-2013 crisis window is closer to moderate.
    """
    return float((table["drawdown"] <= shock).mean())
