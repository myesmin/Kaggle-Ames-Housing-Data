"""The project's chart system.

Every figure in the repository is built from here so all notebooks share one style
(modelled on Economist / FT charts). Rules:

* Subject series in one accent colour; context in grey.
* Title states the finding ("Every loan says 80% LTV", not "LTV distribution"). Kicker
  above names the section; standfirst below says what to look at.
* Direct labels at line ends instead of legends.
* No frame, no tick marks, horizontal hairline gridlines only.
* Spacing in inches, not figure fractions: the title block reserves a fixed physical
  height, so 4-inch and 8-inch figures get the same typography without overlap.
  (Fraction-based spacing caused the earlier overlapping titles.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES

# --------------------------------------------------------------------------
# Surface and ink
# --------------------------------------------------------------------------
# Warm off-white rather than pure white: lowers page contrast so the data can use the
# top of the range.
SURFACE = "#FBFAF7"
PANEL = "#F3F1EC"          # for shaded bands and inset blocks

INK = "#16150F"            # headlines
INK_SECONDARY = "#55534B"  # axis labels, annotations
INK_MUTED = "#8E8B80"      # standfirst, source notes, reference labels
GRID = "#E4E1D9"           # hairline gridlines
RULE = "#CBC7BC"           # the rule under the title block

# --------------------------------------------------------------------------
# Colour, assigned by the job it does
# --------------------------------------------------------------------------
BLUE = "#2a78d6"     # the subject
ORANGE = "#eb6834"   # the contrast / the comparison
AQUA = "#1baf7a"     # a third series, only when needed
YELLOW, MAGENTA, GREEN, VIOLET, RED = "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"

SUBJECT, CONTRAST = BLUE, ORANGE

#: Large filled areas (histograms, bar charts, bands) use the softer tints. A saturated
#: hue is too loud over a big block but fine on a thin line or small dot.
SUBJECT_FILL, CONTRAST_FILL = "#5E9CE4", "#F08E62"

#: Supporting cast.  Context series are drawn from here, never from the accent hues.
CONTEXT = "#C9C5BA"
CONTEXT_DARK = "#A8A396"

#: Safe where every pair is visible at once (scatter, choropleth, small multiples).
CATEGORICAL_3 = [BLUE, ORANGE, AQUA]
#: Safe where only adjacent pairs must separate (bars, lines, stacks).
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SEQUENTIAL_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                    "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("ames_seq", SEQUENTIAL_STEPS)
DIVERGING = LinearSegmentedColormap.from_list(
    "ames_div", ["#0d366b", "#3987e5", "#9ec5f4", "#EDEAE3",
                 "#f3a3a2", "#e34948", "#8f2322"])

#: Reserved.  Never reused as "series four".
STATUS = {"good": "#0f8a4d", "warning": "#d99400", "serious": "#eb6834",
          "critical": "#cf2e2e"}

# --------------------------------------------------------------------------
# Typography
# --------------------------------------------------------------------------
# Avenir Next: has a condensed cut for labels and a demi-bold for headlines. Fallbacks
# cover machines without it; the committed PNGs are rendered with it.
# NB: Avenir Next has no arrow glyphs (U+2190-2193).  Use ASCII "->" in anything that
# reaches a chart; matplotlib renders the missing glyph as an empty box and only warns.
# Markdown cells are unaffected (rendered by the browser, not matplotlib).
SANS = ["Avenir Next", "Avenir", "Helvetica Neue", "DejaVu Sans"]
CONDENSED = ["Avenir Next Condensed", "DIN Condensed", "Arial Narrow"] + SANS


def use_style() -> None:
    """Apply the project chart style.  Call once at the top of every notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": None,          # the layout is computed, not cropped
        "savefig.pad_inches": 0,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 9.5,
        "axes.labelpad": 8,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "demibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 7,
        "ytick.major.pad": 6,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SECONDARY,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.8,
        "lines.linewidth": 1.9,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0,
        "font.size": 9.5,
        "font.family": "sans-serif",
        "font.sans-serif": SANS,
        "figure.autolayout": False,
    })


# --------------------------------------------------------------------------
# The chart factory
# --------------------------------------------------------------------------

def _wrap_to_width(text: str | None, fig_w: float, fontsize: float = 9.8) -> str | None:
    """Wrap a standfirst to the figure's own width.

    Measures in ems against the actual figure width, so text is not clipped at the
    canvas edge on narrow figures (a fixed character count was). Explicit newlines are
    honoured.
    """
    if not text:
        return text
    import textwrap

    em_inches = fontsize / 72 * 0.52          # mean glyph advance for this family
    usable = max(fig_w - 1.5, 2.0)
    limit = max(int(usable / em_inches), 24)
    return "\n".join(
        line for para in text.split("\n")
        for line in (textwrap.wrap(para, limit) or [""]))


def _spaced(text: str) -> str:
    """Poor man's letter-spacing for the kicker.  Matplotlib has no tracking control."""
    return " ".join(text.upper())


def chart(headline: str, standfirst: str | None = None, kicker: str | None = None,
          source: str | None = None, figsize: tuple[float, float] = (9.6, 5.2),
          nrows: int = 1, ncols: int = 1, height_ratios=None, width_ratios=None,
          wspace: float = 0.26, hspace: float = 0.42, sharex: bool = False,
          sharey: bool = False):
    """Create a figure with an editorial title block, and return ``(fig, ax_or_axes)``.

    The title block reserves a fixed height in inches, computed from its contents, so
    typography is the same at any figure size and the plot does not overlap the title.

    Parameters
    ----------
    headline    the finding, stated as a sentence.  ``\\n`` to wrap.
    standfirst  one line telling the reader how to read the chart.
    kicker      small letterspaced label above the headline (section or dataset).
    source      the note under the rule at the bottom.
    """
    fig_w, fig_h = figsize
    pad = 0.34
    kicker_h = 0.22 if kicker else 0.0
    headline = _wrap_to_width(headline, fig_w, 14.5)
    standfirst = _wrap_to_width(standfirst, fig_w)
    head_h = 0.30 * (headline.count("\n") + 1)
    stand_h = 0.24 * (standfirst.count("\n") + 1) if standfirst else 0.0
    rule_gap = 0.40
    # Panels inside a multi-panel chart may carry their own heading and plain line.
    panel_h = 0.0 if (nrows == 1 and ncols == 1) else PANEL_SUBTITLE_INCHES + 0.30
    top_block = pad + kicker_h + head_h + stand_h + rule_gap + panel_h

    bottom_block = 0.72 + (0.34 if source else 0.0)

    fig = plt.figure(figsize=figsize, facecolor=SURFACE)
    gs = fig.add_gridspec(
        nrows, ncols,
        left=0.9 / fig_w, right=1 - 0.5 / fig_w,
        top=1 - top_block / fig_h, bottom=bottom_block / fig_h,
        wspace=wspace, hspace=hspace,
        height_ratios=height_ratios, width_ratios=width_ratios)

    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            share_x = axes[0, c] if (sharex and r) else None
            share_y = axes[r, 0] if (sharey and c) else None
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=share_x, sharey=share_y)

    # --- the title block, positioned from the top in inches ---
    x = 0.9 / fig_w
    y = 1 - pad / fig_h
    if kicker:
        fig.text(x, y, _spaced(kicker), ha="left", va="top", fontsize=8,
                 color=INK_MUTED, family=CONDENSED, fontweight="demibold")
        y -= kicker_h / fig_h
    fig.text(x, y, headline, ha="left", va="top", fontsize=14.5,
             color=INK, fontweight="demibold", linespacing=1.28)
    y -= head_h / fig_h
    if standfirst:
        fig.text(x, y - 0.02 / fig_h, standfirst, ha="left", va="top", fontsize=9.8,
                 color=INK_MUTED, linespacing=1.4)
        y -= stand_h / fig_h
    fig.add_artist(plt.Line2D(
        [x, 1 - 0.5 / fig_w], [y - 0.10 / fig_h] * 2, color=RULE, lw=0.9,
        transform=fig.transFigure, zorder=0))

    if source:
        fig.text(x, 0.26 / fig_h, source, ha="left", va="center", fontsize=8,
                 color=INK_MUTED)

    out = axes[0, 0] if (nrows == 1 and ncols == 1) else (
        axes[0] if nrows == 1 else (axes[:, 0] if ncols == 1 else axes))
    return fig, out


def save(fig, name: str, directory: Path = FIGURES) -> Path:
    """Write a figure to reports/figures/ and report the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, facecolor=SURFACE)
    print(f"  saved {path.relative_to(path.parents[2])}")
    return path


# --------------------------------------------------------------------------
# Panel headings (inside a multi-panel figure)
# --------------------------------------------------------------------------

#: Extra top margin, in inches, that a panel subtitle needs.  ``titled`` places the plain
#: line *outside* the axes, where ``tight_layout`` cannot see it, so the panel records its
#: requirement on the figure and the title block reads it back.
PANEL_SUBTITLE_INCHES = 0.34


def titled(ax, headline: str, plain: str | None = None, pad: int = 26) -> None:
    """Panel heading, with an optional plain-language line beneath it."""
    if not plain:
        ax.set_title(headline, pad=11)
        return
    ax.set_title(headline, pad=pad)
    ax.text(0, 1.020, plain, transform=ax.transAxes, fontsize=8.8,
            color=INK_MUTED, va="bottom", ha="left")
    ax.get_figure()._ames_panel_subtitle = True


def fig_titled(fig, headline: str, plain: str | None = None,
               kicker: str | None = None, source: str | None = None) -> None:
    """Add the editorial title block to a figure built with ``plt.subplots``.

    Same typography and inch-based spacing as :func:`chart`, but reclaims the space
    after the axes exist instead of reserving it up front. Either way spacing is
    computed from the figure's physical height, so titles do not collide on resize.
    """
    fig_w, fig_h = fig.get_size_inches()
    pad = 0.34
    kicker_h = 0.22 if kicker else 0.0
    headline = _wrap_to_width(headline, fig_w, 14.5)
    plain = _wrap_to_width(plain, fig_w)
    head_h = 0.30 * (headline.count("\n") + 1)
    stand_h = 0.24 * (plain.count("\n") + 1) if plain else 0.0
    panel_h = PANEL_SUBTITLE_INCHES if getattr(fig, "_ames_panel_subtitle", False) else 0.0
    top_block = pad + kicker_h + head_h + stand_h + 0.40 + panel_h
    bottom_block = 0.72 + (0.34 if source else 0.0)

    fig.tight_layout()
    fig.subplots_adjust(top=1 - top_block / fig_h, bottom=bottom_block / fig_h)

    x = 0.9 / fig_w
    y = 1 - pad / fig_h
    if kicker:
        fig.text(x, y, _spaced(kicker), ha="left", va="top", fontsize=8,
                 color=INK_MUTED, family=CONDENSED, fontweight="demibold")
        y -= kicker_h / fig_h
    fig.text(x, y, headline, ha="left", va="top", fontsize=14.5, color=INK,
             fontweight="demibold", linespacing=1.28)
    y -= head_h / fig_h
    if plain:
        fig.text(x, y, plain, ha="left", va="top", fontsize=9.8, color=INK_MUTED,
                 linespacing=1.4)
        y -= stand_h / fig_h
    fig.add_artist(plt.Line2D([x, 1 - 0.5 / fig_w], [y - 0.14 / fig_h] * 2,
                              color=RULE, lw=0.9, transform=fig.transFigure, zorder=0))
    if source:
        fig.text(x, 0.26 / fig_h, source, ha="left", va="center", fontsize=8,
                 color=INK_MUTED)


# --------------------------------------------------------------------------
# Annotation
# --------------------------------------------------------------------------

def annotate(ax, x, y, text: str, color: str = INK_SECONDARY, **kwargs) -> None:
    """Selective direct label.  Never a number on every point."""
    ax.annotate(text, (x, y), color=color, fontsize=9,
                xytext=kwargs.pop("offset", (7, 0)), textcoords="offset points",
                va=kwargs.pop("va", "center"), zorder=6, **kwargs)


def label_end(ax, x, y, text: str, color: str = SUBJECT, **kwargs) -> None:
    """Name a series at its endpoint, so it needs no legend entry."""
    annotate(ax, x, y, text, color=color, offset=kwargs.pop("offset", (9, 0)),
             fontweight="demibold", **kwargs)


def reference_line(ax, value: float, label: str, axis: str = "y",
                   color: str = INK_MUTED, lw: float = 1.0,
                   side: str = "right", pad: float = 5.0,
                   linestyle: str = "-") -> None:
    """Draw a threshold and put its label *outside* the data area."""
    if axis == "y":
        ax.axhline(value, color=color, lw=lw, zorder=1, linestyle=linestyle)
        x = 1.0 if side == "right" else 0.0
        ax.annotate(label, xy=(x, value), xycoords=("axes fraction", "data"),
                    xytext=(pad if side == "right" else -pad, 0),
                    textcoords="offset points", fontsize=8.4, color=color,
                    va="center", ha="left" if side == "right" else "right",
                    annotation_clip=False, zorder=5, linespacing=1.3)
    else:
        ax.axvline(value, color=color, lw=lw, zorder=1, linestyle=linestyle)
        y = 1.0 if side == "right" else 0.0
        ax.annotate(label, xy=(value, y), xycoords=("data", "axes fraction"),
                    xytext=(0, pad if side == "right" else -pad),
                    textcoords="offset points", fontsize=8.4, color=color,
                    ha="center", va="bottom" if side == "right" else "top",
                    annotation_clip=False, zorder=5, linespacing=1.3)


def band(ax, lo: float, hi: float, label: str | None = None, axis: str = "x") -> None:
    """A shaded region of interest, drawn behind the data."""
    span = ax.axvspan if axis == "x" else ax.axhspan
    span(lo, hi, color=PANEL, zorder=0, lw=0)
    if label:
        mid = (lo + hi) / 2
        if axis == "x":
            ax.annotate(label, xy=(mid, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, -12), textcoords="offset points", ha="center",
                        va="top", fontsize=8.4, color=INK_MUTED)
        else:
            ax.annotate(label, xy=(1.0, mid), xycoords=("axes fraction", "data"),
                        xytext=(-6, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8.4, color=INK_MUTED)


def callout(ax, x, y, text: str, dx: float = 40, dy: float = 40,
            color: str = INK_SECONDARY, ha: str | None = None,
            dot: bool = True, dot_color: str | None = None,
            width: int = 26) -> None:
    """An annotation with a leader line, pointing at the moment it describes.

    Puts the explanation at the point on the canvas where the event happened, so the
    reader does not have to match text to data.
    """
    import textwrap

    ha = ha or ("left" if dx >= 0 else "right")
    if dot:
        ax.scatter([x], [y], s=34, color=dot_color or color, zorder=6,
                   edgecolor=SURFACE, linewidths=1.6)
    ax.annotate(
        textwrap.fill(text, width), xy=(x, y), xytext=(dx, dy),
        textcoords="offset points", ha=ha, va="center",
        fontsize=8.8, color=color, linespacing=1.45, zorder=6,
        arrowprops=dict(arrowstyle="-", color=RULE, lw=0.9,
                        shrinkA=2, shrinkB=6,
                        connectionstyle="angle3,angleA=0,angleB=90"))


def era(ax, x0, x1, label: str | None = None, color: str = PANEL) -> None:
    """Shade a period and name it once, at the top, out of the data's way."""
    ax.axvspan(x0, x1, color=color, zorder=0, lw=0)
    if label:
        mid = x0 + (x1 - x0) / 2
        ax.annotate(label, xy=(mid, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(0, 0), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8.2, color=INK_MUTED, linespacing=1.35)


def axis_note(ax, text: str) -> None:
    """Units, written horizontally above the axis instead of rotated up its side.

    Rotated axis labels are easy to miss.
    """
    ax.annotate(text, xy=(0, 1.0), xycoords="axes fraction",
                xytext=(0, 9), textcoords="offset points", ha="left", va="bottom",
                fontsize=8.6, color=INK_MUTED, annotation_clip=False)
    ax.set_ylabel("")


def legend_above(ax, ncol: int = 3, **kwargs) -> None:
    """Legend above the plot, never on top of the data."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=ncol,
              frameon=False, fontsize=9, **kwargs)


# --------------------------------------------------------------------------
# Formatters
# --------------------------------------------------------------------------

def money(value, _pos=None) -> str:
    v = float(value)
    if abs(v) >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:,.0f}k"
    return f"${v:,.0f}"


def pct(value, _pos=None) -> str:
    return f"{value:.0%}"


def pct1(value, _pos=None) -> str:
    """One-decimal percent, for axes whose whole range is a point or two wide."""
    return f"{value:.1%}"


# --------------------------------------------------------------------------
# Forms
# --------------------------------------------------------------------------

def hero(ax, value: str, label: str, sublabel: str = "") -> None:
    """A stat tile, for when the finding is a single number."""
    ax.axis("off")
    ax.text(0, 0.62, value, fontsize=34, fontweight="demibold", color=INK, va="center")
    ax.text(0, 0.30, label, fontsize=10.5, color=INK_SECONDARY, va="center")
    if sublabel:
        ax.text(0, 0.12, sublabel, fontsize=9, color=INK_MUTED, va="center")


def dumbbell(ax, categories, left, right, left_label: str, right_label: str,
             left_color: str = SUBJECT, right_color: str = CONTRAST,
             formatter=None) -> None:
    """Paired dot plot: two conditions per category, joined by a rule.

    Use when the finding is the gap between two conditions. Grouped bars from a zero
    baseline waste the space: with values between 5.1% and 5.8%, 90% of the bar is
    empty range.
    """
    y = np.arange(len(categories))
    for i, (a, b) in enumerate(zip(left, right)):
        ax.plot([a, b], [i, i], color=CONTEXT, lw=2.6, zorder=1,
                solid_capstyle="round")
    ax.scatter(left, y, s=86, color=left_color, zorder=3,
               edgecolor=SURFACE, linewidths=1.5, label=left_label)
    ax.scatter(right, y, s=86, color=right_color, zorder=3,
               edgecolor=SURFACE, linewidths=1.5, label=right_label)
    ax.set_yticks(y, categories)
    ax.set_ylim(-0.62, len(categories) - 0.38)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    if formatter is not None:
        ax.xaxis.set_major_formatter(formatter)


def emphasise(ax, x, series: dict, highlight: str, highlight_color: str = CONTRAST,
              muted_color: str = CONTEXT, label: bool = True, **kwargs) -> None:
    """Plot many lines with one picked out and the rest recessive.

    Use instead of many categorical hues when the finding concerns one series: colour
    marks emphasis, not identity.
    """
    for name, values in series.items():
        if name == highlight:
            continue
        ax.plot(x, values, color=muted_color, lw=1.2, zorder=1, **kwargs)
    values = list(series[highlight])
    ax.plot(x, values, color=highlight_color, lw=2.4, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=3, **kwargs)
    if label:
        label_end(ax, x[-1], values[-1], highlight, color=highlight_color)


def bar_labels(ax, bars, labels, color: str = INK_SECONDARY, pad: float = 5,
               fontsize: float = 8.8) -> None:
    """Value labels just outside the end of each **horizontal** bar (``ax.barh``).

    For vertical bars use matplotlib's own ``ax.bar_label``: this reads
    ``get_width()``, which on a vertical bar is its thickness, so it would place
    every label at x = 0.6 without raising anything.
    """
    for rect, text in zip(bars, labels):
        w = rect.get_width()
        ax.annotate(text, xy=(w, rect.get_y() + rect.get_height() / 2),
                    xytext=(pad if w >= 0 else -pad, 0), textcoords="offset points",
                    va="center", ha="left" if w >= 0 else "right",
                    fontsize=fontsize, color=color)
