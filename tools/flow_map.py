"""Draw the notebook dependency map used at the top of the README.

Run with ``make map`` (or ``python tools/flow_map.py``).  Kept as a script rather than a
notebook cell because it documents the repository as a whole, not any one analysis.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ames.viz import (AQUA, BLUE, GRID, INK_MUTED, INK_SECONDARY, ORANGE, SURFACE,
                      VIOLET, save, use_style)

# (key, x, y, title, subtitle, colour)
NODES = [
    ("01", 0.5, 4.00, "01 · Data & EDA", "What is actually\nin the data?", BLUE),
    ("02", 0.5, 2.85, "02 · Macro & geography", "When and where\ndid it happen?", BLUE),
    ("03", 3.0, 3.42, "03 · Hedonic AVM", "What is this\nhouse worth?", ORANGE),
    ("04", 5.5, 4.55, "04 · Uncertainty\n& governance", "How wrong\nmight we be?", AQUA),
    ("05", 5.5, 3.42, "05 · Monitoring", "Is it still\nworking?", AQUA),
    ("06", 5.5, 2.29, "06 · Transaction\nscreening", "Is this even\na real sale?", AQUA),
    ("07", 8.0, 4.95, "07 · Investment\nunderwriting", "Worth buying\nas a rental?", VIOLET),
    ("08", 8.0, 3.82, "08 · Credit risk", "What if prices\nfall 30%?", VIOLET),
    ("09", 8.0, 2.69, "09 · Fair lending", "Who actually\ngets the loan?", VIOLET),
    ("10", 8.0, 1.56, "10 · Client report", "What does the\ncustomer see?", VIOLET),
]

EDGES = [
    ("01", "03"), ("02", "03"),
    ("03", "04"), ("03", "05"), ("03", "06"),
    ("02", "07"),
    ("04", "08"), ("08", "09"),
    ("04", "10"), ("05", "10"), ("06", "10"),
]

STAGES = [
    (0.5, "Foundation", "Understand the data\nand its context"),
    (3.0, "Build", "Fit and interpret\nthe model"),
    (5.5, "Validate & operate", "Measure where it fails,\nand keep watching"),
    (8.0, "Apply & deliver", "Turn it into decisions,\ntested against the real market"),
]

W, H = 2.0, 0.86


def draw() -> None:
    use_style()
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_xlim(-0.75, 9.45)
    ax.set_ylim(0.72, 6.35)
    ax.axis("off")

    positions = {k: (x, y) for k, x, y, *_ in NODES}

    for x, name, blurb in STAGES:
        ax.text(x, 5.95, " ".join(name.upper()), ha="center", fontsize=10,
                fontweight="bold", color=INK_SECONDARY)
        ax.text(x, 5.72, blurb, ha="center", va="top", fontsize=8.5, color=INK_MUTED)

    for src, dst in EDGES:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        start = (x0 + W / 2, y0) if x1 > x0 else (x0, y0 - H / 2)
        end = (x1 - W / 2, y1) if x1 > x0 else (x1, y1 + H / 2)
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            connectionstyle="arc3,rad=0.12" if abs(y1 - y0) > 0.3 else "arc3,rad=0",
            color=GRID, lw=1.4, shrinkA=2, shrinkB=2, zorder=1))

    for key, x, y, title, subtitle, colour in NODES:
        ax.add_patch(FancyBboxPatch(
            (x - W / 2, y - H / 2), W, H,
            boxstyle="round,pad=0.035,rounding_size=0.09",
            facecolor=SURFACE, edgecolor=colour, linewidth=1.8, zorder=2))
        ax.text(x, y + 0.17, title, ha="center", va="center", fontsize=9.5,
                fontweight="semibold", color=colour, zorder=3)
        ax.text(x, y - 0.17, subtitle, ha="center", va="center", fontsize=8,
                color=INK_SECONDARY, linespacing=1.35, zorder=3)

    ax.text(-0.72, 1.62,
            "Arrows are real data dependencies: each notebook reads artefacts the one "
            "before it wrote.  `make run` executes all ten in order from a clean kernel.",
            ha="left", fontsize=8.5, color=INK_MUTED)
    fig.suptitle("How the ten notebooks fit together", x=0.008, ha="left",
                 y=0.985, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "00_notebook_map")
    plt.close(fig)


if __name__ == "__main__":
    draw()
