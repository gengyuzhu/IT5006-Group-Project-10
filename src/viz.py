"""
Shared plotting style and figure export.

Every figure in the report is produced through save_fig(), which writes vector
PDF into report/figures/. LaTeX then includes the PDF directly, so the figures
stay sharp at any zoom and the report never depends on a raster screenshot.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from config import FIGURES_DIR, FIG_HEIGHT, FIG_WIDTH, PALETTE


def use_report_style() -> None:
    """Apply the house style. Call once at the top of every notebook."""
    mpl.rcParams.update(
        {
            "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.4,
        }
    )


def save_fig(fig, name: str) -> str:
    """Write a figure to report/figures/<name>.pdf and return the path."""
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return str(path)


def annotate_events(ax, events, y_frac: float = 0.96) -> None:
    """Mark the real-world events that are visible in the weekly series.

    Labels are staggered vertically because the 2007 strike and the 2008
    financial crisis are only months apart and would otherwise overprint.
    """
    ymin, ymax = ax.get_ylim()
    offsets = [0.0, 0.30, 0.60]
    for i, (date, label) in enumerate(events):
        y = ymin + (y_frac - offsets[i % len(offsets)]) * (ymax - ymin)
        ax.axvline(date, color=PALETTE["accent"], linestyle="--", linewidth=0.9, alpha=0.8)
        ax.annotate(
            label,
            xy=(date, y),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=6.5,
            color=PALETTE["accent"],
            rotation=90,
            va="top",
        )
