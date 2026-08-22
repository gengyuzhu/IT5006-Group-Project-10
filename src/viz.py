"""
Shared plotting style and figure export.

Every figure in the report is produced through save_fig(), which writes vector
PDF into report/figures/. LaTeX includes the PDF directly, so the figures stay
sharp at any zoom and the report never depends on a raster screenshot.
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


def money(ax, axis: str = "y") -> None:
    """Format an axis as Brazilian reais."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def pct(ax, axis: str = "y") -> None:
    """Format an axis as percentages."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
