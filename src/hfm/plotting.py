"""Shared matplotlib styling so every figure in the repo reads as one set.

Colours follow a validated, colour-blind-safe palette: one blue ramp for
ordered quantities (risk level, percentages) and a fixed categorical order
for unordered series.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Sequential blue ramp, light -> dark (steps 100..700).
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
BLUE_CMAP = LinearSegmentedColormap.from_list("hfm_blue", BLUE_RAMP)

# Ordinal steps for Low / Moderate / High (no step lighter than 250 on light).
RISK_COLORS = {"Low": "#86b6ef", "Moderate": "#2a78d6", "High": "#104281"}

# Fixed categorical order for unordered series (never cycled).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def use_style() -> None:
    """Apply the house style to matplotlib's global rcParams."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def bar_labels(ax, bars, fmt="{:.0f}", color=INK_2, pad=2) -> None:
    """Direct-label a set of bars with their height."""
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
        )


def tidy(ax) -> None:
    """Recessive gridlines on the value axis only."""
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    plt.setp(ax.spines.values(), linewidth=0.8)
