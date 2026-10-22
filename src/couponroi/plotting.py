"""One chart style for every figure in the project.

Colours come from a colour-blind-checked categorical palette, used in a fixed order.
Thin marks, hairline grids, no dual axes, and text never wears a series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"  # for "everything else" / reference marks

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
POS, NEG = BLUE, RED  # diverging poles; grey midpoint below
MID = "#f0efec"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 24,
            "axes.labelsize": 9.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "patch.linewidth": 0,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
        }
    )


def subtitle(ax, text: str) -> None:
    """Small grey line under the title that states the takeaway or the unit."""
    ax.annotate(
        text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
        fontsize=9, color=INK_2, va="bottom", ha="left",
    )


def note(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, fontsize=8, color=MUTED, ha="left", va="top")


def dept_label(name: str) -> str:
    """'DRUG GM' -> 'Drug/GM', 'MEAT-PCKGD' -> 'Meat (packaged)' for readable axis labels."""
    special = {"DRUG GM": "Drug/GM", "MEAT-PCKGD": "Meat (packaged)", "SEAFOOD-PCKGD": "Seafood (packaged)"}
    return special.get(name, name.title())


def save(fig, name: str, fig_dir: Path | None = None) -> Path:
    fig_dir = fig_dir or config.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def money(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.1f}K"
    return f"{sign}${x:,.{decimals}f}"


apply_style()
