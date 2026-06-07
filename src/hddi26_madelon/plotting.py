"""Matplotlib plotting helpers."""

from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .settings import FIGURE_DIR, ensure_runtime_dirs


PALETTE = {
    "blue": "#1F5A96",
    "light_blue": "#77A6D8",
    "teal": "#2B9A9C",
    "green": "#66A65C",
    "gold": "#C8A13A",
    "red": "#B64745",
    "purple": "#7E5AA6",
    "gray": "#707070",
    "light_gray": "#D8D8D8",
    "black": "#262626",
}

CLASS_COLORS = {-1: PALETTE["red"], 1: PALETTE["blue"]}
METHOD_COLORS = [
    PALETTE["blue"],
    PALETTE["teal"],
    PALETTE["gold"],
    PALETTE["purple"],
    PALETTE["green"],
    PALETTE["red"],
    PALETTE["light_blue"],
    "#999999",
]


def apply_style(font_size: int = 7) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": font_size,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: mpl.figure.Figure, name: str) -> None:
    ensure_runtime_dirs()
    stem = FIGURE_DIR / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def scatter_by_label(
    ax: mpl.axes.Axes,
    coords: np.ndarray,
    y: Iterable[int],
    title: str,
    xlabel: str,
    ylabel: str,
    palette: Mapping[int, str] | None = None,
    errors: np.ndarray | None = None,
) -> None:
    palette = palette or CLASS_COLORS
    y = np.asarray(y)
    for cls, color in palette.items():
        mask = y == cls
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=15,
                alpha=0.75,
                linewidth=0.25,
                edgecolor="white",
                color=color,
                label=f"class {cls:+d}",
            )
    if errors is not None and errors.any():
        ax.scatter(
            coords[errors, 0],
            coords[errors, 1],
            s=45,
            facecolor="none",
            edgecolor=PALETTE["black"],
            linewidth=1.0,
            label="misclassified",
        )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, markerscale=1.2)
