"""Shared visual system for notebooks, final figures and the PDF report.

Typeface: Source Sans 3 (SIL OFL, vendored in ``assets/fonts``) so figures render
identically on any machine. Colours: validated categorical slots (blue, orange,
aqua) + neutral inks; one accent colour per chart wherever possible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = REPO_ROOT / "assets" / "fonts"
FONT_FAMILY = "Source Sans 3"

# Palette -------------------------------------------------------------------- #
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"       # text inks
GRID, RULE, SURFACE = "#e4e3df", "#c9c8c2", "#ffffff"
HIGHLIGHT = ORANGE      # the one thing the reader should look at
BASE = "#b9c7d8"        # de-emphasised marks (light blue-grey)

ROLE = {"catalogue": BLUE, "exposure": ORANGE, "click": AQUA}


def register_fonts() -> bool:
    """Register vendored Source Sans 3 with matplotlib. Returns True if available."""
    ok = False
    for f in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(f))
        ok = True
    return ok


def apply(dpi: int = 110) -> None:
    """Apply rcParams for the project's chart style."""
    has_font = register_fonts()
    mpl.rcParams.update({
        "font.family": FONT_FAMILY if has_font else "DejaVu Sans",
        "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "figure.dpi": dpi, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
        "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "xtick.major.pad": 4, "ytick.major.pad": 4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def title(ax, text: str, sub: str | None = None) -> None:
    """Left-aligned title with optional muted subtitle above the plot area."""
    ax.set_title(text, loc="left", fontsize=12.5, fontweight="semibold", color=INK, pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=INK2, va="bottom", ha="left")


def note(fig, text: str, y: float = -0.02) -> None:
    """Small source/footnote line at the bottom-left of a figure."""
    fig.text(0.01, y, text, fontsize=8.5, color=MUTED, ha="left", va="top")


def save(fig, path: Path, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Save a figure in several formats; returns written paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = path.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out


def category_label(name: str) -> str:
    """Human-readable category names for judges (MIND uses run-together labels)."""
    return {"foodanddrink": "Food & drink", "tv": "TV", "autos": "Autos",
            "middleeast": "Middle East", "northamerica": "North America"}.get(name, name.capitalize())


def place_labels(ax, xs, ys, texts, fontsize: float = 8.5, color: str = INK2, pad_pts: float = 4.0) -> list:
    """Place point labels without overlaps (greedy).

    For each point, tries offsets (above, below, right, left, diagonals, then farther out) in
    display space and keeps the first whose box overlaps no earlier label and no data point.
    """
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(list(zip(xs, ys)))
    scale = fig.dpi / 72.0
    cands = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1),
             (0, 2.2), (0, -2.2), (2, 0), (-2, 0)]
    placed, artists = [], []
    pt_boxes = [Bbox.from_extents(x - 3 * scale, y - 3 * scale, x + 3 * scale, y + 3 * scale) for x, y in pts]
    for (px, py), t in zip(pts, texts):
        best = None
        for dx, dy in cands:
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
            off = (dx * pad_pts * 1.4, dy * pad_pts)
            art = ax.annotate(t, ax.transData.inverted().transform((px, py)), xytext=off, textcoords="offset points",
                              ha=ha, va=va, fontsize=fontsize, color=color)
            bb = art.get_window_extent(renderer).expanded(1.04, 1.08)
            clash = any(bb.overlaps(o) for o in placed) or any(bb.overlaps(o) for o in pt_boxes)
            inside = ax.get_window_extent(renderer).contains(bb.x0, bb.y0) and ax.get_window_extent(renderer).contains(bb.x1, bb.y1)
            if not clash and inside:
                best = (art, bb)
                break
            art.remove()
        if best is None:  # fall back to above
            art = ax.annotate(t, ax.transData.inverted().transform((px, py)), xytext=(0, pad_pts),
                              textcoords="offset points", ha="center", va="bottom", fontsize=fontsize, color=color)
            best = (art, art.get_window_extent(renderer))
        placed.append(best[1]); artists.append(best[0])
    return artists
