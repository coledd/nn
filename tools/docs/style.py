# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# The shared look of every figure in the examples' READMEs: one palette,
# one typeface, one set of chart conventions, so the eight documents read as
# a series. Imported by tools/docs/<example>.py; nothing here is specific to
# one example.
#
# Colors are the validated reference palette of the data-viz guidelines
# these figures follow (categorical slots in a fixed, colorblind-checked
# order; never cycled). Each figure carries its own light surface, so it
# reads the same in GitHub's light and dark themes.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots, in order. Assign by entity, never by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

# Sequential (blue) and diverging (blue <-> gray <-> red) ramps.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
RED_RAMP = ["#fbdada", "#f6c1c1", "#f0a6a6", "#ea8c8b", "#e46f6e", "#e34948", "#c93a39", "#a92f2e"]
NEUTRAL_MID = "#f0efec"

# Surfaces and ink.
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
REST_GRAY = "#c3c2b7"

FONT = "Noto Sans"
# For hand-written SVG (banners, illustrations): the viewer's own UI sans.
SVG_FONT = "-apple-system, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"


def setup():
    """Applies the house style to matplotlib. Call once per script."""
    plt.rcParams.update({
        "font.family": FONT,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
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
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "svg.fonttype": "path",  # text as outlines: identical rendering everywhere
        "svg.hashsalt": "nn-docs",  # stable ids, so regenerating doesn't churn the diff
    })


def save(fig, path, raster=False):
    """Saves an SVG (vector charts) or a 2x PNG (photos, dense images)."""
    kwargs = dict(bbox_inches="tight", pad_inches=0.25)
    if raster:
        fig.savefig(path, dpi=200, **kwargs)
    else:
        fig.savefig(path, metadata={"Date": None, "Creator": None}, **kwargs)
    plt.close(fig)


def caption_title(ax, title, subtitle=None):
    """A left-aligned chart title with an optional muted subtitle beneath it."""
    ax.set_title(title, loc="left", color=INK, pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, color=INK_2)


def end_label(ax, x, y, text, dx=6, color=INK_2, **kw):
    """A direct label just past a line's last point (text in ink, never series color)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", ha="left", fontsize=9.5, color=color, **kw)


def bottom_legend(fig, ax, ncol, gap=0.1):
    """One shared legend under a row of panels, clear of their tick labels.
    `gap` is in figure heights below the lowest axes' tick labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom = min(a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0 for a in fig.axes)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, bottom - gap * 0.2),
               handlelength=1.6, columnspacing=1.6)


def runs(seq):
    """(start, length, value) for each run of equal values in `seq`."""
    out, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            out.append((start, i - start, seq[start]))
            start = i
    return out
