# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/object_detection/README.md. The photos and their
# labeled boxes are read from the downloaded BCCD dataset (run make in the
# example first); detections, heatmaps and charts plot program output the
# README reports.
#
#   python3 tools/docs/object_detection.py

import os

import numpy as np
from PIL import Image

import style
from banner import banner
from style import INK_2, MUTED, SERIES, plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "object_detection")
OUT = os.path.join(ROOT, "figures")
JPEGS = os.path.join(ROOT, "dataset", "BCCD", "JPEGImages")

CLASSES = ["Red blood cell", "White blood cell", "Platelet"]
SHORT = ["RBC", "WBC", "platelet"]
# One fixed color per class, in categorical slot order.
CLASS_COLORS = SERIES[:3]


def make_banner():
    banner(os.path.join(OUT, "banner.svg"), 6, "Medical imaging · real data",
           "Object Detection",
           ["Finding and counting red cells, white cells and",
            "platelets in a microscope photo of a blood smear."],
           [("97%", "white-cell F1,", "every one found"),
            ("77%", "red-cell F1,", "count unbiased"),
            ("23 KB", "int8 model", "in flash")],
           ["BCCD dataset", "364 labeled photos", "Fully convolutional", "Center heatmaps", "15.8K parameters"])


def architecture():
    """train.c's network, one box per layer, sized to stay legible at document width."""
    from xml.sax.saxutils import escape
    blocks = [  # (title, detail lines, output shape, fill, stroke)
        ("Input", ["RGB, normalized", "per channel"], "64×48×3", "#f1f5f9", "#64748b"),
        ("Conv · GELU", ["5×5, 8 filters", "stride 2"], "32×24×8", "#dbeafe", "#2563eb"),
        ("Conv · GELU", ["3×3, 16 filters", "same padding"], "32×24×16", "#dbeafe", "#2563eb"),
        ("Max pool", ["2×2", "stride 2"], "16×12×16", "#ccfbf1", "#0d9488"),
        ("Conv · GELU", ["3×3, 32 filters", "same padding"], "16×12×32", "#dbeafe", "#2563eb"),
        ("Conv · GELU", ["3×3, 32 filters", "same padding"], "16×12×32", "#dbeafe", "#2563eb"),
        ("Heatmaps", ["1×1 conv, sigmoid", "RBC · WBC · platelet"], "16×12×3", "#dcfce7", "#16a34a"),
    ]
    font = style.SVG_FONT
    bw, gap, x0, y0, bh = 182, 34, 30, 40, 120
    width = x0 * 2 + len(blocks) * bw + (len(blocks) - 1) * gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 252" width="{width}" height="252">',
           '  <title>Fully convolutional heatmap network (train.c)</title>',
           '  <defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#374151"/></marker></defs>',
           f'  <rect width="{width}" height="252" fill="{style.SURFACE}"/>']
    for i, (title, detail, shape, fill, stroke) in enumerate(blocks):
        x = x0 + i * (bw + gap)
        cx = x + bw / 2
        out.append(f'  <rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        out.append(f'  <text x="{cx}" y="{y0 + 38}" text-anchor="middle" font-family="{font}" font-size="19" '
                   f'font-weight="600" fill="#1f2937">{escape(title)}</text>')
        for k, line in enumerate(detail):
            out.append(f'  <text x="{cx}" y="{y0 + 66 + k * 22}" text-anchor="middle" font-family="{font}" '
                       f'font-size="15.5" fill="#374151">{escape(line)}</text>')
        out.append(f'  <text x="{cx}" y="{y0 + bh + 34}" text-anchor="middle" font-family="{font}" font-size="17" '
                   f'font-weight="600" fill="{style.INK_2}">{escape(shape)}</text>')
        if i:
            out.append(f'  <line x1="{x - gap + 2}" y1="{y0 + bh / 2}" x2="{x - 2}" y2="{y0 + bh / 2}" stroke="#374151" '
                       'stroke-width="2" marker-end="url(#arrow)"/>')
    out.append(f'  <text x="{width / 2}" y="236" text-anchor="middle" font-family="{font}" font-size="15" fill="{style.MUTED}">'
               '15,763 parameters · 4.0 million multiply-adds per photo · 23,156 bytes as an int8 inplace model</text>')
    out.append('</svg>\n')
    with open(os.path.join(OUT, "architecture.svg"), "w") as f:
        f.write("\n".join(out))


def boxes(name):
    """A photo's labeled boxes from test_boxes.txt: [(class, x0, y0, x1, y1)]."""
    for line in open(os.path.join(ROOT, "test_boxes.txt")):
        f = line.split()
        if f[0] == name:
            n = int(f[3])
            v = list(map(int, f[4:4 + 5 * n]))
            return [tuple(v[5 * i:5 * i + 5]) for i in range(n)]
    raise KeyError(name)


def draw_boxes(ax, name, lw=1.6):
    for cls, x0, y0, x1, y1 in boxes(name):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=CLASS_COLORS[cls], lw=lw))


def photo_axes(ax, name):
    ax.imshow(Image.open(os.path.join(JPEGS, name + ".jpg")))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def save_jpeg(fig, name):
    """Photographs go out as JPEGs: as PNGs they're ~1 MB each."""
    png = os.path.join(OUT, name + ".png")
    style.save(fig, png, raster=True)
    Image.open(png).convert("RGB").save(os.path.join(OUT, name + ".jpg"), quality=86, optimize=True)
    os.remove(png)


def class_legend(fig, ax, kind="box"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if kind == "box":
        handles = [Patch(facecolor="none", edgecolor=CLASS_COLORS[k], lw=1.6, label=CLASSES[k]) for k in range(3)]
    else:
        handles = [Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=CLASS_COLORS[k],
                          markeredgecolor="white", markeredgewidth=1.5, label=CLASSES[k]) for k in range(3)]
    return handles


def cells():
    """A labeled test photo, plus one cell of each class cropped at the same scale."""
    name = "BloodImage_00011"
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes([0.0, 0.0, 0.62, 1.0])
    photo_axes(ax, name)
    draw_boxes(ax, name)
    img = Image.open(os.path.join(JPEGS, name + ".jpg"))
    labeled = boxes(name)
    picks = [next(b for b in labeled if b[0] == k) for k in range(3)]
    for k, (cls, x0, y0, x1, y1) in enumerate(picks):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = 110  # every crop 220x220 photo pixels: the same scale for all three
        # Keep the crop inside the photo; the box is drawn relative to it.
        cx = min(max(cx, half), img.width - half)
        cy = min(max(cy, half), img.height - half)
        crop = img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
        a = fig.add_axes([0.655 + k * 0.118, 0.34, 0.11, 0.11 * 11 / 4.6])
        a.imshow(crop)
        a.add_patch(plt.Rectangle((x0 - (cx - half), y0 - (cy - half)), x1 - x0, y1 - y0, fill=False,
                                  edgecolor=CLASS_COLORS[cls], lw=1.6))
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.set_title(SHORT[cls] if cls != 2 else "Platelet", fontsize=10, loc="center", pad=4)
        a.text(0.5, -0.08, f"~{max(x1 - x0, y1 - y0)} px across", transform=a.transAxes, ha="center", va="top",
               fontsize=8.5, color=INK_2)
    fig.legend(handles=class_legend(fig, ax), loc="lower left", bbox_to_anchor=(0.655, 0.06), ncol=1, fontsize=9.5)
    save_jpeg(fig, "cells")


# predict.c's detections for BloodImage_00007 (the README's sample output):
# (class, x, y) in the 640x480 photo's pixel coordinates.
DETECTIONS = [(0, 338, 48), (0, 558, 72), (0, 101, 107), (0, 432, 148), (0, 90, 175), (0, 453, 220),
              (0, 550, 246), (0, 73, 344), (0, 221, 343), (0, 416, 349), (0, 549, 354), (0, 296, 377),
              (0, 135, 414), (1, 312, 179), (2, 227, 36)]


def detections():
    name = "BloodImage_00007"
    fig = plt.figure(figsize=(8.0, 6.6))
    ax = fig.add_axes([0, 0.1, 1, 0.9])
    photo_axes(ax, name)
    draw_boxes(ax, name, lw=1.4)
    for cls, x, y in DETECTIONS:
        ax.plot(x, y, "o", markersize=10, markerfacecolor=CLASS_COLORS[cls], markeredgecolor="white",
                markeredgewidth=2)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=MUTED, lw=1.4, label="labeled box"),
               Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=MUTED, markeredgecolor="white",
                      markeredgewidth=1.5, label="detected center")]
    handles += [Patch(color=CLASS_COLORS[k], label=CLASSES[k]) for k in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01), fontsize=9.5,
               columnspacing=1.2, handlelength=1.4)
    save_jpeg(fig, "detections")


# One trained model's raw output for BloodImage_00007, printed with
# nn_predict() after image_prep_frame() exactly as predict.c runs them (the
# model behind figure 3 only; the README's other numbers come from the five
# models it reports).
HEATMAPS = [  # 3 classes x 12 rows x 16 columns
    [
        [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.13, 0.17, 0.04, 0.01, 0.00, 0.00, 0.15, 0.16, 0.03],
        [0.02, 0.08, 0.15, 0.05, 0.17, 0.11, 0.14, 0.22, 0.22, 0.05, 0.01, 0.00, 0.01, 0.58, 0.64, 0.06],
        [0.05, 0.21, 0.35, 0.46, 0.44, 0.24, 0.24, 0.09, 0.04, 0.11, 0.12, 0.01, 0.03, 0.37, 0.67, 0.12],
        [0.05, 0.19, 0.30, 0.28, 0.18, 0.15, 0.09, 0.02, 0.01, 0.12, 0.44, 0.22, 0.03, 0.11, 0.26, 0.08],
        [0.08, 0.35, 0.60, 0.06, 0.02, 0.04, 0.05, 0.01, 0.01, 0.02, 0.26, 0.37, 0.13, 0.16, 0.09, 0.05],
        [0.03, 0.17, 0.19, 0.01, 0.00, 0.01, 0.02, 0.02, 0.01, 0.01, 0.53, 0.49, 0.13, 0.59, 0.27, 0.07],
        [0.02, 0.01, 0.00, 0.00, 0.01, 0.03, 0.07, 0.03, 0.01, 0.01, 0.34, 0.50, 0.11, 0.62, 0.27, 0.03],
        [0.06, 0.18, 0.10, 0.01, 0.24, 0.47, 0.18, 0.02, 0.01, 0.02, 0.23, 0.10, 0.04, 0.19, 0.03, 0.00],
        [0.19, 0.93, 0.74, 0.11, 0.38, 0.84, 0.30, 0.12, 0.09, 0.31, 0.61, 0.16, 0.24, 0.83, 0.25, 0.01],
        [0.10, 0.30, 0.34, 0.30, 0.32, 0.22, 0.25, 0.35, 0.33, 0.60, 0.46, 0.10, 0.14, 0.56, 0.30, 0.04],
        [0.01, 0.02, 0.39, 0.68, 0.16, 0.02, 0.04, 0.16, 0.19, 0.24, 0.06, 0.01, 0.01, 0.11, 0.70, 0.34],
        [0.00, 0.01, 0.20, 0.29, 0.03, 0.01, 0.04, 0.37, 0.22, 0.04, 0.01, 0.00, 0.00, 0.02, 0.28, 0.24],
    ],
    [
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01],
        [0.01, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.04, 0.03, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.06, 0.39, 0.19, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.15, 0.66, 0.39, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.06, 0.29, 0.25, 0.03, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01],
        [0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01],
        [0.00, 0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01],
    ],
    [
        [0.00, 0.00, 0.00, 0.00, 0.24, 0.81, 0.25, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.04, 0.51, 0.16, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.03, 0.01, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01],
        [0.01, 0.00, 0.00, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.01, 0.02, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.01, 0.01, 0.04, 0.03, 0.01, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.04, 0.01, 0.02, 0.01, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.01, 0.04, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.01],
        [0.00, 0.00, 0.01, 0.05, 0.03, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01],
        [0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.01, 0.00, 0.00, 0.01],
        [0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02],
    ],
]


def heatmaps():
    """The network's three output heatmaps for one photo, beside the photo."""
    from matplotlib.colors import LinearSegmentedColormap
    name = "BloodImage_00007"
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.55), gridspec_kw={"wspace": 0.06})
    photo = Image.open(os.path.join(JPEGS, name + ".jpg"))
    axes[0].imshow(photo.resize((64, 48), Image.Resampling.BOX), interpolation="nearest")
    axes[0].set_title("Network input", loc="left", fontsize=10)
    for k in range(3):
        cmap = LinearSegmentedColormap.from_list(SHORT[k], [style.SURFACE, CLASS_COLORS[k]])
        ax = axes[k + 1]
        ax.imshow(np.array(HEATMAPS[k]), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{SHORT[k].capitalize() if k == 2 else SHORT[k]} heatmap", loc="left", fontsize=10)
        ax.set_xticks(np.arange(-0.5, 16), minor=True)
        ax.set_yticks(np.arange(-0.5, 12), minor=True)
        ax.grid(which="minor", color=style.GRID, linewidth=0.6)
        ax.tick_params(which="minor", length=0)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(which="major", visible=False)
        for sp in ax.spines.values():
            sp.set_color(style.GRID)
    style.save(fig, os.path.join(OUT, "heatmaps.png"), raster=True)


def threshold_sweep():
    """test.c's validation sweep, sample run."""
    t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    f1 = [[74.5, 76.4, 78.2, 78.6, 77.6, 74.0, 68.1, 57.0, 35.1],
          [99.4, 99.4, 99.4, 99.4, 98.3, 97.7, 92.7, 78.6, 38.9],
          [69.7, 75.8, 77.0, 78.8, 77.4, 74.5, 75.6, 69.0, 36.9]]
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    ax.axvline(0.4, color=style.AXIS, lw=1)
    ax.text(0.41, 4, "HEATMAP_THRESHOLD = 0.4", fontsize=8.5, color=MUTED)
    for k in range(3):
        ax.plot(t, f1[k], color=CLASS_COLORS[k], marker="o", markersize=5, markeredgecolor=style.SURFACE,
                markeredgewidth=1.2, label=CLASSES[k])
    ax.set_xticks(t, [f"{v:.1f}" for v in t])
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("peak threshold")
    ax.set_ylabel("validation F1")
    fig.subplots_adjust(bottom=0.2)
    style.bottom_legend(fig, ax, 3)
    style.save(fig, os.path.join(OUT, "threshold.svg"))


def results():
    """Test-set precision and recall per class, mean of five models with their range."""
    prec = [(77.2, 76.0, 78.4), (94.7, 94.7, 94.7), (76.5, 75.0, 79.0)]
    rec = [(77.3, 74.0, 78.9), (100.0, 100.0, 100.0), (69.0, 60.9, 73.9)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(3)
    w = 0.16
    for k, (vals, name, color) in enumerate([(prec, "precision", SERIES[6]), (rec, "recall", SERIES[3])]):
        xs = x + (k - 0.5) * (w + 0.02)
        means = [v[0] for v in vals]
        ax.bar(xs, means, width=w, color=color, label=name)
        for xi, (m, lo, hi) in zip(xs, vals):
            if hi > lo:
                ax.plot([xi, xi], [lo, hi], color=style.INK, lw=1.2)
            ax.text(xi, max(m, hi) + 2, f"{m:.0f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, CLASSES)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    fig.subplots_adjust(bottom=0.2)
    style.bottom_legend(fig, ax, 2)
    style.save(fig, os.path.join(OUT, "results.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    architecture()
    cells()
    heatmaps()
    detections()
    threshold_sweep()
    results()
    print("wrote", sorted(os.listdir(OUT)))
