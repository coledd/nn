# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/image_classification/README.md. The photo grid is
# read from the downloaded Kaggle Cats and Dogs images (run make in the
# example first); the charts plot the results the README reports.
#
#   python3 tools/docs/image_classification.py

import os

import numpy as np
from PIL import Image

import style
from banner import banner
from style import INK_2, MUTED, SERIES, plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "image_classification")
OUT = os.path.join(ROOT, "figures")
IMG_SIZE = 48  # image_prep.h

# Species colors, used everywhere: cat takes the first categorical slot, dog the second.
CAT, DOG = SERIES[0], SERIES[1]


def make_banner():
    banner(os.path.join(OUT, "banner.svg"), 5, "Computer vision · real data",
           "Image Classification",
           ["Telling a cat from a dog in an ordinary photo, on",
            "the microcontroller in a camera-equipped pet door."],
           [("86%", "accuracy, balanced", "Kaggle test set"),
            ("36 KB", "int8 model", "in flash"),
            ("25K", "parameters,", "4-stage CNN")],
           ["Kaggle Cats and Dogs + Oxford-IIIT Pet", "~23,300 training photos", "48×48 RGB input",
            "Global average pooling"])


def architecture():
    """train.c's network, one box per stage (convolution + pooling), so the
    diagram stays legible at document width."""
    from xml.sax.saxutils import escape
    blocks = [  # (title, detail lines, output shape, fill, stroke)
        ("Input", ["RGB, normalized", "per image"], "48×48×3", "#f1f5f9", "#64748b"),
        ("Stage 1", ["5×5 conv, 8 filters", "GELU · 2×2 max pool"], "24×24×8", "#dbeafe", "#2563eb"),
        ("Stage 2", ["3×3 conv, 16 filters", "GELU · 2×2 max pool"], "12×12×16", "#dbeafe", "#2563eb"),
        ("Stage 3", ["3×3 conv, 32 filters", "GELU · 2×2 max pool"], "6×6×32", "#dbeafe", "#2563eb"),
        ("Stage 4", ["3×3 conv, 64 filters", "GELU · 2×2 max pool"], "3×3×64", "#dbeafe", "#2563eb"),
        ("Global avg pool", ["one value per", "feature map"], "1×1×64", "#ccfbf1", "#0d9488"),
        ("Output", ["softmax", "cat · dog"], "2", "#dcfce7", "#16a34a"),
    ]
    font = style.SVG_FONT
    bw, gap, x0, y0, bh = 182, 34, 30, 40, 120
    width = x0 * 2 + len(blocks) * bw + (len(blocks) - 1) * gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 252" width="{width}" height="252">',
           '  <title>Image classification CNN (train.c)</title>',
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
               '25,042 parameters · 3.4 million multiply-adds per image · 36,244 bytes as an int8 inplace model</text>')
    out.append('</svg>\n')
    with open(os.path.join(OUT, "architecture.svg"), "w") as f:
        f.write("\n".join(out))


def center_square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def network_input(img):
    """image_prep.c's resize: the whole-frame center square, area-averaged to
    48x48. (The per-image normalization that follows doesn't change how it looks.)"""
    return center_square(img).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BOX)


def photos():
    """Held-out Kaggle test photos (numbers 1 mod 10) and what the network sees of them."""
    pets = os.path.join(ROOT, "dataset", "PetImages")
    rows = [("Cat", [1, 21, 41, 71, 111, 151]), ("Dog", [11, 31, 51, 91, 101, 141])]
    fig, axes = plt.subplots(4, 6, figsize=(11, 7.6), gridspec_kw={"hspace": 0.06, "wspace": 0.05})
    for r, (species, numbers) in enumerate(rows):
        for c, n in enumerate(numbers):
            img = Image.open(os.path.join(pets, species, f"{n}.jpg")).convert("RGB")
            top, bottom = axes[2 * r, c], axes[2 * r + 1, c]
            top.imshow(center_square(img).resize((320, 320), Image.Resampling.LANCZOS))
            bottom.imshow(network_input(img), interpolation="nearest")
            for ax in (top, bottom):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
        axes[2 * r, 0].set_ylabel(f"{species}s", rotation=0, ha="right", va="center", color=INK_2,
                                  fontsize=10, labelpad=10)
        axes[2 * r + 1, 0].set_ylabel("network\ninput, 48×48", rotation=0, ha="right", va="center",
                                      color=MUTED, fontsize=9.5, labelpad=10)
    # A JPEG, unlike the other figures: it's photographs, and as a PNG it's 1.7 MB.
    png = os.path.join(OUT, "photos.png")
    style.save(fig, png, raster=True)
    Image.open(png).convert("RGB").save(os.path.join(OUT, "photos.jpg"), quality=86, optimize=True)
    os.remove(png)


def accuracy():
    """test.c's sample run (one model), with the overall range across five runs."""
    groups = [("Kaggle test set\n(balanced; guessing 50%)", 86.36, (85.2, 86.7), 87.2, 85.5, 50.0),
              ("Oxford-IIIT test set\n(2:1 dogs; always \"dog\" 67.8%)", 84.22, (83.9, 84.9), 77.3, 87.5, 67.8)]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    w = 0.12
    for g, (name, overall, (lo, hi), cat, dog, baseline) in enumerate(groups):
        xs = [g - w - 0.02, g, g + w + 0.02]
        for x, v, color, label in zip(xs, [overall, cat, dog], [style.INK_2, CAT, DOG],
                                      ["overall", "cats correct", "dogs correct"]):
            ax.bar(x, v, width=w, color=color, label=label if g == 0 else None)
            ax.text(x, v + 1.5, f"{v:.1f}", ha="center", fontsize=9, color=INK_2)
        ax.plot([xs[0], xs[0]], [lo, hi], color=style.INK, lw=1.2)
        ax.plot([g - 0.48, g + 0.48], [baseline, baseline], color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(g + 0.48, baseline - 1.2, "baseline", ha="right", va="top", fontsize=8.5, color=MUTED)
    ax.set_xticks([0, 1], [g[0] for g in groups], fontsize=9.5)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("accuracy")
    fig.subplots_adjust(bottom=0.2)
    style.bottom_legend(fig, ax, 3)
    style.save(fig, os.path.join(OUT, "accuracy.svg"))


def breeds():
    """test.c's per-breed table, sample run: the five hardest and four easiest breeds."""
    rows = [("Sphynx", "cat", 51.0), ("Scottish Terrier", "dog", 62.6), ("Bengal", "cat", 71.0),
            ("Japanese Chin", "dog", 71.0), ("Egyptian Mau", "cat", 73.2),
            ("English Setter", "dog", 96.0), ("English Cocker Spaniel", "dog", 97.0),
            ("Saint Bernard", "dog", 97.0), ("Basset Hound", "dog", 98.0)]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ys = list(range(len(rows)))
    ys = [y if y < 5 else y + 0.8 for y in ys]  # a gap between the hardest and the easiest
    for y, (name, species, v) in zip(ys, rows):
        ax.barh(y, v, height=0.5, color=CAT if species == "cat" else DOG,
                label={"cat": "cat breed", "dog": "dog breed"}[species])
        ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.annotate("hardest five", xy=(0, -0.75), xycoords=("axes fraction", "data"), fontsize=9, color=MUTED)
    ax.annotate("easiest four", xy=(0, 5.0), xycoords=("axes fraction", "data"), fontsize=9, color=MUTED,
                va="bottom")
    ax.set_xlabel("photos of the breed classified correctly, Oxford-IIIT test set")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.canvas.draw()
    bottom = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y0
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, bottom - 0.02))
    style.save(fig, os.path.join(OUT, "breeds.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    architecture()
    photos()
    accuracy()
    breeds()
    print("wrote", sorted(os.listdir(OUT)))
