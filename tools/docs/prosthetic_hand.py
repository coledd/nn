# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/prosthetic_hand/README.md. The EMG traces are read
# from the downloaded Ninapro recordings (run make in the example first);
# the charts plot the results the README reports.
#
#   python3 tools/docs/prosthetic_hand.py

import os
import struct
import zlib

import numpy as np

import style
from banner import banner
from style import INK, INK_2, MUTED, SERIES, REST_GRAY, plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "prosthetic_hand")
OUT = os.path.join(ROOT, "figures")

INTENTS = ["REST", "HAND_OPEN", "HAND_CLOSE", "WRIST_PRONATION", "WRIST_SUPINATION", "WRIST_FLEXION", "WRIST_EXTENSION"]
LABELS = ["Rest", "Open hand", "Close hand", "Pronation", "Supination", "Flexion", "Extension"]
# One fixed color per intent, everywhere: rest recedes, the six movements
# take the first six categorical slots in order.
INTENT_COLORS = [REST_GRAY] + SERIES[:6]


def make_banner():
    banner(os.path.join(OUT, "banner.svg"), 8, "Embedded health · real patient data",
           "Prosthetic Hand",
           ["Decoding hand and wrist movements from forearm",
            "muscle signals, calibrated to each wearer on the device."],
           [("78%", "window accuracy,", "intact, 4 reps"),
            ("−55%", "wrong movements", "with rejection"),
            ("4.4K", "parameters,", "fine-tuned on-device")],
           ["Surface EMG", "Ninapro DB2 + DB3", "51 people, 11 amputees", "STM32U585 + ADS1298", "FC network"])


def load_mat(path, names):
    """Minimal MATLAB v5 reader for Ninapro's files (compressed numeric matrices)."""
    d = open(path, "rb").read()
    pos, out = 128, {}
    while pos < len(d):
        typ, n = struct.unpack("<II", d[pos:pos + 8])
        if typ == 15:
            x = zlib.decompress(d[pos + 8:pos + 8 + n])
            q = 8 + 16
            _, dn = struct.unpack("<II", x[q:q + 8])
            dims = struct.unpack("<" + "i" * (dn // 4), x[q + 8:q + 8 + dn])
            q += 8 + (dn + 7) // 8 * 8
            nt, nn_ = struct.unpack("<II", x[q:q + 8])
            if nt >> 16:
                name = x[q + 4:q + 4 + (nt >> 16)].decode()
                q += 8
            else:
                name = x[q + 8:q + 8 + nn_].decode()
                q += 8 + (nn_ + 7) // 8 * 8
            rt, rn = struct.unpack("<II", x[q:q + 8])
            if name in names:
                dtype = {7: "<f4", 9: "<f8", 1: "i1", 2: "u1"}[rt & 0xffff]
                out[name] = np.frombuffer(x[q + 8:q + 8 + rn], dtype=dtype).reshape(dims[1], dims[0]).T
        pos += 8 + n
    return out


def emg_traces():
    """Raw EMG on the 8 ring electrodes for one subject, one repetition of
    four movements, with a second of rest before each."""
    mat = load_mat(os.path.join(ROOT, "dataset", "db2_s33_e1.mat"), {"emg", "restimulus", "rerepetition"})
    emg = mat["emg"][:, :8] * 1e6  # volts -> microvolts
    stim = mat["restimulus"][:, 0].astype(int)
    rep = mat["rerepetition"][:, 0].astype(int)
    fs = 2000
    shown = [(6, "Close hand"), (13, "Wrist flexion"), (14, "Wrist extension"), (10, "Wrist pronation")]
    fig, axes = plt.subplots(8, len(shown), figsize=(11, 6.2), sharex=True, sharey="row")
    segments = []
    for movement, name in shown:
        onset = np.where((stim == movement) & (rep == 2))[0][0]
        length = np.sum((stim == movement) & (rep == 2))
        segments.append((onset, length))
    # Each electrode on its own scale, set by its largest swing in the
    # windows shown, so nothing clips and quiet channels stay readable.
    lo, hi = -1.0, 5.5
    scale = [1.05 * max(np.abs(emg[o + int(lo * fs):o + int(hi * fs), c]).max() for o, _ in segments) for c in range(8)]
    for j, ((movement, name), (onset, length)) in enumerate(zip(shown, segments)):
        start, end = onset + int(lo * fs), onset + int(hi * fs)
        t = (np.arange(start, end) - onset) / fs
        for c in range(8):
            ax = axes[c, j]
            ax.axvspan(0, length / fs, color=SERIES[0], alpha=0.08, lw=0)
            ax.plot(t, emg[start:end, c], color=INK_2, lw=0.3)
            ax.set_ylim(-scale[c], scale[c])
            ax.set_xlim(lo, hi)
            ax.grid(False)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(c == 7)
            if j == 0:
                ax.set_ylabel(f"ch{c + 1}", rotation=0, ha="right", va="center", color=INK_2, fontsize=9.5)
            if c == 0:
                ax.set_title(name, loc="left", fontsize=10.5, pad=6)
        axes[7, j].set_xlabel("seconds from movement onset", fontsize=9)
        axes[7, j].set_xticks([0, 2, 4])
    fig.subplots_adjust(hspace=0.12, wspace=0.08)
    style.save(fig, os.path.join(OUT, "emg_traces.png"), raster=True)


def activation_heatmap():
    """train.c's per-intent mean electrode activity (standardized log MAV)."""
    data = np.array([
        [-0.91, -0.95, -1.01, -0.96, -0.93, -0.96, -1.05, -1.14],
        [-0.01, +0.08, +0.17, +0.08, -0.12, -0.07, +0.12, +0.20],
        [+0.49, +0.89, +0.51, +0.13, +0.13, +0.35, +0.43, +0.18],
        [+0.09, -0.20, -0.17, +0.01, +0.10, +0.12, +0.05, +0.11],
        [-0.10, -0.08, -0.04, -0.19, -0.00, +0.13, +0.07, +0.11],
        [+0.37, +0.54, +0.84, +1.27, +0.92, +0.22, +0.14, +0.26],
        [+0.57, +0.28, +0.30, +0.32, +0.47, +0.70, +0.77, +0.83]])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "div", [style.BLUE_RAMP[10], style.BLUE_RAMP[4], style.NEUTRAL_MID, style.RED_RAMP[3], style.RED_RAMP[7]])
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.imshow(data, cmap=cmap, vmin=-1.3, vmax=1.3, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(v) > 0.75 else INK)
    ax.set_xticks(range(8), [f"ch{c + 1}" for c in range(8)])
    ax.set_yticks(range(7), LABELS)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticks(np.arange(-0.5, 8), minor=True)
    ax.set_yticks(np.arange(-0.5, 7), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    style.save(fig, os.path.join(OUT, "activation.svg"))


def calibration_curves():
    reps = [1, 2, 4]
    groups = [("Intact subjects (8 held out)", {
                  "LDA": [63.8, 68.7, 71.2], "NN, from scratch": [64.6, 73.1, 76.6],
                  "NN, shared model fine-tuned": [66.8, 74.0, 77.9], "NN, shared model as is": [51.7, 51.9, 52.0]}),
              ("Amputees (11 held out)", {
                  "LDA": [44.1, 50.2, 55.2], "NN, from scratch": [44.7, 51.9, 55.9],
                  "NN, shared model fine-tuned": [44.6, 52.1, 56.9], "NN, shared model as is": [24.6, 26.6, 26.6]})]
    colors = {"LDA": SERIES[0], "NN, from scratch": SERIES[1], "NN, shared model fine-tuned": SERIES[2],
              "NN, shared model as is": MUTED}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    for ax, (title, series) in zip(axes, groups):
        for name, ys in series.items():
            ls = (0, (4, 3)) if name.endswith("as is") else "-"
            ax.plot(reps, ys, color=colors[name], ls=ls, marker="o", markersize=6,
                    markeredgecolor=style.SURFACE, markeredgewidth=1.5, label=name)
        ax.set_xticks(reps, ["1 rep", "2 reps", "4 reps"])
        ax.set_xlim(0.8, 4.2)
        ax.set_ylim(0, 85)
        ax.set_yticks(range(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
        ax.set_title(title, loc="left")
        ax.set_xlabel("calibration data (1 rep = one ~5 s hold of each movement)", fontsize=9)
    axes[0].set_ylabel("window accuracy, repetitions 2 and 5")
    for ax in axes:
        ax.axhline(100 / 7, color=style.AXIS, lw=1)
        ax.text(4.15, 100 / 7 + 1.2, "chance, 7 classes", fontsize=8.5, color=MUTED, ha="right", va="bottom")
    fig.subplots_adjust(wspace=0.08)
    style.bottom_legend(fig, axes[0], 4)
    style.save(fig, os.path.join(OUT, "calibration.svg"))


def control_tradeoff():
    """replay.c's sweep: what confidence rejection trades."""
    thresholds = ["none", "0.7", "0.8", "0.9"]
    groups = [("Intact subjects", {"correct": [59.7, 53.3, 50.0, 44.7], "wrong movement": [26.5, 17.2, 13.8, 9.4],
                                   "movement at rest": [17.5, 11.0, 8.5, 5.7], "response (s)": [0.467, 0.583, 0.655, 0.729]}),
              ("Amputees", {"correct": [40.0, 29.8, 26.2, 21.4], "wrong movement": [48.4, 26.8, 20.9, 14.0],
                            "movement at rest": [46.6, 24.1, 17.9, 11.5], "response (s)": [0.514, 0.833, 0.967, 1.212]})]
    colors = {"correct": SERIES[2], "wrong movement": SERIES[7], "movement at rest": SERIES[1]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    x = np.arange(4)
    for ax, (title, s) in zip(axes, groups):
        for name in ("correct", "wrong movement", "movement at rest"):
            ax.plot(x, s[name], color=colors[name], marker="o", markersize=6,
                    markeredgecolor=style.SURFACE, markeredgewidth=1.5, label=name)
            style.end_label(ax, x[-1], s[name][-1], f"{s[name][-1]:.0f}%")
        ax.set_xticks(x, [f"{t}\n{r:.2f} s" for t, r in zip(thresholds, s["response (s)"])])
        ax.set_xlim(-0.2, 3.45)
        ax.set_ylim(0, 65)
        ax.set_yticks(range(0, 61, 20), [f"{v}%" for v in range(0, 61, 20)])
        ax.set_title(title, loc="left")
        ax.set_xlabel("rejection threshold  /  mean response time", fontsize=9)
    axes[0].set_ylabel("share of 50 ms decisions")
    fig.subplots_adjust(wspace=0.1)
    style.bottom_legend(fig, axes[0], 3)
    style.save(fig, os.path.join(OUT, "rejection.svg"))


def electrodes():
    counts = ["8\nforearm ring", "10\n+ finger flexor,\nextensor", "12\n+ biceps,\ntriceps"]
    groups = [("Intact subjects", [71.2, 73.2, 78.1], [77.9, 80.7, 81.4]),
              ("Amputees", [55.2, 57.3, 62.6], [56.9, 58.6, 62.5])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    x = np.arange(3)
    w = 0.2
    for ax, (title, lda, nn) in zip(axes, groups):
        for k, (vals, name, color) in enumerate([(lda, "LDA", SERIES[0]), (nn, "NN, fine-tuned", SERIES[2])]):
            xs = x + (k - 0.5) * (w + 0.02)
            ax.bar(xs, vals, width=w, color=color, label=name)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=9, color=INK_2)
        ax.set_xticks(x, counts, fontsize=9)
        ax.set_ylim(0, 92)
        ax.set_yticks(range(0, 81, 20), [f"{v}%" for v in range(0, 81, 20)])
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("window accuracy, 4 reps of calibration")
    fig.subplots_adjust(wspace=0.08)
    style.bottom_legend(fig, axes[0], 2)
    style.save(fig, os.path.join(OUT, "electrodes.svg"))


# replay.c's timeline for DB2 subject 33, repetition 2 (one character per
# 50ms decision; see the README's sample output).
TIMELINE = {
    "HAND_OPEN": ("....................OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO....................",
                  "...............................................OOOO...OOOOSSSSOOOOOSSSS....OOOOOOO...OOOO...OOOOCOO...............PPPPPPP......................................."),
    "HAND_CLOSE": ("....................CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC....................",
                   ".......................CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC..................."),
    "WRIST_PRONATION": ("....................PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP....................",
                        "........................................SSSSSPPPPPPPPPPPPPPP...PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP..........................................."),
    "WRIST_SUPINATION": ("....................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS....................",
                         "P..........PPPPPPPPPPPPPPP...SSSS.....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSOOOOOOOOOOOO.................PP................PPP"),
    "WRIST_FLEXION": ("....................FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF....................",
                      "..........................EEEE...EEEEEEEEEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF....EEEEEEEEEEEEEEEEEEP.PP.........."),
    "WRIST_EXTENSION": ("....................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....................",
                        "............................EEEEE.....EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE......EEEE............"),
}
CHAR_INTENT = {".": 0, "O": 1, "C": 2, "P": 3, "S": 4, "F": 5, "E": 6}


def timeline():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    row = 0
    yt, yl = [], []
    for name, (actual, decoded) in TIMELINE.items():
        for k, (line, kind) in enumerate([(actual, "labeled"), (decoded, "decoded")]):
            y = row * 2.6 + k * 1.0
            for start, length, ch in style.runs(line):
                ax.add_patch(plt.Rectangle(((start - 20) * 0.05, y), length * 0.05, 0.82,
                                           color=INTENT_COLORS[CHAR_INTENT[ch]], lw=0))
            yt.append(y + 0.41)
            yl.append(f"{LABELS[INTENTS.index(name)]}" if k == 0 else "decoded")
        row += 1
    ax.set_xlim(-1.05, 7.4)
    ax.set_ylim(row * 2.6 - 0.6, -0.4)
    ax.set_yticks(yt, yl, fontsize=9)
    for lab, kind in zip(ax.get_yticklabels(), yl):
        lab.set_color(MUTED if kind == "decoded" else INK)
    ax.set_xlabel("seconds from labeled onset")
    ax.grid(False)
    ax.axvline(0, color=INK_2, lw=0.8)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    from matplotlib.patches import Patch
    handles = [Patch(color=INTENT_COLORS[i], label=LABELS[i]) for i in range(7)]
    ax.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.45, -0.32), handlelength=1.2,
              columnspacing=1.2)
    style.save(fig, os.path.join(OUT, "timeline.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    emg_traces()
    activation_heatmap()
    calibration_curves()
    control_tradeoff()
    electrodes()
    timeline()
    print("wrote", sorted(os.listdir(OUT)))
