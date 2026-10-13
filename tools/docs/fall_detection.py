# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/fall_detection/README.md. The accelerometer streams
# come from the example's own generator (fall_data.c, compiled here into a
# small harness with a fixed seed); the charts plot the results the README
# reports.
#
#   python3 tools/docs/fall_detection.py

import os
import subprocess
import tempfile

import numpy as np

import style
from banner import banner
from style import INK, INK_2, MUTED, SERIES, plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "fall_detection")
OUT = os.path.join(ROOT, "figures")
MODELS = ["RNN", "GRU", "LSTM"]

# Measured with the shipped code: seven train + test runs per architecture
# (train_<arch>, then test on 50 normal + 50 fall sequences).
RUNS = {
    "RNN": {"epochs": [86, 12, 20, 13, 13, 18, 8],
            "accuracy": [100.00, 96.95, 99.35, 99.13, 100.00, 99.75, 97.47],
            "recall": [100, 100, 100, 100, 100, 100, 100], "false_alarms": [0, 0, 0, 0, 0, 0, 0]},
    "GRU": {"epochs": [8, 10, 18, 6, 8, 10, 10],
            "accuracy": [97.02, 97.08, 98.55, 97.16, 99.27, 99.41, 98.88],
            "recall": [100, 100, 100, 92, 100, 88, 98], "false_alarms": [0, 0, 0, 2, 0, 30, 0]},
    "LSTM": {"epochs": [30, 30, 25, 14, 39, 32, 42],
             "accuracy": [97.99, 99.69, 99.94, 99.94, 99.84, 99.29, 98.91],
             "recall": [100, 100, 100, 100, 100, 100, 100], "false_alarms": [0, 0, 0, 0, 0, 0, 0]},
}

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include "fall_data.h"
int main(void) {
  float w[SEQUENCE_LEN][NUM_AXES], l[SEQUENCE_LEN];
  srand(7);
  generate_normal_sequence(w, l);
  for (int t = 0; t < SEQUENCE_LEN; t++) printf("0 %f %f %f %f\n", w[t][0], w[t][1], w[t][2], l[t]);
  generate_fall_sequence(w, l);
  for (int t = 0; t < SEQUENCE_LEN; t++) printf("1 %f %f %f %f\n", w[t][0], w[t][1], w[t][2], l[t]);
  return 0;
}
"""


def sequences():
    """One normal and one fall sequence from fall_data.c."""
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = os.path.join(tmp, "dump.c"), os.path.join(tmp, "dump")
        open(src, "w").write(HARNESS)
        subprocess.run(["cc", "-O2", "-I", ROOT, src, os.path.join(ROOT, "fall_data.c"), "-lm", "-o", exe], check=True)
        rows = np.loadtxt(subprocess.run([exe], capture_output=True, text=True, check=True).stdout.splitlines())
    return rows[rows[:, 0] == 0][:, 1:], rows[rows[:, 0] == 1][:, 1:]


def make_banner():
    banner(os.path.join(OUT, "banner.svg"), 3, "Wearables · simulated data",
           "Fall Detection",
           ["Watching a live accelerometer stream for the free-fall,",
            "impact and stillness of a fall, with three recurrent layers."],
           [("100%", "falls detected,", "RNN and LSTM"),
            ("0", "false alarms,", "RNN and LSTM"),
            ("337", "parameters,", "smallest model")],
           ["3-axis accelerometer", "Synthetic data", "RNN · GRU · LSTM", "150-step windows", "Per-timestep labels"])


def stream_figure():
    normal, fall = sequences()
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    t = np.arange(normal.shape[0])
    for ax, seq, title in [(axes[0], normal, "Normal activity"), (axes[1], fall, "Fall")]:
        mag = np.linalg.norm(seq[:, :3], axis=1)
        onset = np.argmax(seq[:, 3] > 0.5) if seq[:, 3].max() > 0.5 else None
        if onset is not None:
            ax.axvspan(onset, t[-1] + 0.5, color=SERIES[7], alpha=0.07, lw=0)
        for k, name in enumerate(["x", "y", "z"]):
            ax.plot(t, seq[:, k], color=SERIES[k], lw=1.2, label=f"{name} axis")
        ax.plot(t, mag, color=INK, lw=2, label="|a|, magnitude")
        ax.set_ylim(-2.2, 4.6)
        ax.set_yticks([-2, 0, 2, 4], ["−2 g", "0 g", "2 g", "4 g"])
        ax.set_title(title, loc="left")
        if onset is not None:
            dip = onset + np.argmin(mag[onset:onset + 10])
            peak = onset + np.argmax(mag[onset:onset + 12])
            ax.annotate("free fall: |a| near 0 g", (dip, mag[dip]), xytext=(dip - 38, -1.7), fontsize=9.5,
                        color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
            ax.annotate("impact", (peak, mag[peak]), xytext=(peak - 28, 3.7), fontsize=9.5, color=INK_2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
            ax.annotate("stillness at a new\norientation", (peak + 12, mag[peak + 12]), xytext=(peak + 5, 2.4),
                        fontsize=9.5, color=INK_2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].set_xlabel("timestep")
    axes[1].set_xlim(0, t[-1])
    fig.subplots_adjust(hspace=0.3)
    style.bottom_legend(fig, axes[1], 4)
    style.save(fig, os.path.join(OUT, "stream.svg"))


def runs_figure():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    rng = np.random.default_rng(1)
    panels = [("epochs", "Training epochs to early stop", None),
              ("accuracy", "Per-timestep test accuracy", (95, 100.6)),
              ("recall", "Falls detected (recall)", (85, 101.5))]
    for ax, (key, title, ylim) in zip(axes, panels):
        for i, m in enumerate(MODELS):
            ys = np.array(RUNS[m][key], float)
            xs = i + rng.uniform(-0.12, 0.12, len(ys))
            ax.scatter(xs, ys, s=42, color=SERIES[i], edgecolor=style.SURFACE, linewidth=1.5, zorder=3)
            ax.plot([i - 0.22, i + 0.22], [np.median(ys)] * 2, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(3), MODELS)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(title, loc="left")
        if ylim:
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        else:
            ax.set_ylim(0, 95)
    axes[2].annotate("30% false alarms\nin this run", (1.0, 88), xytext=(1.35, 90), fontsize=8.5, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    fig.subplots_adjust(wspace=0.3)
    style.save(fig, os.path.join(OUT, "runs.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    stream_figure()
    runs_figure()
    print("wrote", sorted(os.listdir(OUT)))
