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

# Measured with the shipped code: twenty train + test runs per architecture
# (train_<arch>, then test on 50 normal + 50 fall sequences). Every run
# detected all 50 falls with no false alarms, so only epochs and
# per-timestep accuracy vary.
RUNS = {
    "RNN": {"epochs": [34, 25, 65, 96, 104, 500, 34, 24, 26, 83, 54, 500, 17, 66, 40, 25, 28, 16, 33, 33],
            "accuracy": [100.00, 99.99, 100.00, 100.00, 100.00, 100.00, 99.99, 100.00, 99.99, 100.00,
                         100.00, 100.00, 100.00, 100.00, 100.00, 99.69, 99.99, 99.99, 100.00, 99.87]},
    "GRU": {"epochs": [101, 386, 278, 97, 74, 199, 118, 36, 35, 185, 114, 43, 89, 90, 84, 86, 113, 89, 148, 185],
            "accuracy": [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.99, 100.00, 100.00,
                         100.00, 100.00, 100.00, 100.00, 100.00, 99.93, 100.00, 100.00, 100.00, 100.00]},
    "LSTM": {"epochs": [83, 29, 18, 116, 46, 127, 19, 38, 37, 187, 24, 35, 52, 27, 18, 29, 105, 90, 44, 104],
             "accuracy": [99.83, 99.75, 99.69, 99.73, 98.74, 100.00, 99.33, 99.95, 99.49, 100.00,
                          99.55, 99.97, 99.97, 98.98, 99.20, 99.63, 99.93, 99.99, 99.95, 100.00]},
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
           [("100%", "falls detected,", "all 60 runs"),
            ("0", "false alarms,", "all 60 runs"),
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    rng = np.random.default_rng(1)
    panels = [("epochs", "Training epochs before early stopping"),
              ("accuracy", "Per-timestep test accuracy")]
    for ax, (key, title) in zip(axes, panels):
        for i, m in enumerate(MODELS):
            ys = np.array(RUNS[m][key], float)
            xs = i + rng.uniform(-0.14, 0.14, len(ys))
            ax.scatter(xs, ys, s=36, color=SERIES[i], edgecolor=style.SURFACE, linewidth=1.5, zorder=3)
            ax.plot([i - 0.24, i + 0.24], [np.median(ys)] * 2, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(3), MODELS)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(title, loc="left")
    axes[0].set_yscale("log")
    axes[0].set_ylim(10, 700)
    axes[0].set_yticks([10, 20, 50, 100, 200, 500], ["10", "20", "50", "100", "200", "500"])
    axes[0].minorticks_off()
    axes[0].annotate("500-epoch cap", (0.0, 500), xytext=(0.3, 560), fontsize=8.5, color=INK_2, va="center")
    axes[1].set_ylim(98.5, 100.1)
    axes[1].set_yticks([98.5, 99.0, 99.5, 100.0])
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%")
    fig.subplots_adjust(wspace=0.25)
    style.save(fig, os.path.join(OUT, "runs.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    stream_figure()
    runs_figure()
    print("wrote", sorted(os.listdir(OUT)))
