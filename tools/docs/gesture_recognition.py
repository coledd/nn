# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/gesture_recognition/README.md. The accelerometer
# windows come from the example's own generator (gesture_data.c, compiled
# here and called directly), and the per-timestep accuracy from a model
# trained by the example's own train program (run make in the example
# first).
#
#   python3 tools/docs/gesture_recognition.py

import ctypes
import os
import subprocess
import tempfile

import numpy as np

import style
from banner import banner
from style import INK_2, MUTED, SERIES, plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
ROOT = os.path.join(REPO, "examples", "gesture_recognition")
OUT = os.path.join(ROOT, "figures")

GESTURES = ["Still", "Shake", "Tilt left", "Tilt right"]
AXES = ["x", "y", "z"]
WINDOW_LEN = 32

# Runs a trained model over freshly generated windows and prints, for every
# gesture and timestep, how many windows the running classification already
# gets right -- the same nn_reset_state()/nn_predict() loop test.c uses, but
# reading the output after every timestep instead of only the last.
PROBE_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "gesture_data.h"
#define WINDOWS 200
int main(int argc, char **argv) {
  nn_t *nn = nn_load_model(argv[1]);
  if (!nn) return 1;
  srand(7);
  for (int g = 0; g < GESTURE_COUNT; g++) {
    int correct[WINDOW_LEN] = {0};
    for (int k = 0; k < WINDOWS; k++) {
      float w[WINDOW_LEN][NUM_AXES];
      generate_gesture((gesture_t)g, w);
      nn_reset_state(nn);
      for (int t = 0; t < WINDOW_LEN; t++) {
        float *o = nn_predict(nn, w[t]);
        int p = 0;
        for (int c = 1; c < GESTURE_COUNT; c++) if (o[c] > o[p]) p = c;
        correct[t] += p == g;
      }
    }
    for (int t = 0; t < WINDOW_LEN; t++) printf("%d%c", correct[t], t == WINDOW_LEN - 1 ? '\n' : ' ');
  }
  return 0;
}
"""

CFLAGS = ["-Ofast", "-march=native", "-flto", "-fPIC"]


def make_banner():
    banner(os.path.join(OUT, "banner.svg"), 2, "Wearables · simulated data",
           "Gesture Recognition",
           ["Recognizing a shake or a tilt from a wearable's",
            "accelerometer, one sample at a time, with a recurrent network."],
           [("100%", "test accuracy,", "200 new windows"),
            ("388", "parameters", "in total"),
            ("1.5 KB", "model,", "32-bit floats")],
           ["3-axis accelerometer", "Synthetic gestures", "RNN, 16 units", "Truncated BPTT", "Microcontroller"])


def generator(tmp):
    lib = os.path.join(tmp, "libgesture.so")
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", lib, os.path.join(ROOT, "gesture_data.c")], check=True)
    return ctypes.CDLL(lib)


def traces(gen):
    """One generated window per gesture, the three axes over 32 timesteps."""
    libc = ctypes.CDLL(None)
    libc.srand(3)
    window = (ctypes.c_float * (WINDOW_LEN * 3))()
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharey=True)
    t = np.arange(WINDOW_LEN)
    for g, ax in enumerate(axes):
        gen.generate_gesture(g, window)
        w = np.frombuffer(window, dtype=np.float32).reshape(WINDOW_LEN, 3)
        for a in range(3):
            ax.plot(t, w[:, a], color=SERIES[a], lw=1.8, label=f"{AXES[a]} axis")
        ax.set_title(GESTURES[g], loc="left")
        ax.set_xticks([0, 15, 31])
        ax.set_xlim(-1, 32)
        ax.set_xlabel("timestep", fontsize=9)
    axes[0].set_ylabel("acceleration (g)")
    axes[0].set_ylim(-1.6, 1.6)
    fig.subplots_adjust(wspace=0.08)
    style.bottom_legend(fig, axes[0], 3)
    style.save(fig, os.path.join(OUT, "gestures.svg"))


def running_accuracy(tmp):
    """Share of windows whose running classification is already right, by timestep."""
    model = os.path.join(tmp, "gesture_model.txt")
    subprocess.run([os.path.join(ROOT, "train"), model], check=True, stdout=subprocess.DEVNULL)
    src = os.path.join(tmp, "probe.c")
    open(src, "w").write(PROBE_C)
    probe = os.path.join(tmp, "probe")
    subprocess.run(["cc", *CFLAGS, f"-I{REPO}", f"-I{ROOT}", src, os.path.join(ROOT, "gesture_data.c"),
                    os.path.join(REPO, "libnn.a"), "-lm", "-lmvec", "-o", probe], check=True)
    rows = subprocess.run([probe, model], check=True, capture_output=True, text=True).stdout.split("\n")
    acc = np.array([[int(v) for v in r.split()] for r in rows if r.strip()]) / 200 * 100
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    t = np.arange(1, WINDOW_LEN + 1)
    for g in range(4):
        ax.plot(t, acc[g], color=SERIES[g], label=GESTURES[g])
    ax.axhline(25, color=style.AXIS, lw=1)
    ax.text(16, 26.5, "chance, 4 classes", fontsize=8.5, color=MUTED, ha="right", va="bottom")
    # Only the start of the window is interesting: every curve stays at
    # 100% from here to timestep 32.
    ax.set_xlim(0.5, 16.5)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_xlabel("timesteps of the window seen so far")
    ax.set_ylabel("windows classified correctly")
    style.bottom_legend(fig, ax, 4)
    style.save(fig, os.path.join(OUT, "running_accuracy.svg"))
    return acc


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    with tempfile.TemporaryDirectory() as tmp:
        traces(generator(tmp))
        acc = running_accuracy(tmp)
    for g in range(4):
        first = int(np.argmax(acc[g] >= 95)) + 1 if (acc[g] >= 95).any() else None
        print(f"{GESTURES[g]}: final {acc[g][-1]:.1f}%, first >=95% at t={first}, t=4 {acc[g][3]:.1f}%, t=8 {acc[g][7]:.1f}%")
    print("wrote", sorted(os.listdir(OUT)))
