# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/character_recognition/README.md, from the MNIST files
# the example downloads and from models trained by its own programs:
#
#   cd examples/character_recognition && make
#   ./train_cnn model_cnn.txt > train_cnn.log && ./train_fc model_fc.txt > train_fc.log
#   ./predict model_cnn.txt > predict_cnn.log
#   ./test model_cnn.txt > test_cnn.log && ./test model_fc.txt > test_fc.log
#   python3 ../../tools/docs/character_recognition.py model_cnn.txt train_cnn.log train_fc.log \
#       predict_cnn.log test_cnn.log test_fc.log
#
# The test-set predictions behind the confusion matrix are computed here by
# a small probe linked against the library, over the example's test.csv.

import os
import re
import subprocess
import sys
import tempfile

import numpy as np

import style
from banner import banner
from style import INK, INK_2, MUTED, SERIES, plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
ROOT = os.path.join(REPO, "examples", "character_recognition")
OUT = os.path.join(ROOT, "figures")

# Prints "actual predicted" for every row of test.csv, exactly as test.c
# scores them (argmax of the softmax output against the one-hot target).
PROBE_C = r"""
#include <stdio.h>
#include "nn.h"
#include "data_prep.h"
int main(int argc, char **argv) {
  nn_t *nn = nn_load_model(argv[1]);
  data_t *d = data_load(argv[2], nn->width[0], nn->width[nn->depth - 1]);
  if (!nn || !d) return 1;
  for (int r = 0; r < d->num_rows; r++) {
    float *o = nn_predict(nn, d->input[r]);
    int p = 0, a = 0;
    for (int c = 1; c < 10; c++) { if (o[c] > o[p]) p = c; if (d->target[r][c] > d->target[r][a]) a = c; }
    printf("%d %d %.4f\n", a, p, o[p]);
  }
  return 0;
}
"""


def load_idx(name):
    d = open(os.path.join(ROOT, "dataset", name), "rb").read()
    if name.endswith("idx3-ubyte"):
        n, rows, cols = (int.from_bytes(d[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(d[16:], np.uint8).reshape(n, rows, cols)
    return np.frombuffer(d[8:], np.uint8)


def parse_log(path):
    rows = [line.split(",") for line in open(path) if re.match(r"^\d+\.\d+, \d+\.\d+, ", line)]
    return np.array([[float(r[0]), float(r[1])] for r in rows])


def make_banner(cnn_test, fc_test):
    banner(os.path.join(OUT, "banner.svg"), 1, "Computer vision · real data",
           "Character Recognition",
           ["Reading handwritten digits on a microcontroller,",
            "and measuring what convolution buys over a plain network."],
           [(f"{cnn_test:.1f}%", "test accuracy,", "CNN"),
            (f"{fc_test:.1f}%", "test accuracy,", "fully connected"),
            ("100K", "parameters,", "CNN")],
           ["MNIST, 70,000 digits", "Standard 10,000-digit test", "28 × 28 grayscale", "CNN vs. FC", "STM32H7"])


def samples():
    """Real MNIST test digits, gray levels scaled to 0-1 exactly as prepare_data.c does."""
    images, labels = load_idx("t10k-images-idx3-ubyte"), load_idx("t10k-labels-idx1-ubyte")
    per = 14
    grid = np.ones((10 * 30 + 2, per * 30 + 2))
    for digit in range(10):
        idx = np.where(labels == digit)[0][:per]
        for k, i in enumerate(idx):
            grid[digit * 30 + 2:digit * 30 + 30, k * 30 + 2:k * 30 + 30] = 1 - images[i] / 255.0
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([digit * 30 + 16 for digit in range(10)], [str(d) for d in range(10)])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    style.save(fig, os.path.join(OUT, "mnist_samples.png"), raster=True)


def training_curves(cnn_log, fc_log):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, log, title in zip(axes, (cnn_log, fc_log), ("CNN (train_cnn)", "Fully connected (train_fc)")):
        e = np.arange(1, len(log) + 1)
        ax.plot(e, log[:, 0], color=SERIES[0], label="training error")
        ax.plot(e, log[:, 1], color=SERIES[1], label="validation error")
        best = int(np.argmin(log[:, 1]))
        ax.plot(e[best], log[best, 1], "o", color=SERIES[1], markeredgecolor=style.SURFACE, markeredgewidth=1.5)
        ax.annotate("saved model", (e[best], log[best, 1]), xytext=(0, -18), textcoords="offset points",
                    ha="center", fontsize=9, color=INK_2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
        ax.set_xticks(e if len(e) <= 16 else [1] + list(range(5, len(e) + 1, 5)))
    axes[0].set_ylabel("cross-entropy")
    axes[0].set_ylim(0, None)
    fig.subplots_adjust(wspace=0.08)
    style.bottom_legend(fig, axes[0], 2)
    style.save(fig, os.path.join(OUT, "training.svg"))


def probe(model):
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "probe.c")
        open(src, "w").write(PROBE_C)
        exe = os.path.join(tmp, "probe")
        subprocess.run(["cc", "-Ofast", "-march=native", "-flto", "-fPIC", f"-I{REPO}", src,
                        os.path.join(REPO, "data_prep.o"), os.path.join(REPO, "libnn.a"), "-lm", "-lmvec",
                        "-o", exe], check=True)
        out = subprocess.run([exe, model, os.path.join(ROOT, "test.csv")], check=True,
                             capture_output=True, text=True).stdout
    rows = np.array([line.split() for line in out.strip().split("\n")], dtype=float)
    return rows[:, 0].astype(int), rows[:, 1].astype(int)


def confusion(actual, predicted):
    m = np.zeros((10, 10), int)
    for a, p in zip(actual, predicted):
        m[a, p] += 1
    # Off-diagonal cells only: the diagonal (correct) would swamp the scale.
    off = m.astype(float).copy()
    np.fill_diagonal(off, np.nan)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", [style.SURFACE] + style.RED_RAMP[1:7])
    cmap.set_bad(style.NEUTRAL_MID)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.imshow(off, cmap=cmap, vmin=0, vmax=max(1, np.nanmax(off)))
    for a in range(10):
        for p in range(10):
            v = m[a, p]
            if a == p:
                ax.text(p, a, f"{v}", ha="center", va="center", fontsize=7.5, color=INK_2)
            elif v:
                ax.text(p, a, f"{v}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 * np.nanmax(off) else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("actual digit")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, 10), minor=True)
    ax.grid(which="minor", color=style.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    style.save(fig, os.path.join(OUT, "confusion.svg"))
    return m


def mistakes(actual, predicted):
    """The first misclassified test.csv rows, drawn from the CSV itself."""
    wrong = np.where(actual != predicted)[0][:24]
    rows = {}
    with open(os.path.join(ROOT, "test.csv")) as f:
        for i, line in enumerate(f):
            if i in set(wrong):
                rows[i] = np.array(line.split(",")[:784], dtype=float).reshape(28, 28)
            if i > wrong[-1]:
                break
    fig, axes = plt.subplots(3, 8, figsize=(8.2, 3.6))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(1 - rows[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{actual[i]} → {predicted[i]}", fontsize=9.5, color=INK, pad=3, loc="center",
                     fontweight="regular", fontfamily="DejaVu Sans")
    fig.subplots_adjust(wspace=0.12, hspace=0.42)
    style.save(fig, os.path.join(OUT, "mistakes.png"), raster=True)


def prediction(predict_log):
    """predict.c's built-in sample and the CNN's output for it."""
    src = open(os.path.join(ROOT, "predict.c")).read()
    body = src[src.index("sample[IMG_SIZE][IMG_SIZE] = {"):]
    body = body[:body.index("};")]
    img = np.array([[int(v) for v in r.split(",") if v.strip()] for r in re.findall(r"\{([0-9,\s]+)\}", body)])
    probs = [float(line.split(":")[1]) for line in open(predict_log) if re.match(r"^\d: ", line)]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1, 2.4]})
    a0.imshow(1 - img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a0.set_xticks([])
    a0.set_yticks([])
    a0.grid(False)
    for sp in a0.spines.values():
        sp.set_visible(False)
    a0.set_title("Input, 28 × 28", loc="left")
    y = np.arange(10)
    best = int(np.argmax(probs))
    a1.barh(y, probs, height=0.55, color=[SERIES[0] if d == best else style.AXIS for d in y])
    a1.set_yticks(y, [str(d) for d in y])
    a1.invert_yaxis()
    a1.set_xlim(0, 1.12)
    a1.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    a1.grid(axis="x")
    a1.grid(axis="y", visible=False)
    a1.text(probs[best] + 0.02, best, f"{probs[best]:.4f}", va="center", fontsize=9.5, color=INK_2)
    a1.set_title("Softmax output", loc="left")
    a1.set_xlabel("probability")
    style.save(fig, os.path.join(OUT, "predict.svg"))
    return probs


if __name__ == "__main__":
    model, cnn_log_path, fc_log_path, predict_log, cnn_test_log, fc_test_log = sys.argv[1:7]
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    cnn_log, fc_log = parse_log(cnn_log_path), parse_log(fc_log_path)
    actual, predicted = probe(model)
    argmax_acc = 100 * np.mean(actual == predicted)

    def test_acc(path):  # test.c's own "Test :" line
        return float(re.search(r"Test : \d+/\d+ = ([\d.]+)%", open(path).read()).group(1))

    make_banner(test_acc(cnn_test_log), test_acc(fc_test_log))
    samples()
    training_curves(cnn_log, fc_log)
    m = confusion(actual, predicted)
    mistakes(actual, predicted)
    probs = prediction(predict_log)
    off = m.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[a, p], a, p) for a in range(10) for p in range(10)), reverse=True)[:5]
    print(f"CNN argmax test accuracy {argmax_acc:.2f}% ({np.sum(actual == predicted)}/{len(actual)}); "
          f"errors {np.sum(actual != predicted)}; top confusions {pairs}")
    print("per-digit accuracy", [f"{d}:{100 * m[d, d] / m[d].sum():.1f}" for d in range(10)])
    print("predict", probs)
    print("wrote", sorted(os.listdir(OUT)))
