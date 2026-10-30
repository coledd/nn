# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/wake_word_detection/README.md. The waveforms and
# log-mel features are computed from the downloaded Speech Commands clips
# (run make in the example first) with the same transform as
# ../../audio_features.c -- checked against it to within float rounding;
# the results charts plot the runs the README reports.
#
#   python3 tools/docs/wake_word_detection.py

import os
import wave

import numpy as np

import style
from banner import banner
from style import INK_2, MUTED, SERIES, plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "wake_word_detection")
OUT = os.path.join(ROOT, "figures")

# wake_word_data.h
SAMPLE_RATE, FRAME_LEN, HOP_LEN, NUM_MEL_BINS = 16000, 512, 320, 20

# Five train_cnn + five train_gru runs of the shipped code, each scored by
# test on test.csv (195 "marvin", 10,853 other clips, speakers unseen in
# training) at the default 0.5 threshold: false-reject %, false-accept %.
RUNS = {"CNN": [(2.6, 2.50), (5.6, 0.96), (2.6, 3.12), (3.6, 1.08), (5.1, 1.12)],
        "GRU": [(4.1, 2.34), (4.1, 1.89), (4.6, 1.52), (5.1, 2.36), (8.2, 1.07)]}

# test's threshold table on test.csv for one representative run of each
# (threshold, false-reject %, false-accept %): CNN run 4, GRU run 3.
TRADEOFF = {
    "CNN": [(0.05, 0.5, 5.84), (0.10, 1.0, 4.13), (0.15, 1.0, 3.34), (0.20, 2.6, 2.79), (0.25, 2.6, 2.33),
            (0.30, 2.6, 1.95), (0.35, 3.6, 1.72), (0.40, 3.6, 1.47), (0.45, 3.6, 1.29), (0.50, 3.6, 1.08),
            (0.55, 4.1, 0.96), (0.60, 4.1, 0.81), (0.65, 4.1, 0.66), (0.70, 4.6, 0.63), (0.75, 4.6, 0.53),
            (0.80, 4.6, 0.45), (0.85, 5.6, 0.33), (0.90, 6.7, 0.21), (0.95, 8.7, 0.15)],
    "GRU": [(0.05, 1.0, 7.56), (0.10, 1.0, 5.12), (0.15, 1.5, 4.16), (0.20, 1.5, 3.47), (0.25, 2.1, 2.89),
            (0.30, 2.1, 2.56), (0.35, 3.1, 2.18), (0.40, 4.6, 1.94), (0.45, 4.6, 1.70), (0.50, 4.6, 1.52),
            (0.55, 5.1, 1.36), (0.60, 5.1, 1.17), (0.65, 5.1, 1.00), (0.70, 5.1, 0.81), (0.75, 5.6, 0.70),
            (0.80, 6.2, 0.58), (0.85, 7.7, 0.41), (0.90, 8.7, 0.27), (0.95, 11.3, 0.16)],
}
# Each CNN run's threshold chosen on validation.csv (the lowest giving at
# most 1% false accepts there), and what it scored on test.csv.
TUNED = [(0.75, 5.1, 0.92), (0.50, 5.6, 0.96), (0.90, 6.2, 0.57), (0.60, 4.1, 0.81), (0.55, 5.1, 0.84)]

# stream's table on the test stream at 20 dB SNR (threshold, missed %,
# false wake-ups per hour): CNN run 4 trained on whole clips only (before
# prepare_data.c added streaming windows), and CNN run 4 and GRU run 3
# trained with them.
STREAM = {
    "CNN, trained on whole clips only": [
        (0.05, 0.5, 2177.49), (0.10, 0.5, 1839.75), (0.20, 0.5, 1429.40), (0.30, 0.5, 1163.73), (0.40, 0.5, 950.52),
        (0.50, 0.5, 786.60), (0.60, 0.5, 643.69), (0.70, 0.5, 512.97), (0.80, 1.5, 387.91), (0.85, 2.6, 315.49),
        (0.90, 4.6, 235.82), (0.95, 7.2, 130.72), (0.97, 10.3, 85.14), (0.98, 11.8, 64.48), (0.99, 14.4, 39.21),
        (0.995, 16.4, 25.08)],
    "CNN, with streaming windows": [
        (0.05, 0.5, 891.17), (0.10, 0.5, 702.34), (0.20, 1.0, 534.70), (0.30, 1.0, 442.85), (0.40, 1.0, 362.65),
        (0.50, 2.6, 274.15), (0.60, 3.1, 180.35), (0.70, 3.1, 126.30), (0.80, 3.6, 70.13), (0.85, 4.1, 56.70),
        (0.90, 4.1, 34.80), (0.95, 5.6, 21.90), (0.97, 6.7, 14.84), (0.98, 6.7, 11.31), (0.99, 9.2, 6.36),
        (0.995, 10.8, 3.71)],
    "GRU, with streaming windows": [
        (0.05, 1.0, 642.81), (0.10, 2.6, 362.65), (0.20, 3.6, 214.27), (0.30, 4.6, 158.27), (0.40, 4.6, 124.71),
        (0.50, 5.6, 100.69), (0.60, 5.6, 83.91), (0.70, 6.2, 68.01), (0.80, 7.2, 51.93), (0.85, 8.2, 45.57),
        (0.90, 10.3, 35.86), (0.95, 14.9, 11.48), (0.97, 17.4, 7.07), (0.98, 20.5, 4.24), (0.99, 26.2, 1.06),
        (0.995, 35.9, 0.53)],
}
# The streaming-trained CNN runs whose validation stream reached at most 10
# false wake-ups per hour (four of five), at that threshold on the test
# stream: (false wake-ups per hour, missed %).
STREAM_TUNED = [(7.60, 10.8), (6.71, 13.8), (6.36, 9.2), (8.30, 9.7)]

def load_clip(path):
    """16-bit PCM -> float in [-1, 1), padded or truncated to 1 s (prepare_data.c)."""
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    clip = np.zeros(SAMPLE_RATE, np.float32)
    clip[:min(len(x), SAMPLE_RATE)] = x[:SAMPLE_RATE]
    return clip


def log_mel(clip):
    """audio_features_extract_sequence(): Hann window, FFT power, triangular
    mel filterbank, log, then normalized to zero mean, unit variance per clip."""
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    points = np.linspace(mel(0), mel(SAMPLE_RATE / 2), NUM_MEL_BINS + 2)
    bins = np.floor((FRAME_LEN + 1) * imel(points) / SAMPLE_RATE).astype(int)
    weights = np.zeros((NUM_MEL_BINS, FRAME_LEN // 2 + 1))
    for m in range(1, NUM_MEL_BINS + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(FRAME_LEN // 2 + 1):
            if left <= k <= center and center > left:
                weights[m - 1, k] = (k - left) / (center - left)
            elif center <= k <= right and right > center:
                weights[m - 1, k] = (right - k) / (right - center)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))
    frames = 1 + (SAMPLE_RATE - FRAME_LEN) // HOP_LEN
    f = np.array([np.log(weights @ np.abs(np.fft.rfft(clip[t * HOP_LEN:t * HOP_LEN + FRAME_LEN] * window)) ** 2 + 1e-6)
                  for t in range(frames)])
    return (f - f.mean()) / np.sqrt(f.var() + 1e-6)


def make_banner():
    banner(os.path.join(OUT, "banner.svg"), 4, "Audio · real data",
           "Wake Word Detection",
           ["Listening for one spoken word in a second of audio,",
            "with log-mel features and a small CNN."],
           [("~8%", "of wake words missed", "at 1% false triggers"),
            ("49 × 20", "log-mel frames", "per 1 s clip"),
            ("21 KB", "of flash,", "quantized to int8")],
           ["Google Speech Commands", "Unseen speakers", "16 kHz audio", "Log-mel features", "CNN"])


def audio_figure():
    clips = [("“marvin”: the wake word", "marvin/00b01445_nohash_0.wav"),
             ("“house”: another word", "house/00b01445_nohash_0.wav"),
             ("Background noise", None)]
    noise = os.path.join(ROOT, "dataset", "_background_noise_", "running_tap.wav")
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.0), gridspec_kw={"height_ratios": [1, 1.6]})
    for j, (title, rel) in enumerate(clips):
        if rel is None:
            w = wave.open(noise)
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            clip = x[5 * SAMPLE_RATE:6 * SAMPLE_RATE]
        else:
            clip = load_clip(os.path.join(ROOT, "dataset", rel))
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        ax = axes[0, j]
        ax.plot(t, clip, color=INK_2, lw=0.4)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left")
        ax = axes[1, j]
        f = log_mel(clip)
        ax.imshow(f.T, origin="lower", aspect="auto", cmap="Blues", vmin=-2.5, vmax=2.5,
                  extent=(0, f.shape[0] * HOP_LEN / SAMPLE_RATE + (FRAME_LEN - HOP_LEN) / SAMPLE_RATE, 0, NUM_MEL_BINS))
        ax.grid(False)
        ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1 s"])
        ax.set_yticks([0.5, 19.5], ["0 Hz", "8 kHz"] if j == 0 else ["", ""])
        if j == 0:
            ax.set_ylabel("20 mel bands", fontsize=9)
    axes[0, 0].set_ylabel("waveform", fontsize=9)
    fig.subplots_adjust(hspace=0.12, wspace=0.08)
    style.save(fig, os.path.join(OUT, "audio.svg"))


def runs_figure():
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    rng = np.random.default_rng(3)
    for k, (name, runs) in enumerate(RUNS.items()):
        for j in range(2):
            ys = np.array([r[j] for r in runs])
            x = j * 2.4 + k
            ax.scatter(x + rng.uniform(-0.1, 0.1, len(ys)), ys, s=44, color=SERIES[k], edgecolor=style.SURFACE,
                       linewidth=1.5, zorder=3, label=name if j == 0 else None)
            med = np.median(ys)
            ax.plot([x - 0.2, x + 0.2], [med, med], color=style.INK, lw=2, zorder=4)
            ax.text(x + 0.26, med, f"{med:.1f}%", va="center", fontsize=9, color=INK_2)
    ax.set_xticks([0.5, 2.9], ["false-reject rate\n(missed “marvin”)", "false-accept rate\n(other audio taken for it)"])
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.legend(loc="upper right", frameon=False)
    ax.text(-0.45, 9.6, "5 runs each · threshold 0.5 · 11,048 test clips", fontsize=8.5, color=MUTED, va="top")
    style.save(fig, os.path.join(OUT, "runs.svg"))


def tradeoff_figure():
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for k, (name, rows) in enumerate(TRADEOFF.items()):
        fa = [r[2] for r in rows]
        fr = [r[1] for r in rows]
        ax.plot(fa, fr, color=SERIES[k], lw=2, marker="o", ms=3.5, label=f"{name}, one run", zorder=3)
        half = next(r for r in rows if r[0] == 0.5)
        ax.scatter([half[2]], [half[1]], s=70, facecolor=style.SURFACE, edgecolor=SERIES[k], linewidth=2, zorder=4)
        ax.annotate("0.5", (half[2], half[1]), xytext=(6, -12), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.scatter([t[2] for t in TUNED], [t[1] for t in TUNED], s=60, marker="D", color=style.INK, zorder=5,
               label="CNN, all 5 runs, threshold\nset on validation for at most 1% false accepts")
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10], ["0.2%", "0.5%", "1%", "2%", "5%", "10%"])
    ax.minorticks_off()
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 14)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("false-accept rate (other audio taken for “marvin”)", fontsize=9)
    ax.set_ylabel("false-reject rate\n(missed “marvin”)", fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    style.save(fig, os.path.join(OUT, "tradeoff.svg"))


def stream_figure():
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    colors = [MUTED, SERIES[0], SERIES[1]]
    for k, (name, rows) in enumerate(STREAM.items()):
        ax.plot([r[2] for r in rows], [r[1] for r in rows], color=colors[k], lw=2, marker="o", ms=3.5,
                ls="--" if k == 0 else "-", label=name, zorder=3)
    ax.scatter([t[0] for t in STREAM_TUNED], [t[1] for t in STREAM_TUNED], s=60, marker="D", color=style.INK, zorder=5,
               label="CNN with streaming windows, 4 runs, threshold\nset on validation for at most 10 false wake-ups/h")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1,000"])
    ax.minorticks_off()
    ax.set_xlim(0.4, 2500)
    ax.set_ylim(0, 40)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("false wake-ups per hour", fontsize=9)
    ax.set_ylabel("missed “marvin”", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.text(0.45, 38.5, "5.7-hour test stream · 20 dB SNR · window every 100 ms", fontsize=8.5, color=MUTED, va="top")
    style.save(fig, os.path.join(OUT, "stream.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    audio_figure()
    runs_figure()
    tradeoff_figure()
    stream_figure()
    print("wrote", sorted(os.listdir(OUT)))
