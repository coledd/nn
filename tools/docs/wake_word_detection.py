# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/wake_word_detection/README.md. The waveforms and
# log-mel features are computed from the downloaded Speech Commands clips
# (run make in the example first) with the same transform as
# ../../audio_features.c -- checked against it to within float rounding;
# the results chart plots the runs the README reports.
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

# Seven train + test runs of the shipped code (test.csv: 217 "marvin", 224 other).
RUNS = {"accuracy": [84.13, 85.26, 88.89, 84.13, 89.12, 78.23, 86.62],
        "false reject": [17.5, 26.3, 10.6, 9.7, 6.0, 20.7, 15.7],
        "false accept": [14.3, 3.6, 11.6, 21.9, 15.6, 22.8, 11.2]}


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
            "with log-mel features and a 32-unit GRU."],
           [("85%", "median test", "accuracy, 7 runs"),
            ("49 × 20", "log-mel frames", "per 1 s clip"),
            ("5,121", "parameters,", "one GRU layer")],
           ["Google Speech Commands", "Real recordings", "16 kHz audio", "Log-mel features", "GRU"])


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
    for i, (name, ys) in enumerate(RUNS.items()):
        ys = np.array(ys)
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(ys)), ys, s=44, color=SERIES[i], edgecolor=style.SURFACE,
                   linewidth=1.5, zorder=3)
        med = np.median(ys)
        ax.plot([i - 0.2, i + 0.2], [med, med], color=style.INK, lw=2, zorder=4)
        ax.text(i + 0.26, med, f"median {med:.1f}%", va="center", fontsize=9, color=INK_2)
    ax.set_xticks(range(3), ["accuracy", "false-reject rate\n(missed “marvin”)", "false-accept rate\n(other audio taken for it)"])
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.text(-0.45, 2, "441 held-out clips per run", fontsize=8.5, color=MUTED)
    style.save(fig, os.path.join(OUT, "runs.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    audio_figure()
    runs_figure()
    print("wrote", sorted(os.listdir(OUT)))
