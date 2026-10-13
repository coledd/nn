# Neural Network library
# Copyright (c) 2019-2026 Cole Design and Development, LLC
# https://coledd.com
# SPDX-License-Identifier: Apache-2.0
#
# Figures for examples/predictive_maintenance/README.md. The vibration
# figures come from the example's own simulator and feature extractor: a
# small C harness (below) is compiled against machine_sim.c,
# vibration_features.c and health_monitor.c and dumps what they compute.
# The results charts plot the numbers the README reports.
#
#   python3 tools/docs/predictive_maintenance.py

import os
import subprocess
import tempfile

import numpy as np

import style
from banner import banner
from style import INK, INK_2, MUTED, SERIES, plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "predictive_maintenance"))
LIB = os.path.abspath(os.path.join(ROOT, "..", ".."))
OUT = os.path.join(ROOT, "figures")

CONDITIONS = ["HEALTHY", "IMBALANCE", "MISALIGNMENT", "LOOSENESS", "BEARING_OUTER", "BEARING_INNER", "GEAR_TOOTH",
              "LUBRICATION"]
LABELS = ["Healthy", "Imbalance", "Misalignment", "Looseness", "Bearing, outer race", "Bearing, inner race",
          "Gear tooth", "Lubrication"]

HARNESS = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "machine_sim.h"

static float samples[VF_SNAPSHOT_SAMPLES];
static vibration_features_t vf;
static health_monitor_t hm;

/* One snapshot through the feature extractor; writes the frame-averaged
   vibration and envelope power spectra (one line each). */
static void spectra(FILE *f, const machine_t *m, machine_condition_t c, float severity)
{
  float fp[VF_FINGERPRINT_LEN];
  machine_snapshot(m, c, severity, samples);
  vibration_features_init(&vf);
  for (int i = 0; i < VF_FRAMES_PER_SNAPSHOT; i++)
    vibration_features_add_frame(&vf, samples + i * VF_FRAME_LEN);
  for (int k = 0; k < VF_NUM_BINS; k++) fprintf(f, "%g ", vf.spectrum[k] / VF_FRAMES_PER_SNAPSHOT);
  fprintf(f, "\n");
  for (int k = 0; k < VF_NUM_BINS; k++) fprintf(f, "%g ", vf.envelope[k] / VF_FRAMES_PER_SNAPSHOT);
  fprintf(f, "\n");
  vibration_features_finish(&vf, fp);
}

int main(int argc, char **argv)
{
  srand(7);
  machine_t m;
  machine_create(&m);
  char path[512];

  /* Raw acceleration, the first frame of one snapshot per condition. */
  snprintf(path, sizeof(path), "%s/raw.txt", argv[1]);
  FILE *f = fopen(path, "w");
  for (int c = 0; c < CONDITION_COUNT; c++) {
    machine_snapshot(&m, (machine_condition_t)c, 0.5f, samples);
    for (int n = 0; n < VF_FRAME_LEN; n++) fprintf(f, "%g ", samples[n]);
    fprintf(f, "\n");
  }
  fclose(f);

  /* Spectra: healthy, outer-race bearing defect, lubrication, at 0.5. */
  snprintf(path, sizeof(path), "%s/spectra.txt", argv[1]);
  f = fopen(path, "w");
  spectra(f, &m, CONDITION_HEALTHY, 0.0f);
  spectra(f, &m, CONDITION_BEARING_OUTER_RACE, 0.5f);
  spectra(f, &m, CONDITION_LUBRICATION, 0.5f);
  fclose(f);

  /* Fingerprint change from this machine's commissioning baseline, per
     condition at severity 0.5, averaged over 4 snapshots; plus band edges
     and the ISO 10816 velocity. */
  snprintf(path, sizeof(path), "%s/delta.txt", argv[1]);
  f = fopen(path, "w");
  health_monitor_init(&hm, HM_INPUTS_WITH_BASELINE);
  machine_commission(&hm, &m);
  for (int c = 0; c < CONDITION_COUNT; c++) {
    float sum[VF_FINGERPRINT_LEN] = {0}, vel = 0;
    for (int r = 0; r < 4; r++) {
      health_report_t rep;
      machine_observe(&hm, NULL, &m, (machine_condition_t)c, 0.5f, &rep);
      for (int i = 0; i < VF_FINGERPRINT_LEN; i++) sum[i] += rep.model_input[VF_FINGERPRINT_LEN + i] / 4;
      vel += rep.velocity_rms / 4;
    }
    fprintf(f, "%g ", vel);
    for (int i = 0; i < VF_FINGERPRINT_LEN; i++) fprintf(f, "%g ", sum[i]);
    fprintf(f, "\n");
  }
  float bin_hz = VF_SAMPLE_RATE / VF_FRAME_LEN;
  for (int b = 0; b <= VF_SPECTRUM_BANDS; b++) fprintf(f, "%g ", hm.vf.spectrum_edges[b] * bin_hz);
  fprintf(f, "\n");
  for (int b = 0; b <= VF_ENVELOPE_BANDS; b++) fprintf(f, "%g ", hm.vf.envelope_edges[b] * bin_hz);
  fprintf(f, "\n");
  fclose(f);
  return 0;
}
'''


def run_harness(tmp):
    src = os.path.join(tmp, "harness.c")
    exe = os.path.join(tmp, "harness")
    open(src, "w").write(HARNESS)
    subprocess.run(["cc", "-O2", "-I" + ROOT, "-I" + LIB, src,
                    os.path.join(ROOT, "machine_sim.c"), os.path.join(ROOT, "vibration_features.c"),
                    os.path.join(ROOT, "health_monitor.c"), os.path.join(LIB, "fft.c"), os.path.join(LIB, "nn.c"),
                    "-lm", "-o", exe], check=True)
    subprocess.run([exe, tmp], check=True)


def make_banner():
    banner(os.path.join(OUT, "banner.svg"), 7, "Industrial IoT · simulated data",
           "Predictive Maintenance",
           ["Naming the fault developing in a motor and gearbox",
            "from its vibration, months before it fails."],
           [("68–81", "days' median", "warning"),
            ("0", "false alarms in", "5,845 healthy days"),
            ("4.8K", "parameters,", "5.1 KB int8 model")],
           ["Vibration analysis", "FFT + envelope spectrum", "7 fault types", "STM32U585 + IIS3DWB", "FC network"])


def raw_traces(tmp):
    raw = np.loadtxt(os.path.join(tmp, "raw.txt"))
    fs = 26667.0 / 4.0
    n = int(0.25 * fs)
    t = np.arange(n) / fs * 1000
    shown = [0, 1, 4, 7]
    fig, axes = plt.subplots(len(shown), 1, figsize=(11, 5.2), sharex=True)
    scale = max(np.abs(raw[c, :n]).max() for c in shown) * 1.05
    for ax, c in zip(axes, shown):
        ax.plot(t, raw[c, :n], color=INK_2, lw=0.6)
        ax.set_ylim(-scale, scale)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(ax is axes[-1])
        ax.set_ylabel(LABELS[c], rotation=0, ha="right", va="center", color=INK, fontsize=10)
    axes[-1].set_xlabel("milliseconds")
    axes[-1].set_xlim(0, t[-1])
    fig.subplots_adjust(hspace=0.15)
    style.save(fig, os.path.join(OUT, "raw.svg"))
    return scale


def spectra(tmp):
    lines = np.loadtxt(os.path.join(tmp, "spectra.txt"))
    fs = 26667.0 / 4.0
    f = np.arange(lines.shape[1]) * fs / 4096
    names = [("Healthy", MUTED), ("Bearing, outer race (severity 0.5)", SERIES[0]),
             ("Lubrication (severity 0.5)", SERIES[1])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    band = (f >= 8) & (f <= 500)
    for i, (name, color) in enumerate(names):
        spec, env = lines[2 * i], lines[2 * i + 1]
        db = 10 * np.log10(spec / lines[0].max() + 1e-12)
        axes[0].plot(f, db, color=color, lw=1.0 if i == 0 else 1.2, label=name, alpha=0.95)
        # Each envelope spectrum relative to its own median level: what
        # matters is lines standing out of the floor, not the floor itself.
        envdb = 10 * np.log10(env / np.median(env[band]) + 1e-12)
        axes[1].plot(f[band], envdb[band], color=color, lw=1.0 if i == 0 else 1.4, label=name, alpha=0.95)
    axes[0].set_xscale("log")
    axes[0].set_xlim(8, 3300)
    axes[0].set_xticks([10, 30, 100, 300, 1000, 3000], ["10", "30", "100", "300", "1k", "3k"])
    axes[0].set_ylim(-70, 5)
    axes[0].set_ylabel("power, dB relative to the healthy peak")
    axes[0].set_xlabel("frequency, Hz")
    axes[0].set_title("Vibration spectrum", loc="left")
    axes[0].axvspan(1000, 3200, color=style.GRID, alpha=0.5, lw=0)
    axes[0].text(1780, 3, "envelope band", ha="center", va="top", fontsize=8.5, color=MUTED)
    # The outer-race defect frequency, found in the data: the strongest
    # envelope line between 60 and 120 Hz.
    env_b = lines[3]
    win = (f > 60) & (f < 120)
    bpfo = f[win][np.argmax(env_b[win])]
    top = 10 * np.log10(env_b / np.median(env_b[band]))
    for k in (1, 2, 3):
        j = np.argmin(np.abs(f - k * bpfo))
        lab = f"defect frequency {bpfo:.0f} Hz" if k == 1 else f"{k}×"
        axes[1].annotate(lab, (f[j], top[j]), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=INK_2)
    axes[1].set_xlim(0, 500)
    axes[1].set_ylim(-10, 32)
    axes[1].set_ylabel("dB above the spectrum's median")
    axes[1].set_xlabel("frequency, Hz")
    axes[1].set_title("Envelope spectrum (1–3.2 kHz band)", loc="left")
    fig.subplots_adjust(wspace=0.22)
    style.bottom_legend(fig, axes[0], 3)
    style.save(fig, os.path.join(OUT, "spectra.svg"))


def fingerprint_change(tmp):
    rows = open(os.path.join(tmp, "delta.txt")).read().strip().split("\n")
    data = np.array([[float(v) for v in r.split()] for r in rows[:8]])
    vel, delta = data[:, 0], data[:, 1:]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "div", [style.BLUE_RAMP[8], style.NEUTRAL_MID, style.RED_RAMP[3], style.RED_RAMP[7]])
    fig, ax = plt.subplots(figsize=(11, 4.0))
    im = ax.imshow(delta, cmap=cmap, vmin=-1, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(8), [f"{l}  ({v:.2f} mm/s)" for l, v in zip(LABELS, vel)], fontsize=9.5)
    ticks = [0, 12, 24, 36, 47, 48, 55, 63]
    ax.set_xticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(47.5, color=style.SURFACE, lw=4)
    ax.text(23.5, 8.0, "vibration spectrum: 48 bands, 8 Hz – 3.2 kHz (log-spaced)", ha="center", va="top",
            fontsize=9, color=INK_2)
    ax.text(55.5, 8.0, "envelope spectrum:\n16 bands, 8–500 Hz", ha="center", va="top", fontsize=9, color=INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, colors=INK_2)
    cb.set_label("change in log10 energy", fontsize=9, color=INK_2)
    del ticks
    style.save(fig, os.path.join(OUT, "fingerprint.svg"))


def warning_time():
    faults = ["Imbalance", "Misalignment", "Looseness", "Bearing, outer race", "Bearing, inner race", "Gear tooth",
              "Lubrication"]
    model = [70, 78, 81, 78, 68, 73, 68]
    iso = [32, 12, 25, 4, 1, 5, 3]
    iso_warned = [17, 15, 18, 6, 2, 7, 8]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    y = np.arange(len(faults))
    h = 0.32
    ax.barh(y - h / 2 - 0.01, model, height=h, color=SERIES[0], label="Model (warned 20 of 20)")
    ax.barh(y + h / 2 + 0.01, iso, height=h, color=SERIES[1], label="ISO 10816 alarm (when it warned at all)")
    for yi, v in zip(y, model):
        ax.text(v + 1, yi - h / 2 - 0.01, f"{v}", va="center", fontsize=9, color=INK_2)
    for yi, v, w in zip(y, iso, iso_warned):
        ax.text(v + 1, yi + h / 2 + 0.01, f"{v}  (warned {w} of 20)", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, faults)
    ax.invert_yaxis()
    ax.set_xlim(0, 95)
    ax.set_xlabel("median warning before failure, days")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    style.bottom_legend(fig, ax, 2)
    style.save(fig, os.path.join(OUT, "warning.svg"))


def baseline_value():
    labels = ["Accuracy,\nseverity 0.05–0.10", "Accuracy,\nseverity 0.10–0.20", "Accuracy,\noverall",
              "Diagnosis right\nat first alert"]
    with_b = [95.1, 100.0, 99.75, 137 / 140 * 100]
    without = [67.2, 83.2, 96.0, 124 / 140 * 100]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.22
    for k, (vals, name, color) in enumerate([(with_b, "With baseline (128 inputs)", SERIES[0]),
                                             (without, "Present fingerprint only (64 inputs)", SERIES[2])]):
        xs = x + (k - 0.5) * (w + 0.02)
        ax.bar(xs, vals, width=w, color=color, label=name)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.set_title("Snapshot accuracy and diagnosis", loc="left")
    ax2 = axes[1]
    for k, (v, color) in enumerate([(0, SERIES[0]), (9, SERIES[2])]):
        ax2.bar(k, v, width=0.3, color=color)
        ax2.text(k, v + 0.3, f"{v}", ha="center", fontsize=9, color=INK_2)
    ax2.set_xticks([0, 1], ["with\nbaseline", "present\nonly"], fontsize=9)
    ax2.set_ylim(0, 11)
    ax2.set_yticks([0, 3, 6, 9])
    ax2.set_title("Machines falsely alarmed (of 160)", loc="left")
    fig.subplots_adjust(wspace=0.25)
    style.bottom_legend(fig, ax, 2)
    style.save(fig, os.path.join(OUT, "baseline.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    style.setup()
    make_banner()
    with tempfile.TemporaryDirectory() as tmp:
        run_harness(tmp)
        raw_traces(tmp)
        spectra(tmp)
        fingerprint_change(tmp)
    warning_time()
    baseline_value()
    print("wrote", sorted(os.listdir(OUT)))
