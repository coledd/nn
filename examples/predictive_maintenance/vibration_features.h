/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef VIBRATION_FEATURES_H
#define VIBRATION_FEATURES_H

// Turns raw accelerometer samples from a running machine into a compact
// spectral "fingerprint" -- the vibration equivalent of what
// ../../audio_features.[ch] does for speech. Like that file, this one is
// meant to ship into firmware: the exact same calls that turn simulated
// vibration into training rows on the desktop run, unchanged, against the
// live accelerometer on the target, so the model never sees features at
// inference that differ from what it was trained on.
//
// Every snapshot is VF_FRAMES_PER_SNAPSHOT back-to-back frames of
// VF_FRAME_LEN samples, fed in one at a time with vibration_features_add_frame()
// (so the caller only ever needs one frame of sample memory -- the
// accelerometer's DMA buffer), then summarized with
// vibration_features_finish(). Each frame contributes to two averaged
// (Welch) power spectra, which together make up the fingerprint:
//
//   1. The ordinary vibration spectrum (FFT of the Hann-windowed frame).
//      Shaft-rate faults -- imbalance, misalignment, looseness -- show up
//      here as growth at 1x, 2x, 3x... the running speed, or at half-orders.
//      Gear damage raises energy around the gear-mesh frequency, and poor
//      lubrication raises the broadband high-frequency floor.
//
//   2. The envelope spectrum. A rolling-element bearing defect makes a tiny
//      impact every time a ball passes over it, ringing the machine's
//      structural resonances at a few kHz. In the ordinary spectrum that's
//      just a rise in the high-frequency floor (indistinguishable from
//      friction from poor lubrication). Band-passing the resonance region
//      (VF_ENVELOPE_LOW_HZ..VF_ENVELOPE_HIGH_HZ), taking its amplitude
//      envelope (the magnitude of the analytic signal, via an inverse FFT),
//      and taking the FFT of *that* recovers the impact repetition rate
//      instead: a peak at the outer-race (~3.6x shaft speed) or inner-race
//      (~5.4x) defect frequency, or at 1x for a damaged gear tooth that hits
//      once per revolution. Friction stays flat. This is the classic
//      "envelope analysis" technique of vibration-based condition monitoring.
//
// Each spectrum is reduced to log-energy in logarithmically spaced bands
// (fine resolution around the shaft harmonics, coarser up near the
// resonances), each set of bands normalized to zero mean so a sensor's
// overall mounting gain drops out.
//
// No heap allocation: a vibration_features_t is a plain struct (~66KB,
// dominated by the FFT buffers and the two accumulated spectra) that the
// caller places wherever suits the target -- a static on a microcontroller.

#include <stdint.h>

// The IIS3DWB wideband vibration accelerometer (the sensor on ST's
// STWIN.box condition-monitoring kit) samples at a fixed 26.667kHz; the
// firmware decimates by 4, which keeps everything up to ~3.3kHz -- enough to
// cover the shaft harmonics, the gear-mesh frequency, and the structural
// resonances bearing impacts excite.
#define VF_SAMPLE_RATE (26667.0f / 4.0f)
#define VF_FRAME_LEN 4096                       // FFT size: ~0.61s per frame, 1.63Hz per bin
#define VF_NUM_BINS (VF_FRAME_LEN / 2 + 1)      // non-redundant half of a real-input FFT
#define VF_FRAMES_PER_SNAPSHOT 8                // ~4.9s of vibration per snapshot
#define VF_SNAPSHOT_SAMPLES (VF_FRAME_LEN * VF_FRAMES_PER_SNAPSHOT)

#define VF_SPECTRUM_BANDS 48
#define VF_SPECTRUM_LOW_HZ 8.0f       // below half the running speed -- low enough to see half-order looseness
#define VF_SPECTRUM_HIGH_HZ 3200.0f
#define VF_ENVELOPE_BANDS 16
#define VF_ENVELOPE_LOW_HZ 1000.0f    // the resonance band whose envelope is analyzed
#define VF_ENVELOPE_HIGH_HZ 3200.0f
#define VF_ENVELOPE_SPECTRUM_LOW_HZ 8.0f  // envelope-spectrum range: impact repetition rates
#define VF_ENVELOPE_SPECTRUM_HIGH_HZ 500.0f
#define VF_FINGERPRINT_LEN (VF_SPECTRUM_BANDS + VF_ENVELOPE_BANDS)

// ISO 10816-3's overall vibration severity is the RMS vibration *velocity*
// between 10Hz and 1kHz -- the single number a traditional threshold-alarm
// vibration monitor watches. vibration_features_finish() reports it
// alongside the fingerprint so the two can be compared.
#define VF_VELOCITY_LOW_HZ 10.0f
#define VF_VELOCITY_HIGH_HZ 1000.0f

typedef struct {
  float window[VF_FRAME_LEN];        // Hann window, precomputed once
  float real[VF_FRAME_LEN];          // FFT scratch
  float imag[VF_FRAME_LEN];          // FFT scratch
  float spectrum[VF_NUM_BINS];       // accumulated vibration power spectrum
  float envelope[VF_NUM_BINS];       // accumulated envelope power spectrum
  float window_sum;                  // sum of window[]
  float window_power;                // sum of window[]^2
  uint16_t spectrum_edges[VF_SPECTRUM_BANDS + 1]; // band b spans bins [edges[b], edges[b+1])
  uint16_t envelope_edges[VF_ENVELOPE_BANDS + 1];
  uint16_t envelope_low_bin;
  uint16_t envelope_high_bin;
  int frames;                        // frames accumulated into the current snapshot
} vibration_features_t;

// Precomputes the window and band edges and starts an empty snapshot.
void vibration_features_init(vibration_features_t *vf);

// Adds one frame of VF_FRAME_LEN raw acceleration samples (in g) to the
// snapshot in progress. Allocation-free; three VF_FRAME_LEN-point FFTs.
void vibration_features_add_frame(vibration_features_t *vf, const float *frame);

// Summarizes every frame added since the last call (normally
// VF_FRAMES_PER_SNAPSHOT of them) into VF_FINGERPRINT_LEN values in
// `fingerprint` -- the VF_SPECTRUM_BANDS vibration-spectrum bands first,
// then the VF_ENVELOPE_BANDS envelope-spectrum bands -- and returns the ISO
// 10816 RMS velocity over the same data, in mm/s. Resets for the next
// snapshot. Returns -1 without touching `fingerprint` if no frames were added.
float vibration_features_finish(vibration_features_t *vf, float *fingerprint);

#endif /* VIBRATION_FEATURES_H */
