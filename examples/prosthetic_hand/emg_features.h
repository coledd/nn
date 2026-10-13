/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef EMG_FEATURES_H
#define EMG_FEATURES_H

// On-device surface-EMG feature extraction: turns the raw samples from a
// ring of electrodes around the forearm into a short feature vector every
// EF_HOP samples, describing the last EF_WINDOW samples on every channel.
// No dependencies beyond libm, so it ships to the target unchanged; every
// training row prepare_data.c writes comes out of this same code.
//
// A muscle's surface EMG is the sum of many motor units' action
// potentials: a noise-like signal whose *amplitude* says how hard the
// muscle is working and whose *frequency content* shifts with how many
// units fire and how fast. Each movement of the hand and wrist uses a
// different mix of forearm muscles, so the pattern of activity around the
// ring identifies it. The four features per channel are the classic set
// from Hudgins, Parker & Scott (1993), "A new strategy for multifunction
// myoelectric control", which is still what commercial pattern-recognition
// prosthesis controllers are built on:
//
//   MAV  mean absolute value -- the contraction's intensity
//   WL   waveform length (summed absolute sample-to-sample change) --
//        intensity and frequency together
//   ZC   zero crossings -- a cheap estimate of the signal's dominant frequency
//   SSC  slope sign changes -- the same, weighted toward higher frequencies
//
// MAV and WL are passed through a log, because EMG amplitude spans orders
// of magnitude between rest and a strong contraction. ZC and SSC only count
// crossings and turns larger than EF_DEADBAND times the window's own MAV,
// so the electrode noise at rest doesn't count and neither feature depends
// on the amplifier's gain.

// The ring of 8 around the forearm by default; `make CHANNELS=12` builds
// everything (features.csv included) for all 12 of Ninapro's electrodes
// instead -- see ninapro.h.
#ifndef EF_CHANNELS
#define EF_CHANNELS 8
#endif
#define EF_SAMPLE_RATE 2000
#define EF_WINDOW 400 // 200ms: long enough for stable features, short enough to respond in time
#define EF_HOP 100    // a new decision every 50ms
#define EF_FEATURES_PER_CHANNEL 4
#define EF_FEATURES (EF_CHANNELS * EF_FEATURES_PER_CHANNEL)
#define EF_DEADBAND 0.1f

extern const char *ef_feature_names[EF_FEATURES_PER_CHANNEL];

typedef struct {
  float ring[EF_WINDOW][EF_CHANNELS]; // the last EF_WINDOW samples, oldest at `head`
  int head;
  int filled;            // samples received so far, capped at EF_WINDOW
  int since_last;        // samples since the last feature vector
} emg_features_t;

void emg_features_init(emg_features_t *ef);

// Adds one sample from every channel (EF_CHANNELS values, in volts or any
// consistent unit). Returns 1 -- having written EF_FEATURES values to
// `features`, grouped by channel -- when a full window is available and
// EF_HOP samples have arrived since the last feature vector; 0 otherwise.
int emg_features_add_sample(emg_features_t *ef, const float *sample, float *features);

// The same features, computed directly from a window of EF_WINDOW samples
// x EF_CHANNELS (row-major, oldest first).
void emg_features_compute(const float *window, float *features);

#endif /* EMG_FEATURES_H */
