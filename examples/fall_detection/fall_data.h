/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FALL_DATA_H
#define FALL_DATA_H

#include <stdbool.h>

// Synthetic 3-axis accelerometer data for train_rnn.c/train_gru.c/
// train_lstm.c: a continuous monitoring stream that either stays "normal
// daily activity" throughout,
// or contains exactly one fall event (free-fall dip, impact spike, then
// post-fall stillness) inserted at a random point. Unlike
// examples/gesture_recognition/train.c's fixed one-label-per-window scheme, each
// timestep here has its own ground truth: 0 throughout a normal sequence,
// or 0 up to the fall's onset and 1
// from onset through the end of the sequence (the alert, once raised,
// stays raised for the rest of this monitoring window) -- closer to how a
// real always-on embedded monitor actually labels a live stream.

#define NUM_AXES 3       // simulated accelerometer channels: x, y, z
#define SEQUENCE_LEN 150 // timesteps (samples) per monitoring window

// Fills `window`/`label` with a pure normal-activity sequence (gentle,
// noisy daily-activity motion throughout; label 0.0f at every timestep).
void generate_normal_sequence(float window[SEQUENCE_LEN][NUM_AXES], float label[SEQUENCE_LEN]);

// Fills `window`/`label` with a normal-activity sequence that has exactly
// one fall event spliced in at a random onset: a brief free-fall dip
// (acceleration magnitude collapses toward ~0g), a sharp impact spike, then
// stillness for the remainder of the sequence (low-variance, and at a
// resting orientation that may differ from ordinary standing/sitting --
// the body/device has been reoriented by the fall). `label[t]` is 0.0f
// before the fall's onset and 1.0f from onset through the end of the
// sequence.
void generate_fall_sequence(float window[SEQUENCE_LEN][NUM_AXES], float label[SEQUENCE_LEN]);

#endif /* FALL_DATA_H */
