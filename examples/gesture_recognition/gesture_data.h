/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef GESTURE_DATA_H
#define GESTURE_DATA_H

// Synthetic 3-axis accelerometer "gesture" data, shared by train.c (which
// trains an RNN on it) and test.c (which independently re-evaluates a saved
// model against freshly-generated windows the model never saw during
// training). See train.c's top-of-file comment for why this is synthesized
// on the fly instead of read from a dataset file.

#define NUM_AXES 3    // simulated accelerometer channels: x, y, z
#define WINDOW_LEN 32 // timesteps (samples) per gesture window

// Each gesture has a distinct characteristic waveform on the simulated
// 3-axis accelerometer stream -- see generate_gesture().
typedef enum {
  GESTURE_STILL = 0,
  GESTURE_SHAKE,
  GESTURE_TILT_LEFT,
  GESTURE_TILT_RIGHT,
  GESTURE_COUNT
} gesture_t;

extern const char *gesture_names[GESTURE_COUNT];

// Synthesizes one WINDOW_LEN x NUM_AXES gesture window for class `g`, as if
// freshly read off a 3-axis accelerometer. See gesture_data.c for the
// per-class waveform details. Frequency/phase/noise are re-randomized on
// every call, so no two windows of the same class are identical.
void generate_gesture(gesture_t g, float window[WINDOW_LEN][NUM_AXES]);

// Fills `target` with a one-hot vector (GESTURE_COUNT elements) for class `g`.
void gesture_one_hot(gesture_t g, float *target);

#endif /* GESTURE_DATA_H */
