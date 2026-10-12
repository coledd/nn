/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdlib.h>
#include "gesture_data.h"

const char *gesture_names[GESTURE_COUNT] = {"STILL", "SHAKE", "TILT_LEFT", "TILT_RIGHT"};

// Uniform random float in [-1, 1].
static float randf(void)
{
  return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// Cheap, approximately-Gaussian noise (average of three uniforms) -- good
// enough to make every generated window slightly different, like real
// sensor noise, without needing a full Box-Muller transform.
static float noise(float amplitude)
{
  return amplitude * (randf() + randf() + randf()) / 3.0f;
}

// STILL is just sensor noise around the resting +1g on the z-axis; SHAKE is
// a fast, randomly-phased oscillation on x/y; TILT_LEFT/TILT_RIGHT are a
// slow monotonic ramp in opposite directions on x, as the device is
// rotated -- the same way no two real gestures are identical.
void generate_gesture(gesture_t g, float window[WINDOW_LEN][NUM_AXES])
{
  const float pi = 3.14159265f;
  float phase = randf() * pi;
  float freq = 0.8f + 0.4f * ((float)rand() / (float)RAND_MAX); // jittered cycle count across the window
  for (int t = 0; t < WINDOW_LEN; t++) {
    float frac = (float)t / (float)(WINDOW_LEN - 1);
    float x = 0.0f, y = 0.0f, z = 1.0f; // resting accelerometer reads ~1g on z
    switch (g) {
      case GESTURE_STILL:
        break;
      case GESTURE_SHAKE:
        x = sinf(2.0f * pi * freq * 4.0f * frac + phase);
        y = sinf(2.0f * pi * freq * 4.0f * frac + phase + 1.0f);
        break;
      case GESTURE_TILT_LEFT:
        x = -1.2f * frac;
        break;
      case GESTURE_TILT_RIGHT:
        x = 1.2f * frac;
        break;
      default:
        break;
    }
    window[t][0] = x + noise(0.08f);
    window[t][1] = y + noise(0.08f);
    window[t][2] = z + noise(0.08f);
  }
}

void gesture_one_hot(gesture_t g, float *target)
{
  for (int i = 0; i < GESTURE_COUNT; i++)
    target[i] = (i == (int)g) ? 1.0f : 0.0f;
}
