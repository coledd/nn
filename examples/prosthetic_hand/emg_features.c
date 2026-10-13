/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <string.h>
#include "emg_features.h"

const char *ef_feature_names[EF_FEATURES_PER_CHANNEL] = {"MAV", "WL", "ZC", "SSC"};

// Keeps log() finite on a dead (all-zero) channel.
#define EF_FLOOR 1e-12f

void emg_features_init(emg_features_t *ef)
{
  memset(ef, 0, sizeof(*ef));
}

// Features of one channel; `x` steps through the window `stride` floats
// at a time.
static void channel_features(const float *x, int stride, float *out)
{
  float mav = 0.0f, wl = 0.0f;
  for (int i = 0; i < EF_WINDOW; i++) {
    mav += fabsf(x[i * stride]);
    if (i > 0)
      wl += fabsf(x[i * stride] - x[(i - 1) * stride]);
  }
  mav /= EF_WINDOW;
  wl /= EF_WINDOW - 1;
  float deadband = EF_DEADBAND * mav;
  int zc = 0, ssc = 0;
  for (int i = 1; i < EF_WINDOW; i++) {
    float prev = x[(i - 1) * stride], cur = x[i * stride];
    if (prev * cur < 0.0f && fabsf(cur - prev) >= deadband)
      zc++;
    if (i < EF_WINDOW - 1) {
      float next = x[(i + 1) * stride];
      if ((cur - prev) * (cur - next) > 0.0f &&
          (fabsf(cur - prev) >= deadband || fabsf(cur - next) >= deadband))
        ssc++;
    }
  }
  out[0] = logf(mav + EF_FLOOR);
  out[1] = logf(wl + EF_FLOOR);
  // As fractions of the window, so they're on a scale comparable to the
  // log amplitudes rather than in the hundreds.
  out[2] = (float)zc / (float)EF_WINDOW;
  out[3] = (float)ssc / (float)EF_WINDOW;
}

void emg_features_compute(const float *window, float *features)
{
  for (int c = 0; c < EF_CHANNELS; c++)
    channel_features(window + c, EF_CHANNELS, features + c * EF_FEATURES_PER_CHANNEL);
}

int emg_features_add_sample(emg_features_t *ef, const float *sample, float *features)
{
  memcpy(ef->ring[ef->head], sample, sizeof(ef->ring[0]));
  ef->head = (ef->head + 1) % EF_WINDOW;
  if (ef->filled < EF_WINDOW)
    ef->filled++;
  ef->since_last++;
  if (ef->filled < EF_WINDOW || ef->since_last < EF_HOP)
    return 0;
  ef->since_last = 0;
  // Unroll the ring into time order. 400 x 8 floats is 12.8KB of stack;
  // a static buffer keeps it off small embedded stacks.
  static float window[EF_WINDOW][EF_CHANNELS];
  int tail = EF_WINDOW - ef->head;
  memcpy(window[0], ef->ring[ef->head], (size_t)tail * sizeof(ef->ring[0]));
  memcpy(window[tail], ef->ring[0], (size_t)ef->head * sizeof(ef->ring[0]));
  emg_features_compute(&window[0][0], features);
  return 1;
}
