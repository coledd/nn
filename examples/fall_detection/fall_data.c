/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdlib.h>
#include "fall_data.h"

// Uniform random float in [-1, 1].
static float randf(void)
{
  return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// Cheap, approximately-Gaussian noise (average of three uniforms) -- same
// technique as gesture_data.c's noise(), good enough to make every
// generated sequence slightly different without a full Box-Muller transform.
static float noise(float amplitude)
{
  return amplitude * (randf() + randf() + randf()) / 3.0f;
}

void generate_normal_sequence(float window[SEQUENCE_LEN][NUM_AXES], float label[SEQUENCE_LEN])
{
  const float pi = 3.14159265f;
  // A slow, wandering low-frequency drift on x (shifting posture/gait),
  // plus noise on all three axes and a resting ~1g on z -- ordinary daily
  // activity, not a held-still pose (contrast with the post-fall stillness
  // phase in generate_fall_sequence(), which is much lower-variance).
  float phase = randf() * pi;
  float freq = 0.03f + 0.03f * ((float)rand() / (float)RAND_MAX);
  for (int t = 0; t < SEQUENCE_LEN; t++) {
    float drift = 0.2f * sinf(2.0f * pi * freq * (float)t + phase);
    window[t][0] = drift + noise(0.12f);
    window[t][1] = noise(0.12f);
    window[t][2] = 1.0f + noise(0.12f);
    label[t] = 0.0f;
  }
}

void generate_fall_sequence(float window[SEQUENCE_LEN][NUM_AXES], float label[SEQUENCE_LEN])
{
  // Start from ordinary activity, then splice a fall event in at a random
  // onset, leaving room before it (so the network sees some normal
  // activity first, same as a real monitoring stream would) and after it
  // (so there's always a meaningful stretch of post-fall stillness).
  generate_normal_sequence(window, label);

  const int freefall_len = 5 + rand() % 4;  // 5-8 timesteps of near-weightlessness
  const int impact_len = 1 + rand() % 2;    // 1-2 timesteps of sharp impact
  const int min_before = 30;
  const int min_after = 20;
  const int onset = min_before + rand() % (SEQUENCE_LEN - min_before - freefall_len - impact_len - min_after);

  int t = onset;
  // Free fall: magnitude collapses toward zero on every axis.
  for (int k = 0; k < freefall_len; k++, t++) {
    window[t][0] = noise(0.05f);
    window[t][1] = noise(0.05f);
    window[t][2] = noise(0.05f);
    label[t] = 1.0f;
  }
  // Impact: a sharp spike in a random direction.
  float dir[3] = {randf(), randf(), randf()};
  float dir_mag = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  if (dir_mag < 1e-6f)
    dir_mag = 1.0f;
  for (int k = 0; k < impact_len; k++, t++) {
    float spike = 3.0f + 1.0f * ((float)rand() / (float)RAND_MAX);
    window[t][0] = dir[0] / dir_mag * spike + noise(0.1f);
    window[t][1] = dir[1] / dir_mag * spike + noise(0.1f);
    window[t][2] = dir[2] / dir_mag * spike + noise(0.1f);
    label[t] = 1.0f;
  }
  // Post-fall stillness: the remainder of the sequence, at a random fixed
  // resting orientation (the body/device may now be lying at a different
  // angle than while standing/sitting) with much lower noise than ordinary
  // activity -- a fallen person isn't moving the way an active one is.
  float rest[3] = {randf(), randf(), randf()};
  float rest_mag = sqrtf(rest[0] * rest[0] + rest[1] * rest[1] + rest[2] * rest[2]);
  if (rest_mag < 1e-6f)
    rest_mag = 1.0f;
  for (; t < SEQUENCE_LEN; t++) {
    window[t][0] = rest[0] / rest_mag + noise(0.02f);
    window[t][1] = rest[1] / rest_mag + noise(0.02f);
    window[t][2] = rest[2] / rest_mag + noise(0.02f);
    label[t] = 1.0f;
  }
}
