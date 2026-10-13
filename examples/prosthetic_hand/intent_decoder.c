/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "intent_decoder.h"

const char *intent_names[INTENT_COUNT] = {
  "REST", "HAND_OPEN", "HAND_CLOSE", "WRIST_PRONATION", "WRIST_SUPINATION", "WRIST_FLEXION", "WRIST_EXTENSION"
};

void intent_decoder_init(intent_decoder_t *id)
{
  memset(id, 0, sizeof(*id));
  emg_features_init(&id->ef);
  for (int i = 0; i < EF_FEATURES; i++)
    id->cal.inv_std[i] = 1.0f;
}

// Mean log MAV across the electrodes: the overall contraction effort.
static float effort(const float *features)
{
  float sum = 0.0f;
  for (int c = 0; c < EF_CHANNELS; c++)
    sum += features[c * EF_FEATURES_PER_CHANNEL];
  return sum / EF_CHANNELS;
}

void intent_decoder_calibrate(intent_decoder_t *id, const float *features, const int *labels, int n)
{
  id_calibration_t *cal = &id->cal;
  // Two passes rather than a running variance: calibration sets are a few
  // thousand vectors at most, and this is exact.
  for (int i = 0; i < EF_FEATURES; i++) {
    double sum = 0.0;
    for (int r = 0; r < n; r++)
      sum += features[(size_t)r * EF_FEATURES + i];
    cal->mean[i] = (float)(sum / n);
    double var = 0.0;
    for (int r = 0; r < n; r++) {
      double d = features[(size_t)r * EF_FEATURES + i] - cal->mean[i];
      var += d * d;
    }
    float std = (float)sqrt(var / n);
    // A feature that never varied (a disconnected electrode) is zeroed
    // rather than blown up.
    cal->inv_std[i] = std > 1e-6f ? 1.0f / std : 0.0f;
  }
  double rest = 0.0, active = 0.0;
  int rest_n = 0, active_n = 0;
  for (int r = 0; r < n; r++) {
    float e = effort(features + (size_t)r * EF_FEATURES);
    if (labels[r] == INTENT_REST) {
      rest += e;
      rest_n++;
    } else {
      active += e;
      active_n++;
    }
  }
  cal->rest_level = rest_n > 0 ? (float)(rest / rest_n) : 0.0f;
  cal->active_level = active_n > 0 ? (float)(active / active_n) : cal->rest_level + 1.0f;
}

intent_t id_vote_add(id_vote_t *vote, intent_t decision)
{
  vote->votes[vote->head] = decision;
  vote->head = (vote->head + 1) % ID_VOTES;
  if (vote->count < ID_VOTES)
    vote->count++;
  int count[INTENT_COUNT] = {0};
  for (int v = 0; v < vote->count; v++)
    count[vote->votes[v]]++;
  // Walking back from the most recent decision, a candidate only replaces
  // the winner with a strictly higher count, so ties go to the more recent
  // intent -- the output follows a genuine change as soon as it's tied.
  int winner = decision;
  for (int k = 1; k <= vote->count; k++) {
    int candidate = vote->votes[(vote->head - k + ID_VOTES) % ID_VOTES];
    if (count[candidate] > count[winner])
      winner = candidate;
  }
  return (intent_t)winner;
}

void intent_decoder_standardize(const intent_decoder_t *id, const float *features, float *out)
{
  for (int i = 0; i < EF_FEATURES; i++)
    out[i] = (features[i] - id->cal.mean[i]) * id->cal.inv_std[i];
}

void intent_decoder_fine_tune(const intent_decoder_t *id, nn_t *nn, const float *features, const int *labels, int n,
                              int epochs, float rate, float *scratch)
{
  float *input = scratch, *target = scratch + EF_FEATURES;
  nn_set_optimizer(nn, NN_OPTIMIZER_ADAM, 0.9f, 0.999f, 1e-8f);
  for (int e = 0; e < epochs; e++) {
    // A fresh random order each pass, without an index array: stepping
    // through 0..n-1 by a random stride coprime with n visits every
    // vector exactly once.
    int start = rand() % n, stride;
    do {
      stride = 1 + rand() % n;
      int a = stride, b = n;
      while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
      }
      if (a == 1)
        break;
    } while (1);
    for (int k = 0; k < n; k++) {
      int r = (int)(((long long)start + (long long)k * stride) % n);
      intent_decoder_standardize(id, features + (size_t)r * EF_FEATURES, input);
      memset(target, 0, INTENT_COUNT * sizeof(float));
      target[labels[r]] = 1.0f;
      nn_train(nn, input, target, rate);
    }
  }
  // Back to plain SGD, which frees Adam's two moment buffers (twice the
  // model's size) for the rest of the session.
  nn_set_optimizer(nn, NN_OPTIMIZER_SGD, 0.0f, 0.0f, 0.0f);
}

bool intent_decoder_add_sample(intent_decoder_t *id, nn_t *nn, const float *sample, id_report_t *report)
{
  if (!emg_features_add_sample(&id->ef, sample, id->features))
    return false;
  intent_decoder_standardize(id, id->features, id->model_input);
  const float *out = nn_predict(nn, id->model_input);
  int best = 0;
  for (int c = 1; c < INTENT_COUNT; c++)
    if (out[c] > out[best])
      best = c;

  float confidence = out[best];
  intent_t decision = confidence >= ID_REJECT_BELOW ? (intent_t)best : INTENT_REST;
  intent_t winner = id_vote_add(&id->vote, decision);

  float span = id->cal.active_level - id->cal.rest_level;
  float speed = span > 0.0f ? (effort(id->features) - id->cal.rest_level) / span : 0.0f;
  report->raw = (intent_t)best;
  report->intent = (intent_t)winner;
  report->confidence = confidence;
  report->speed = speed < 0.0f ? 0.0f : (speed > 1.0f ? 1.0f : speed);
  report->features = id->features;
  report->probabilities = out;
  return true;
}
