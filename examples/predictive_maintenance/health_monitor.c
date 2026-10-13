/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string.h>
#include "health_monitor.h"

const char *condition_names[CONDITION_COUNT] = {
  "HEALTHY", "IMBALANCE", "MISALIGNMENT", "LOOSENESS",
  "BEARING_OUTER", "BEARING_INNER", "GEAR_TOOTH", "LUBRICATION"
};

bool health_monitor_init(health_monitor_t *hm, int num_inputs)
{
  if (num_inputs != HM_INPUTS_WITH_BASELINE && num_inputs != HM_INPUTS_PRESENT_ONLY)
    return false;
  vibration_features_init(&hm->vf);
  hm->num_inputs = num_inputs;
  memset(hm->baseline, 0, sizeof(hm->baseline));
  hm->baseline_snapshots = 0;
  hm->unhealthy_streak = 0;
  return true;
}

void health_monitor_set_baseline(health_monitor_t *hm, const float *baseline)
{
  memcpy(hm->baseline, baseline, sizeof(hm->baseline));
  hm->baseline_snapshots = HM_COMMISSIONING_SNAPSHOTS;
}

void health_monitor_add_frame(health_monitor_t *hm, const float *frame)
{
  vibration_features_add_frame(&hm->vf, frame);
}

void health_monitor_end_snapshot(health_monitor_t *hm, nn_t *nn, health_report_t *report)
{
  memset(report, 0, sizeof(*report));
  report->velocity_rms = vibration_features_finish(&hm->vf, hm->fingerprint);

  if (hm->baseline_snapshots < HM_COMMISSIONING_SNAPSHOTS) {
    // Running mean, so the baseline is valid (if noisier) at every step.
    hm->baseline_snapshots++;
    float weight = 1.0f / (float)hm->baseline_snapshots;
    for (int i = 0; i < VF_FINGERPRINT_LEN; i++)
      hm->baseline[i] += (hm->fingerprint[i] - hm->baseline[i]) * weight;
    report->commissioning = true;
    return;
  }

  for (int i = 0; i < VF_FINGERPRINT_LEN; i++) {
    hm->model_input[i] = hm->fingerprint[i];
    if (hm->num_inputs == HM_INPUTS_WITH_BASELINE)
      hm->model_input[VF_FINGERPRINT_LEN + i] = hm->fingerprint[i] - hm->baseline[i];
  }
  report->model_input = hm->model_input;
  if (nn == NULL)
    return;

  const float *out = nn_predict(nn, hm->model_input);
  int best = 0;
  for (int c = 1; c < CONDITION_COUNT; c++)
    if (out[c] > out[best])
      best = c;
  report->condition = (machine_condition_t)best;
  report->confidence = out[best];
  report->probabilities = out;

  // Counts consecutive *unhealthy* snapshots, whatever fault each names:
  // at a fault's earliest stage the diagnosis can still waver between
  // related conditions (a bearing defect's first impacts look much like
  // friction) before settling, and that shouldn't hold back the alert.
  hm->unhealthy_streak = report->condition == CONDITION_HEALTHY ? 0 : hm->unhealthy_streak + 1;
  report->alert = hm->unhealthy_streak >= HM_ALERT_STREAK;
}
