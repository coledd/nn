/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HEALTH_MONITOR_H
#define HEALTH_MONITOR_H

// The firmware-side application around the model: everything between "a
// frame of accelerometer samples arrived" and "raise a maintenance alert",
// built on vibration_features.[ch] and nn.[ch] and nothing else, so it
// ships to the target unchanged. train.c, test.c and monitor.c all drive
// this same code on the desktop -- training rows are literally the
// `model_input` a deployed monitor would hand nn_predict().
//
// Lifecycle on the target:
//
//   1. Commissioning. The first HM_COMMISSIONING_SNAPSHOTS snapshots after
//      the sensor is installed on a machine (assumed healthy at that point
//      -- normally right after installation or an overhaul) are averaged into
//      a baseline fingerprint: what *this* machine looks like when it's
//      healthy. Every machine has its own: a slightly different running
//      speed, residual imbalance, structural resonances, gear-mesh level and
//      sensor mounting. The firmware should persist the baseline to flash
//      (health_monitor_set_baseline() restores it after a reset), so
//      commissioning happens once per installation, not once per boot.
//
//   2. Monitoring. Each later snapshot's fingerprint is compared with the
//      baseline, and the model is given both: the present fingerprint, and
//      how far it has moved from the past one. An early fault is often a
//      small *change* -- the 1x peak up 50%, a few new envelope-spectrum
//      lines -- that is well within the range healthy machines span, and
//      only stands out relative to this machine's own history. (Models
//      trained on the present fingerprint alone, with no baseline, are
//      supported too, for comparison -- see train.c's --present-only flag.)
//
//   3. Alerting. Snapshots have to be classified as faulty (as any fault)
//      HM_ALERT_STREAK times in a row before the report's `alert` flag is
//      raised, so one noisy snapshot never pages anybody.

#include <stdbool.h>
#include "nn.h"
#include "vibration_features.h"

// The model's output classes: a healthy machine, or the single most likely
// developing fault. Each produces a distinct change in the fingerprint --
// see vibration_features.h for the two spectra it's built from and
// machine_sim.c for the physics behind each signature.
typedef enum {
  CONDITION_HEALTHY = 0,
  CONDITION_IMBALANCE,          // 1x running speed grows
  CONDITION_MISALIGNMENT,       // 2x (and 3x) running speed grow
  CONDITION_LOOSENESS,          // a long series of harmonics, plus half-orders
  CONDITION_BEARING_OUTER_RACE, // impacts at the outer-race defect frequency (envelope spectrum)
  CONDITION_BEARING_INNER_RACE, // impacts at the inner-race defect frequency, modulated by 1x
  CONDITION_GEAR_TOOTH,         // a damaged tooth: gear-mesh sidebands, impacts once per revolution
  CONDITION_LUBRICATION,        // broadband high-frequency friction, no impact periodicity
  CONDITION_COUNT
} machine_condition_t;

extern const char *condition_names[CONDITION_COUNT];

#define HM_COMMISSIONING_SNAPSHOTS 4
#define HM_ALERT_STREAK 3
// The two model input layouts health_monitor_end_snapshot() can build.
#define HM_INPUTS_WITH_BASELINE (2 * VF_FINGERPRINT_LEN) // present fingerprint, then present minus baseline
#define HM_INPUTS_PRESENT_ONLY VF_FINGERPRINT_LEN        // present fingerprint only

typedef struct {
  vibration_features_t vf;
  int num_inputs;                          // HM_INPUTS_WITH_BASELINE or HM_INPUTS_PRESENT_ONLY
  float baseline[VF_FINGERPRINT_LEN];
  int baseline_snapshots;                  // commissioning snapshots averaged into `baseline` so far
  float fingerprint[VF_FINGERPRINT_LEN];   // most recent snapshot's
  float model_input[HM_INPUTS_WITH_BASELINE];
  int unhealthy_streak;                    // consecutive snapshots classified as any fault
} health_monitor_t;

typedef struct {
  bool commissioning;            // this snapshot went into the baseline; nothing below it is valid
  float velocity_rms;            // ISO 10816 overall RMS velocity, mm/s
  const float *model_input;      // the num_inputs values given to (or, with no model, built for) nn_predict()
  machine_condition_t condition; // most likely condition (only with a model)
  float confidence;              // its softmax probability
  const float *probabilities;    // all CONDITION_COUNT probabilities, owned by the model
  bool alert;                    // the last HM_ALERT_STREAK snapshots, this one included, were all classified as faults
} health_report_t;

// Starts a monitor in the commissioning phase, building `num_inputs`-wide
// model inputs (HM_INPUTS_WITH_BASELINE or HM_INPUTS_PRESENT_ONLY -- pass
// the loaded model's nn->width[0]). Returns false for any other width.
bool health_monitor_init(health_monitor_t *hm, int num_inputs);

// Restores a baseline saved from an earlier commissioning (the contents of
// hm->baseline once commissioning finished), skipping straight to monitoring.
void health_monitor_set_baseline(health_monitor_t *hm, const float *baseline);

// Adds one frame of VF_FRAME_LEN accelerometer samples (in g).
void health_monitor_add_frame(health_monitor_t *hm, const float *frame);

// Ends the current snapshot. During commissioning, folds it into the
// baseline. Afterwards, builds the model input and -- when `nn` is not
// NULL -- classifies it and updates the alert streak. Fills `report`.
void health_monitor_end_snapshot(health_monitor_t *hm, nn_t *nn, health_report_t *report);

#endif /* HEALTH_MONITOR_H */
