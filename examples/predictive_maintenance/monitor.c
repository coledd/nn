/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Follows one simulated machine from installation to failure, one
// snapshot a day, printing what the device would report: the ISO 10816
// overall vibration level a traditional monitor would show, and the
// model's diagnosis and alert state. Where test.c summarizes hundreds of
// runs, this shows one of them as it unfolds -- including how long the
// overall vibration level looks perfectly normal while the model is
// already flagging a fault.
//
// The main loop below is structured the way the firmware's would be. Only
// machine_observe() is simulation: on the STM32 it becomes "for each of
// VF_FRAMES_PER_SNAPSHOT frames, wait for the accelerometer's DMA buffer
// and pass it to health_monitor_add_frame(); then call
// health_monitor_end_snapshot()".

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include "machine_sim.h"
#include "nn.h"

#define FAILURE_DAY 120
#define ONSET_DAY 25
#define REPORT_EVERY_DAYS 7 // plus every day the model's output or the ISO alarm state changes
#define ISO_ALARM_MM_S 2.8f // same setpoint as test.c's

static health_monitor_t hm; // ~70KB: static rather than on the stack

// ISO 10816-3 zone for group 2 machines on rigid foundations.
static char iso_zone(float velocity_rms)
{
  return velocity_rms < 1.4f ? 'A' : velocity_rms < 2.8f ? 'B' : velocity_rms < 4.5f ? 'C' : 'D';
}

int main(int argc, char *argv[]) {
  machine_condition_t fault = CONDITION_COUNT;
  if (argc == 3 || argc == 4) {
    for (int c = 0; c < CONDITION_COUNT; c++)
      if (strcasecmp(argv[2], condition_names[c]) == 0)
        fault = (machine_condition_t)c;
  }
  if (fault == CONDITION_COUNT) {
    printf("Usage: %s <model-file> <condition> [seed]\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c\n");
    printf("  <condition>  : The fault to develop:");
    for (int c = 0; c < CONDITION_COUNT; c++)
      printf(" %s", condition_names[c]);
    printf("\n  [seed]       : Random seed, to replay the same machine (default: time)\n");
    return 1;
  }
  unsigned seed = argc == 4 ? (unsigned)strtoul(argv[3], NULL, 10) : (unsigned)time(NULL);
  // Inference-only, so nn_load_model() suffices for ascii/binary models;
  // see test.c for loading the inplace format too.
  nn_t *model = nn_load_model(argv[1]);
  if (model == NULL || !health_monitor_init(&hm, (int)model->width[0]) ||
      model->width[model->depth - 1] != (uint32_t)CONDITION_COUNT) {
    fprintf(stderr, "Error: Missing or invalid model file (or not a train.c model): %s\n", argv[1]);
    nn_free(model);
    return 1;
  }
  srand(seed);
  machine_t machine;
  machine_create(&machine);
  printf("Machine (seed %u): %s develops from day %d, fails on day %d.\n\n", seed,
         condition_names[fault], ONSET_DAY, FAILURE_DAY);

  machine_commission(&hm, &machine);
  printf("Commissioned: baseline fingerprint averaged over %d snapshots.\n\n", HM_COMMISSIONING_SNAPSHOTS);
  printf("day  severity  ISO 10816 velocity       model says            model alert\n");
  int last_alert = 0;
  machine_condition_t last_condition = CONDITION_HEALTHY;
  int last_iso_alarm = 0;
  for (int day = 1; day <= FAILURE_DAY; day++) {
    float severity = fault == CONDITION_HEALTHY ? 0.0f : fault_severity_on_day(day, ONSET_DAY, FAILURE_DAY);
    health_report_t report;
    machine_observe(&hm, model, &machine, fault, severity, &report);
    char zone = iso_zone(report.velocity_rms);
    int iso_alarm = report.velocity_rms >= ISO_ALARM_MM_S;
    int changed = report.alert != last_alert || report.condition != last_condition || iso_alarm != last_iso_alarm;
    if (changed || day == 1 || day % REPORT_EVERY_DAYS == 0 || day == FAILURE_DAY) {
      printf("%3d  %8.3f  %5.2f mm/s zone %c %-6s  %-14s %3.0f%%  %s\n", day, severity, report.velocity_rms, zone,
             iso_alarm ? "ALARM" : "", condition_names[report.condition], 100.0f * report.confidence,
             report.alert ? (last_alert ? "ALERT" : "ALERT raised") : "");
    }
    last_alert = report.alert;
    last_condition = report.condition;
    last_iso_alarm = iso_alarm;
  }
  nn_free(model);
  return 0;
}
