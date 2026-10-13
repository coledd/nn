/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently evaluates a model saved by train.c on freshly simulated
// machines it has never seen, two ways:
//
//   1. Snapshot accuracy: a confusion matrix over single snapshots at
//      random severities, then accuracy broken down by severity -- how
//      early in a fault's life the model can name it.
//
//   2. Run to failure: the question predictive maintenance actually asks.
//      Each simulated machine is commissioned, runs healthy for a while,
//      then develops a fault that grows until it fails on day FAILURE_DAY,
//      with one snapshot a day through health_monitor.[ch] -- the same
//      alert logic (HM_ALERT_STREAK days in a row) the firmware uses. The
//      model's warning time is compared with a traditional ISO 10816-3
//      overall-vibration alarm on the same machine and the same days
//      (RMS velocity at or above ISO_ALARM_MM_S, held for the same
//      HM_ALERT_STREAK days). Healthy machines run the same number of days
//      to count false alarms.
//
// Works with either input layout train.c produces (with or without
// --present-only), detected from the model's input width.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "machine_sim.h"
#include "nn.h"

#define TEST_MACHINES 100
#define SNAPSHOTS_PER_MACHINE 16
#define MIN_TEST_SEVERITY 0.05f // same range train.c draws from

#define RUNS_PER_CONDITION 20
#define FAILURE_DAY 120
#define EARLIEST_ONSET_DAY 10
#define LATEST_ONSET_DAY 40
// ISO 10816-3, group 2 machines (15-300kW) on rigid foundations: the zone
// B/C boundary -- "unsatisfactory for long-term continuous operation" --
// is the usual alert setpoint for a threshold-based vibration monitor.
#define ISO_ALARM_MM_S 2.8f

static const float severity_edges[] = {0.05f, 0.1f, 0.2f, 0.4f, 0.7f, 1.0f};
#define NUM_SEVERITY_BUCKETS ((int)(sizeof(severity_edges) / sizeof(severity_edges[0])) - 1)

static health_monitor_t hm; // ~70KB: static rather than on the stack

static nn_t *load_model(const char *model_path, uint8_t **inplace_buf)
{
  // See ../character_recognition/test.c's identical block for why the
  // inplace format needs to be read into a buffer first.
  *inplace_buf = NULL;
  if (nn_model_format(model_path) != NN_MODEL_FORMAT_INPLACE)
    return nn_load_model((char *)model_path);
  FILE *file = fopen(model_path, "rb");
  if (!file)
    return NULL;
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  *inplace_buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
  if (!*inplace_buf || fread(*inplace_buf, 1, (size_t)size, file) != (size_t)size) {
    fclose(file);
    return NULL;
  }
  fclose(file);
  return nn_load_model_inplace(*inplace_buf, (size_t)size);
}

static void snapshot_accuracy(nn_t *model, int num_inputs)
{
  int confusion[CONDITION_COUNT][CONDITION_COUNT] = {{0}}; // [actual][predicted]
  int bucket_correct[NUM_SEVERITY_BUCKETS] = {0}, bucket_total[NUM_SEVERITY_BUCKETS] = {0};
  for (int i = 0; i < TEST_MACHINES; i++) {
    machine_t m;
    machine_create(&m);
    health_monitor_init(&hm, num_inputs);
    machine_commission(&hm, &m);
    for (int j = 0; j < SNAPSHOTS_PER_MACHINE; j++) {
      machine_condition_t actual = (machine_condition_t)(rand() % CONDITION_COUNT);
      float severity = sim_uniform(MIN_TEST_SEVERITY, 1.0f);
      health_report_t report;
      machine_observe(&hm, model, &m, actual, severity, &report);
      confusion[actual][report.condition]++;
      if (actual != CONDITION_HEALTHY) {
        int b = 0;
        while (b < NUM_SEVERITY_BUCKETS - 1 && severity >= severity_edges[b + 1])
          b++;
        bucket_total[b]++;
        if (report.condition == actual)
          bucket_correct[b]++;
      }
    }
  }

  printf("Snapshot accuracy: %d unseen machines, %d snapshots each, fault severity %.2f-1.00\n",
         TEST_MACHINES, SNAPSHOTS_PER_MACHINE, MIN_TEST_SEVERITY);
  printf("Confusion matrix (rows = actual, columns = predicted):\n");
  printf("%-14s", "");
  for (int p = 0; p < CONDITION_COUNT; p++)
    printf("%6.6s ", condition_names[p]);
  printf("\n");
  int total_correct = 0, total = 0;
  for (int a = 0; a < CONDITION_COUNT; a++) {
    int class_total = 0;
    printf("%-14s", condition_names[a]);
    for (int p = 0; p < CONDITION_COUNT; p++) {
      printf("%6d ", confusion[a][p]);
      class_total += confusion[a][p];
    }
    total += class_total;
    total_correct += confusion[a][a];
    printf(" (%5.1f%%)\n", 100.0f * confusion[a][a] / (float)class_total);
  }
  printf("Overall accuracy: %d/%d = %.2f%%\n\n", total_correct, total, 100.0f * total_correct / (float)total);
  printf("Fault accuracy by severity (0 = no fault, 1 = failing):\n");
  for (int b = 0; b < NUM_SEVERITY_BUCKETS; b++)
    printf("  %.2f-%.2f: %4d/%-4d = %5.1f%%\n", severity_edges[b], severity_edges[b + 1],
           bucket_correct[b], bucket_total[b], 100.0f * bucket_correct[b] / (float)bucket_total[b]);
  printf("\n");
}

static int compare_ints(const void *a, const void *b)
{
  return *(const int *)a - *(const int *)b;
}

// Median of the first `n` entries of `values`, which it sorts; -1 if n == 0.
static int median(int *values, int n)
{
  if (n == 0)
    return -1;
  qsort(values, (size_t)n, sizeof(int), compare_ints);
  return values[n / 2];
}

static void run_to_failure(nn_t *model, int num_inputs)
{
  printf("Run to failure: %d unseen machines per condition, failing on day %d (onset day %d-%d), one snapshot a day.\n",
         RUNS_PER_CONDITION, FAILURE_DAY, EARLIEST_ONSET_DAY, LATEST_ONSET_DAY);
  printf("Warning = days before failure that each alarm first fired (held %d days, at least a day before failure); ISO alarm = RMS velocity >= %.1f mm/s.\n",
         HM_ALERT_STREAK, ISO_ALARM_MM_S);
  printf("%-14s %9s %9s %10s %12s %12s %12s\n", "", "NN warned", "diagnosis", "NN median", "NN severity", "ISO warned", "ISO median");
  printf("%-14s %9s %9s %10s %12s %12s %12s\n", "", "in time", "correct", "warning", "at warning", "in time", "warning");
  int false_alarm_runs = 0, false_alarm_days = 0, healthy_days = 0;
  for (int c = 0; c < CONDITION_COUNT; c++) {
    int nn_warning[RUNS_PER_CONDITION], iso_warning[RUNS_PER_CONDITION];
    float severity_sum = 0.0f;
    int nn_warned = 0, iso_warned = 0, correct = 0;
    int misdiagnosed[CONDITION_COUNT] = {0};
    for (int run = 0; run < RUNS_PER_CONDITION; run++) {
      machine_t m;
      machine_create(&m);
      health_monitor_init(&hm, num_inputs);
      machine_commission(&hm, &m);
      int onset = EARLIEST_ONSET_DAY + rand() % (LATEST_ONSET_DAY - EARLIEST_ONSET_DAY + 1);
      int nn_day = -1, iso_day = -1, iso_streak = 0;
      machine_condition_t nn_condition = CONDITION_HEALTHY;
      bool false_alarm = false;
      for (int day = 1; day <= FAILURE_DAY; day++) {
        float severity = c == CONDITION_HEALTHY ? 0.0f : fault_severity_on_day(day, onset, FAILURE_DAY);
        health_report_t report;
        machine_observe(&hm, model, &m, (machine_condition_t)c, severity, &report);
        if (severity == 0.0f)
          healthy_days++;
        iso_streak = report.velocity_rms >= ISO_ALARM_MM_S ? iso_streak + 1 : 0;
        // Either alarm counts as a warning only if it comes at least a day
        // before the failure itself.
        if (iso_day < 0 && iso_streak >= HM_ALERT_STREAK && severity > 0.0f && day < FAILURE_DAY)
          iso_day = day;
        if (!report.alert)
          continue;
        // An alert while there is no fault yet is a false alarm --
        // someone gets sent to inspect a healthy machine. Keep going: a
        // real alert later still counts as the warning.
        if (severity == 0.0f) {
          false_alarm = true;
          false_alarm_days++;
        } else if (nn_day < 0 && day < FAILURE_DAY) {
          nn_day = day;
          nn_condition = report.condition;
          severity_sum += severity;
        }
      }
      if (false_alarm)
        false_alarm_runs++;
      if (c == CONDITION_HEALTHY)
        continue;
      if (nn_day > 0) {
        nn_warning[nn_warned++] = FAILURE_DAY - nn_day;
        if (nn_condition == (machine_condition_t)c)
          correct++;
        else
          misdiagnosed[nn_condition]++;
      }
      if (iso_day > 0)
        iso_warning[iso_warned++] = FAILURE_DAY - iso_day;
    }
    if (c == CONDITION_HEALTHY)
      continue;
    char nn_median[16], iso_median[16], severity_text[16];
    int nn_med = median(nn_warning, nn_warned), iso_med = median(iso_warning, iso_warned);
    snprintf(nn_median, sizeof(nn_median), nn_med < 0 ? "-" : "%d days", nn_med);
    snprintf(iso_median, sizeof(iso_median), iso_med < 0 ? "-" : "%d days", iso_med);
    snprintf(severity_text, sizeof(severity_text), nn_warned ? "%.2f" : "-", nn_warned ? severity_sum / nn_warned : 0.0f);
    printf("%-14s %6d/%-2d %6d/%-2d %10s %12s %9d/%-2d %12s", condition_names[c], nn_warned, RUNS_PER_CONDITION,
           correct, nn_warned, nn_median, severity_text, iso_warned, RUNS_PER_CONDITION, iso_median);
    for (int w = 0; w < CONDITION_COUNT; w++)
      if (misdiagnosed[w])
        printf("  (%d first called %s)", misdiagnosed[w], condition_names[w]);
    printf("\n");
  }
  printf("False alarms (alert raised before any fault existed): %d of %d machines, %d alert-days in %d machine-days of healthy running.\n",
         false_alarm_runs, CONDITION_COUNT * RUNS_PER_CONDITION, false_alarm_days, healthy_days);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c (ascii, binary, or inplace)\n");
    return 1;
  }
  const char *model_path = argv[1];
  uint8_t *inplace_buf;
  nn_t *model = load_model(model_path, &inplace_buf);
  if (model == NULL) {
    fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
    free(inplace_buf);
    return 1;
  }
  int num_inputs = (int)model->width[0];
  if (!health_monitor_init(&hm, num_inputs) || model->width[model->depth - 1] != (uint32_t)CONDITION_COUNT) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match this task (%d or %d inputs, %d outputs) -- is this a train.c model?\n",
            model->width[0], model->width[model->depth - 1], HM_INPUTS_WITH_BASELINE, HM_INPUTS_PRESENT_ONLY, CONDITION_COUNT);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }
  printf("Model inputs: %s\n\n", num_inputs == HM_INPUTS_WITH_BASELINE ? "present fingerprint + change from baseline" : "present fingerprint only (--present-only)");
  srand((unsigned)time(NULL));
  snapshot_accuracy(model, num_inputs);
  run_to_failure(model, num_inputs);
  nn_free(model);
  free(inplace_buf);
  return 0;
}
