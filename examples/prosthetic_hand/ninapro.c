/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include "mat_reader.h"
#include "ninapro.h"

_Static_assert(EF_CHANNELS >= 1 && EF_CHANNELS <= 12, "Ninapro DB2/DB3 recordings have 12 electrodes");

// Exercise-1 movement numbers, from Atzori et al. (2012), "Building the
// NINAPRO database", table I: the 8 hand postures come first, then the 9
// wrist movements. 5 is "abduction of the fingers" (an open, spread hand),
// 6 "fingers flexed together" (a fist), 9 and 10 wrist supination and
// pronation about the middle finger's axis, 13 and 14 wrist flexion and
// extension.
const int ninapro_movement[INTENT_COUNT] = {0, 5, 6, 10, 9, 13, 14};

void ninapro_path(char *buf, size_t len, const char *dataset_dir, int db, int subject)
{
  snprintf(buf, len, "%s/db%d_s%d_e1.mat", dataset_dir, db, subject);
}

int ninapro_load(const char *path, ninapro_recording_t *rec)
{
  // The refined labels: Ninapro re-derived each movement's start and end
  // from the recorded signals after the fact, since people never start or
  // stop exactly when the on-screen cue does.
  const char *names[] = {"emg", "restimulus", "rerepetition"};
  mat_var_t vars[3];
  if (mat_read(path, names, 3, vars) != 0)
    return -1;
  int n = vars[0].rows;
  if (vars[0].cols < EF_CHANNELS || vars[1].rows != n || vars[2].rows != n) {
    fprintf(stderr, "Error: %s: unexpected variable shapes\n", path);
    for (int v = 0; v < 3; v++)
      mat_var_free(&vars[v]);
    return -1;
  }
  rec->samples = n;
  rec->emg = (float *)malloc((size_t)n * EF_CHANNELS * sizeof(float));
  rec->intent = (signed char *)malloc((size_t)n);
  rec->repetition = (signed char *)malloc((size_t)n);
  rec->next_movement = (signed char *)malloc((size_t)n);
  rec->run = (int *)malloc((size_t)n * sizeof(int));
  if (rec->emg == NULL || rec->intent == NULL || rec->repetition == NULL || rec->next_movement == NULL || rec->run == NULL) {
    fprintf(stderr, "Error: Out of memory loading %s\n", path);
    for (int v = 0; v < 3; v++)
      mat_var_free(&vars[v]);
    ninapro_free(rec);
    return -1;
  }
  int stride = vars[0].cols;
  for (int i = 0; i < n; i++) {
    for (int c = 0; c < EF_CHANNELS; c++)
      rec->emg[(size_t)i * EF_CHANNELS + c] = vars[0].data[(size_t)i * stride + c];
    int movement = (int)vars[1].data[i];
    rec->intent[i] = -1;
    for (int k = 0; k < INTENT_COUNT; k++)
      if (ninapro_movement[k] == movement)
        rec->intent[i] = (signed char)k;
    rec->repetition[i] = (signed char)vars[2].data[i];
  }
  for (int v = 0; v < 3; v++)
    mat_var_free(&vars[v]);
  signed char upcoming = -1;
  for (int i = n - 1; i >= 0; i--) {
    if (rec->intent[i] != INTENT_REST)
      upcoming = rec->intent[i]; // one of ours, or -1
    rec->next_movement[i] = upcoming;
  }
  for (int i = 0; i < n; i++)
    rec->run[i] = (i > 0 && rec->intent[i] == rec->intent[i - 1] && rec->repetition[i] == rec->repetition[i - 1])
                  ? rec->run[i - 1] + 1 : 1;
  return 0;
}

int ninapro_window_intent(const ninapro_recording_t *rec, int end)
{
  int intent = rec->intent[end];
  if (intent < 0 || rec->run[end] < EF_WINDOW)
    return -1;
  // Rest before a movement that isn't one of ours ends at a sample
  // labeled -1, not at one of our intents.
  if (intent == INTENT_REST && rec->next_movement[end] <= INTENT_REST)
    return -1;
  return intent;
}

void ninapro_free(ninapro_recording_t *rec)
{
  free(rec->emg);
  free(rec->intent);
  free(rec->repetition);
  free(rec->next_movement);
  free(rec->run);
  rec->emg = NULL;
  rec->intent = NULL;
  rec->repetition = NULL;
  rec->next_movement = NULL;
  rec->run = NULL;
}
