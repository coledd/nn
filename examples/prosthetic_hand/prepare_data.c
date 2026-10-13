/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Turns the downloaded Ninapro recordings into feature rows, one per
// decision window, by streaming every recording sample by sample through
// emg_features.[ch] -- the same code, at the same window and hop, that runs
// on the prosthesis.
//
// Unlike most of the other examples' prepare steps, this doesn't split the
// rows into train/validation/test CSVs: which rows train what depends on
// the experiment (a model shared across many people, or one wearer's own
// calibration -- see train.c and test.c), so every row carries the
// metadata those splits need:
//
//   db, subject, repetition, intent, then EF_FEATURES raw features
//
// A window is kept only if all of its samples carry the same intent and
// repetition, so no row straddles a movement's start or end; replay.c is
// where the transitions are measured. Rest is kept only from the rest
// periods just before the six movements used here, and only one window in
// NINAPRO_REST_THINNING of it, so it's about as common as each movement.

#include <stdio.h>
#include <stdlib.h>
#include "emg_features.h"
#include "ninapro.h"


static int write_subject(FILE *out, const char *dataset_dir, int db, int subject, int *counts)
{
  char path[512];
  ninapro_path(path, sizeof(path), dataset_dir, db, subject);
  ninapro_recording_t rec;
  if (ninapro_load(path, &rec) != 0)
    return -1;
  emg_features_t ef;
  emg_features_init(&ef);
  float features[EF_FEATURES];
  int rest_windows = 0;
  for (int i = 0; i < rec.samples; i++) {
    if (!emg_features_add_sample(&ef, rec.emg + (size_t)i * EF_CHANNELS, features))
      continue;
    int intent = ninapro_window_intent(&rec, i);
    if (intent < 0 || (intent == INTENT_REST && (rest_windows++ % NINAPRO_REST_THINNING) != 0))
      continue;
    fprintf(out, "%d,%d,%d,%d", db, subject, rec.repetition[i], intent);
    for (int f = 0; f < EF_FEATURES; f++)
      fprintf(out, ",%.5g", features[f]);
    fprintf(out, "\n");
    counts[intent]++;
  }
  ninapro_free(&rec);
  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <dataset-dir> <output.csv>\n", argv[0]);
    return 1;
  }
  FILE *out = fopen(argv[2], "w");
  if (out == NULL) {
    fprintf(stderr, "Error: Could not create %s\n", argv[2]);
    return 1;
  }
  const struct { int db, subjects; } sets[] = {{2, NINAPRO_DB2_SUBJECTS}, {3, NINAPRO_DB3_SUBJECTS}};
  for (int s = 0; s < 2; s++) {
    for (int subject = 1; subject <= sets[s].subjects; subject++) {
      int counts[INTENT_COUNT] = {0};
      if (write_subject(out, argv[1], sets[s].db, subject, counts) != 0) {
        fclose(out);
        remove(argv[2]);
        return 1;
      }
      printf("DB%d subject %2d:", sets[s].db, subject);
      for (int k = 0; k < INTENT_COUNT; k++)
        printf(" %s %d%s", intent_names[k], counts[k], k < INTENT_COUNT - 1 ? "," : "\n");
      fflush(stdout);
    }
  }
  fclose(out);
  return 0;
}
