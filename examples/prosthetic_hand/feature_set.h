/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FEATURE_SET_H
#define FEATURE_SET_H

// Desktop-only: loads the features.csv prepare_data.c writes and picks out
// the rows each experiment needs -- one person's, from some of their
// repetitions -- standardized the way intent_decoder.[ch] would with that
// person's calibration. Shared by train.c and test.c.

#include "emg_features.h"
#include "intent_decoder.h"

// Who trains what. The shared model (train.c) learns from DB2's first 28
// intact subjects and stops early on the next four; test.c evaluates on
// people neither has seen: DB2's last eight and all of DB3's amputees.
#define TRAIN_FIRST 1
#define TRAIN_LAST 28
#define VALIDATION_FIRST 29
#define VALIDATION_LAST 32
#define TEST_INTACT_FIRST 33
#define TEST_INTACT_LAST 40
// Ninapro's convention, and this example's: calibrate on some of
// repetitions 1, 3, 4 and 6; measure on 2 and 5, which come from the
// beginning and the middle of the session.
#define TEST_REPS (1u << 2 | 1u << 5)

typedef struct {
  int rows;
  int *db, *subject, *repetition, *intent;
  float *x; // rows x EF_FEATURES, raw
} features_t;

// A set of rows gathered into contiguous arrays.
typedef struct {
  int rows;
  float *x;   // rows x EF_FEATURES
  int *label; // intent of each row
} rowset_t;

// Bit r set: include repetition r (1..6).
#define REPS(...) reps_mask((int[]){__VA_ARGS__, 0})
unsigned reps_mask(const int *reps);

int features_load(const char *path, features_t *f);
void features_free(features_t *f);

// The rows of one subject whose repetition is in `reps`, raw.
rowset_t features_select(const features_t *f, int db, int subject, unsigned reps);

// Appends `b`'s rows to `a`.
void rowset_append(rowset_t *a, const rowset_t *b);

// Standardizes every row in place with `id`'s calibration.
void rowset_standardize(rowset_t *s, const intent_decoder_t *id);

void rowset_free(rowset_t *s);

// Every row of subjects first..last of `db`, each subject standardized
// with a calibration computed from all of their own rows -- how a shared
// model sees each training subject: as they'd look to a decoder calibrated
// to them.
rowset_t features_pool(const features_t *f, int db, int first, int last);

#endif /* FEATURE_SET_H */
