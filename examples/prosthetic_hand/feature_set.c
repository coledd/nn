/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_set.h"

unsigned reps_mask(const int *reps)
{
  unsigned mask = 0;
  for (; *reps != 0; reps++)
    mask |= 1u << *reps;
  return mask;
}

static void *xmalloc(size_t size)
{
  void *p = malloc(size);
  if (p == NULL) {
    fprintf(stderr, "Error: Out of memory.\n");
    exit(1);
  }
  return p;
}

int features_load(const char *path, features_t *f)
{
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "Error: Could not open %s (run make to build it)\n", path);
    return -1;
  }
  int capacity = 1 << 16;
  f->rows = 0;
  f->db = (int *)xmalloc((size_t)capacity * sizeof(int));
  f->subject = (int *)xmalloc((size_t)capacity * sizeof(int));
  f->repetition = (int *)xmalloc((size_t)capacity * sizeof(int));
  f->intent = (int *)xmalloc((size_t)capacity * sizeof(int));
  f->x = (float *)xmalloc((size_t)capacity * EF_FEATURES * sizeof(float));
  char line[4096];
  while (fgets(line, sizeof(line), file) != NULL) {
    if (f->rows == capacity) {
      capacity *= 2;
      f->db = (int *)realloc(f->db, (size_t)capacity * sizeof(int));
      f->subject = (int *)realloc(f->subject, (size_t)capacity * sizeof(int));
      f->repetition = (int *)realloc(f->repetition, (size_t)capacity * sizeof(int));
      f->intent = (int *)realloc(f->intent, (size_t)capacity * sizeof(int));
      f->x = (float *)realloc(f->x, (size_t)capacity * EF_FEATURES * sizeof(float));
      if (f->db == NULL || f->subject == NULL || f->repetition == NULL || f->intent == NULL || f->x == NULL) {
        fprintf(stderr, "Error: Out of memory.\n");
        exit(1);
      }
    }
    int r = f->rows;
    char *p = line, *end;
    int *meta[4] = {&f->db[r], &f->subject[r], &f->repetition[r], &f->intent[r]};
    for (int m = 0; m < 4; m++) {
      *meta[m] = (int)strtol(p, &end, 10);
      p = end + 1;
    }
    for (int i = 0; i < EF_FEATURES; i++) {
      f->x[(size_t)r * EF_FEATURES + i] = strtof(p, &end);
      p = end + 1;
    }
    f->rows++;
  }
  fclose(file);
  return 0;
}

void features_free(features_t *f)
{
  free(f->db);
  free(f->subject);
  free(f->repetition);
  free(f->intent);
  free(f->x);
}

rowset_t features_select(const features_t *f, int db, int subject, unsigned reps)
{
  rowset_t s = {0, NULL, NULL};
  int n = 0;
  for (int r = 0; r < f->rows; r++)
    if (f->db[r] == db && f->subject[r] == subject && (reps >> f->repetition[r]) & 1u)
      n++;
  s.x = (float *)xmalloc((size_t)(n > 0 ? n : 1) * EF_FEATURES * sizeof(float));
  s.label = (int *)xmalloc((size_t)(n > 0 ? n : 1) * sizeof(int));
  for (int r = 0; r < f->rows; r++) {
    if (f->db[r] == db && f->subject[r] == subject && (reps >> f->repetition[r]) & 1u) {
      memcpy(s.x + (size_t)s.rows * EF_FEATURES, f->x + (size_t)r * EF_FEATURES, EF_FEATURES * sizeof(float));
      s.label[s.rows++] = f->intent[r];
    }
  }
  return s;
}

void rowset_append(rowset_t *a, const rowset_t *b)
{
  a->x = (float *)realloc(a->x, (size_t)(a->rows + b->rows) * EF_FEATURES * sizeof(float));
  a->label = (int *)realloc(a->label, (size_t)(a->rows + b->rows) * sizeof(int));
  if (a->x == NULL || a->label == NULL) {
    fprintf(stderr, "Error: Out of memory.\n");
    exit(1);
  }
  memcpy(a->x + (size_t)a->rows * EF_FEATURES, b->x, (size_t)b->rows * EF_FEATURES * sizeof(float));
  memcpy(a->label + a->rows, b->label, (size_t)b->rows * sizeof(int));
  a->rows += b->rows;
}

void rowset_standardize(rowset_t *s, const intent_decoder_t *id)
{
  for (int r = 0; r < s->rows; r++)
    intent_decoder_standardize(id, s->x + (size_t)r * EF_FEATURES, s->x + (size_t)r * EF_FEATURES);
}

void rowset_free(rowset_t *s)
{
  free(s->x);
  free(s->label);
  s->x = NULL;
  s->label = NULL;
  s->rows = 0;
}

rowset_t features_pool(const features_t *f, int db, int first, int last)
{
  rowset_t pool = {0, NULL, NULL};
  for (int subject = first; subject <= last; subject++) {
    rowset_t s = features_select(f, db, subject, ~0u);
    intent_decoder_t id;
    intent_decoder_init(&id);
    intent_decoder_calibrate(&id, s.x, s.label, s.rows);
    rowset_standardize(&s, &id);
    rowset_append(&pool, &s);
    rowset_free(&s);
  }
  return pool;
}
