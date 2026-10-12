/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently re-evaluates a model saved by train.c as a detector and a
// counter, against the labeled boxes prepare_data.c wrote alongside each
// evaluation CSV -- not against the heatmap targets themselves, which
// can't say how big an object is or which detection belongs to which
// object.
//
// Each photo's heatmaps are decoded into detections exactly the way
// predict.c (and a deployed device) does it, with heatmap_decode(). A
// detection is a hit if its center falls inside a labeled box of the same
// class that no higher-scoring detection has already claimed (the nearest,
// if several qualify); otherwise it's a false positive, and every labeled
// box left unclaimed is a miss. From those, per class:
//
//   precision -- the fraction of detections that are real objects
//   recall    -- the fraction of real objects that were detected
//   count MAE -- mean absolute difference, per photo, between the number of
//                detections and the number of labeled objects (what a cell
//                counter is actually judged on)
//   count     -- total detections vs. total labeled objects over the set,
//                which shows whether the model systematically over- or
//                under-counts
//
// The threshold is chosen on validation.csv, never test.csv: test.c first
// sweeps it there and reports the result, then evaluates test.csv once at
// HEATMAP_THRESHOLD (or the threshold given on the command line).

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_prep.h"
#include "heatmap.h"
#include "image_prep.h"
#include "nn.h"

#define MAX_OBJECTS 128
#define MAX_DETECTIONS 256

typedef struct {
  int cls;
  float x0, y0, x1, y1;
} box_t;

typedef struct {
  char name[256];
  int width, height;
  int num_boxes;
  box_t boxes[MAX_OBJECTS];
} photo_t;

typedef struct {
  int tp[CLASS_COUNT], fp[CLASS_COUNT], fn[CLASS_COUNT];
  int predicted[CLASS_COUNT], labeled[CLASS_COUNT];
  float abs_count_error[CLASS_COUNT];
  int photos;
} stats_t;

// Reads one line of a *_boxes.txt file (see prepare_data.c). Returns 1, or 0
// at end of file or on a malformed line.
static int read_photo(FILE *file, photo_t *photo)
{
  if (fscanf(file, "%255s %d %d %d", photo->name, &photo->width, &photo->height, &photo->num_boxes) != 4)
    return 0;
  if (photo->num_boxes < 0 || photo->num_boxes > MAX_OBJECTS)
    return 0;
  for (int b = 0; b < photo->num_boxes; b++) {
    box_t *box = &photo->boxes[b];
    if (fscanf(file, "%d %f %f %f %f", &box->cls, &box->x0, &box->y0, &box->x1, &box->y1) != 5)
      return 0;
  }
  return 1;
}

static int compare_score(const void *a, const void *b)
{
  const detection_t *x = a, *y = b;
  return (x->score < y->score) - (x->score > y->score);
}

// Scores one photo's detections against its labeled boxes, adding to `stats`.
static void score_photo(const photo_t *photo, detection_t *detections, int num_detections, stats_t *stats)
{
  int claimed[MAX_OBJECTS] = { 0 };
  int predicted[CLASS_COUNT] = { 0 }, labeled[CLASS_COUNT] = { 0 };
  // Highest-scoring detections claim boxes first.
  qsort(detections, (size_t)num_detections, sizeof(detection_t), compare_score);
  for (int d = 0; d < num_detections; d++) {
    const detection_t *det = &detections[d];
    float x = det->x * photo->width, y = det->y * photo->height;
    int best = -1;
    float best_dist = INFINITY;
    for (int b = 0; b < photo->num_boxes; b++) {
      const box_t *box = &photo->boxes[b];
      if (claimed[b] || box->cls != det->cls || x < box->x0 || x > box->x1 || y < box->y0 || y > box->y1)
        continue;
      float dx = x - (box->x0 + box->x1) * 0.5f, dy = y - (box->y0 + box->y1) * 0.5f;
      if (dx * dx + dy * dy < best_dist) {
        best_dist = dx * dx + dy * dy;
        best = b;
      }
    }
    if (best >= 0) {
      claimed[best] = 1;
      stats->tp[det->cls]++;
    } else {
      stats->fp[det->cls]++;
    }
    predicted[det->cls]++;
  }
  for (int b = 0; b < photo->num_boxes; b++) {
    labeled[photo->boxes[b].cls]++;
    if (!claimed[b])
      stats->fn[photo->boxes[b].cls]++;
  }
  for (int c = 0; c < CLASS_COUNT; c++) {
    stats->predicted[c] += predicted[c];
    stats->labeled[c] += labeled[c];
    stats->abs_count_error[c] += fabsf((float)(predicted[c] - labeled[c]));
  }
  stats->photos++;
}

// Runs every row of an evaluation set through the model once, keeping its
// heatmaps (so a threshold sweep doesn't have to rerun the network), and
// reads the matching boxes file. Returns the number of photos, or -1.
static int load_set(nn_t *model, const char *csv_path, const char *boxes_path, float **heatmaps, photo_t **photos)
{
  data_t *data = data_load((char *)csv_path, IMG_INPUTS, HEATMAP_OUTPUTS);
  if (data == NULL) {
    fprintf(stderr, "Error: Could not load %s. Run `make` first to build it from the dataset.\n", csv_path);
    return -1;
  }
  FILE *file = fopen(boxes_path, "r");
  if (!file) {
    fprintf(stderr, "Error: Could not read %s. Run `make` first to build it from the dataset.\n", boxes_path);
    data_free(data);
    return -1;
  }
  int n = data->num_rows;
  *heatmaps = malloc((size_t)n * HEATMAP_OUTPUTS * sizeof(float));
  *photos = malloc((size_t)n * sizeof(photo_t));
  if (!*heatmaps || !*photos) {
    fprintf(stderr, "Error: Out of memory\n");
    fclose(file);
    data_free(data);
    return -1;
  }
  for (int i = 0; i < n; i++) {
    if (!read_photo(file, &(*photos)[i])) {
      fprintf(stderr, "Error: %s has fewer (or malformed) lines than %s has rows\n", boxes_path, csv_path);
      fclose(file);
      data_free(data);
      return -1;
    }
    memcpy(*heatmaps + (size_t)i * HEATMAP_OUTPUTS, nn_predict(model, data->input[i]), HEATMAP_OUTPUTS * sizeof(float));
  }
  fclose(file);
  data_free(data);
  return n;
}

static void evaluate(const float *heatmaps, const photo_t *photos, int n, float threshold, stats_t *stats)
{
  static detection_t detections[MAX_DETECTIONS];
  memset(stats, 0, sizeof(*stats));
  for (int i = 0; i < n; i++) {
    int found = heatmap_decode(heatmaps + (size_t)i * HEATMAP_OUTPUTS, threshold, detections, MAX_DETECTIONS);
    if (found > MAX_DETECTIONS)
      found = MAX_DETECTIONS;
    score_photo(&photos[i], detections, found, stats);
  }
}

static float ratio(int a, int b)
{
  return b ? (float)a / (float)b : 0.0f;
}

static float f1(const stats_t *s, int c)
{
  float p = ratio(s->tp[c], s->tp[c] + s->fp[c]), r = ratio(s->tp[c], s->tp[c] + s->fn[c]);
  return p + r > 0.0f ? 2.0f * p * r / (p + r) : 0.0f;
}

static void print_stats(const stats_t *s)
{
  printf("%-10s %9s %9s %9s %10s %16s\n", "class", "precision", "recall", "F1", "count MAE", "count (vs. labeled)");
  for (int c = 0; c < CLASS_COUNT; c++) {
    printf("%-10s %8.1f%% %8.1f%% %8.1f%% %10.2f %8d (%d)\n", class_names[c],
           100.0f * ratio(s->tp[c], s->tp[c] + s->fp[c]), 100.0f * ratio(s->tp[c], s->tp[c] + s->fn[c]),
           100.0f * f1(s, c), s->abs_count_error[c] / (float)s->photos, s->predicted[c], s->labeled[c]);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2 && argc != 3) {
    printf("Usage: %s <model-file> [<threshold>]\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c (ascii, binary, or inplace)\n");
    printf("  <threshold>  : Heatmap peak threshold for test.csv (default %.2f)\n", HEATMAP_THRESHOLD);
    return 1;
  }
  const char *model_path = argv[1];
  float threshold = argc == 3 ? strtof(argv[2], NULL) : HEATMAP_THRESHOLD;
  // See examples/character_recognition/test.c's identical block for why the
  // inplace format needs to be read into a buffer first.
  nn_model_format_t format = nn_model_format(model_path);
  nn_t *model;
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    inplace_buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!inplace_buf || fread(inplace_buf, 1, (size_t)size, file) != (size_t)size) {
      fprintf(stderr, "Error: Could not read model file: %s\n", model_path);
      fclose(file);
      free(inplace_buf);
      return 1;
    }
    fclose(file);
    model = nn_load_model_inplace(inplace_buf, (size_t)size);
  } else {
    model = nn_load_model((char *)model_path);
  }
  if (model == NULL) {
    fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
    free(inplace_buf);
    return 1;
  }
  if ((model->width[0] != (uint32_t)IMG_INPUTS) || (model->width[model->depth - 1] != (uint32_t)HEATMAP_OUTPUTS)) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match this task (%d inputs, %d outputs) -- is this a train.c model?\n",
            model->width[0], model->width[model->depth - 1], IMG_INPUTS, HEATMAP_OUTPUTS);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }

  int status = 1;
  float *val_heatmaps = NULL, *test_heatmaps = NULL;
  photo_t *val_photos = NULL, *test_photos = NULL;
  int val_n = load_set(model, "validation.csv", "validation_boxes.txt", &val_heatmaps, &val_photos);
  int test_n = val_n < 0 ? -1 : load_set(model, "test.csv", "test_boxes.txt", &test_heatmaps, &test_photos);
  if (test_n >= 0) {
    stats_t stats;
    printf("Threshold sweep on validation.csv (%d photos), F1 per class:\n", val_n);
    printf("%-10s", "threshold");
    for (int c = 0; c < CLASS_COUNT; c++)
      printf(" %9s", class_names[c]);
    printf("\n");
    for (int t = 1; t <= 9; t++) {
      evaluate(val_heatmaps, val_photos, val_n, t * 0.1f, &stats);
      printf("%-10.1f", t * 0.1f);
      for (int c = 0; c < CLASS_COUNT; c++)
        printf(" %8.1f%%", 100.0f * f1(&stats, c));
      printf("\n");
    }
    printf("\ntest.csv (%d photos, BCCD's published test split), threshold %.2f:\n", test_n, threshold);
    evaluate(test_heatmaps, test_photos, test_n, threshold, &stats);
    print_stats(&stats);
    status = 0;
  }
  free(val_heatmaps);
  free(val_photos);
  free(test_heatmaps);
  free(test_photos);
  nn_free(model);
  free(inplace_buf);
  return status;
}
