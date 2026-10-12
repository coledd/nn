/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently re-evaluates a model saved by train.c against the two
// held-out test sets prepare_data.c writes, neither of which training or
// early stopping ever touched:
//
//   test.csv        -- held-out Kaggle Cats and Dogs photos (whole frames)
//   test_oxford.csv -- the Oxford-IIIT Pet dataset's published test split
//
// and reports a confusion matrix and per-class/overall accuracy for each.
// For the Oxford set it also reports accuracy per breed (from
// test_oxford_labels.txt, one breed name per row), sorted worst-first:
// "cat vs. dog" is really dozens of breeds, and an overall number hides
// which of them the model can't tell apart.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_prep.h"
#include "image_prep.h"
#include "nn.h"

#define MAX_BREEDS 64
#define MAX_BREED_LEN 64

typedef struct {
  char name[MAX_BREED_LEN];
  int label; // CLASS_CAT or CLASS_DOG
  int correct;
  int total;
} breed_stats_t;

static int compare_accuracy(const void *a, const void *b)
{
  const breed_stats_t *x = a, *y = b;
  float ax = (float)x->correct / (float)x->total, ay = (float)y->correct / (float)y->total;
  return (ax > ay) - (ax < ay);
}

static void evaluate(nn_t *model, const char *csv_path, const char *labels_path, const char *description)
{
  data_t *data = data_load((char *)csv_path, IMG_INPUTS, CLASS_COUNT);
  if (data == NULL) {
    fprintf(stderr, "Error: Could not load %s. Run `make` first to build it from the dataset.\n", csv_path);
    return;
  }
  // Optional: without a labels file, just skip the per-breed breakdown.
  FILE *labels = labels_path ? fopen(labels_path, "r") : NULL;
  static breed_stats_t breeds[MAX_BREEDS];
  memset(breeds, 0, sizeof(breeds));
  int num_breeds = 0;

  // confusion[actual][predicted]
  int confusion[CLASS_COUNT][CLASS_COUNT] = {{0}};
  for (int i = 0; i < data->num_rows; i++) {
    float *out = nn_predict(model, data->input[i]);
    int actual = 0, predicted = 0;
    for (int c = 1; c < CLASS_COUNT; c++) {
      if (data->target[i][c] > data->target[i][actual])
        actual = c;
      if (out[c] > out[predicted])
        predicted = c;
    }
    confusion[actual][predicted]++;

    char breed[MAX_BREED_LEN];
    if (labels && fscanf(labels, "%63s", breed) == 1) {
      int b = 0;
      while (b < num_breeds && strcmp(breeds[b].name, breed) != 0)
        b++;
      if (b == num_breeds && num_breeds < MAX_BREEDS) {
        snprintf(breeds[b].name, sizeof(breeds[b].name), "%s", breed);
        breeds[b].label = actual;
        num_breeds++;
      }
      if (b < num_breeds) {
        breeds[b].total++;
        breeds[b].correct += (predicted == actual);
      }
    }
  }
  if (labels)
    fclose(labels);

  printf("%s (%s):\n", csv_path, description);
  printf("%-10s", "");
  for (int p = 0; p < CLASS_COUNT; p++)
    printf("%-10s", class_names[p]);
  printf("\n");
  int total_correct = 0, total = 0;
  for (int a = 0; a < CLASS_COUNT; a++) {
    printf("%-10s", class_names[a]);
    int class_total = 0;
    for (int p = 0; p < CLASS_COUNT; p++) {
      printf("%-10d", confusion[a][p]);
      class_total += confusion[a][p];
    }
    total += class_total;
    total_correct += confusion[a][a];
    printf("  (%d/%d = %.1f%%)\n", confusion[a][a], class_total,
           class_total ? 100.0f * confusion[a][a] / (float)class_total : 0.0f);
  }
  printf("Overall accuracy: %d/%d = %.2f%%\n", total_correct, total,
         total ? 100.0f * total_correct / (float)total : 0.0f);

  if (num_breeds > 0) {
    qsort(breeds, (size_t)num_breeds, sizeof(breed_stats_t), compare_accuracy);
    printf("\nAccuracy by breed (worst first):\n");
    for (int b = 0; b < num_breeds; b++)
      printf("  %-28s %-4s %3d/%3d = %5.1f%%\n", breeds[b].name, class_names[breeds[b].label],
             breeds[b].correct, breeds[b].total, 100.0f * breeds[b].correct / (float)breeds[b].total);
  }

  data_free(data);
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c (ascii, binary, or inplace)\n");
    return 1;
  }
  const char *model_path = argv[1];
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
  if ((model->width[0] != (uint32_t)IMG_INPUTS) || (model->width[model->depth - 1] != (uint32_t)CLASS_COUNT)) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match this task (%d inputs, %d outputs) -- is this a train.c model?\n",
            model->width[0], model->width[model->depth - 1], IMG_INPUTS, CLASS_COUNT);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }

  evaluate(model, "test.csv", NULL, "held-out Kaggle Cats and Dogs photos");
  printf("\n");
  evaluate(model, "test_oxford.csv", "test_oxford_labels.txt", "Oxford-IIIT Pet test split");

  nn_free(model);
  free(inplace_buf);
  return 0;
}
