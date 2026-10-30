/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently re-evaluates a model saved by train_gru.c or train_cnn.c
// against test.csv -- the held-out split prepare_data.c built from the
// dataset's testing_list.txt (speakers never seen in training), never
// touched during training or early-stopping validation. Works with either
// model: wake_word_probability() (wake_word_data.h) reads its shape.
// Reports the standard keyword-spotting metrics at the chosen threshold --
// false-reject rate (wake-word clips the model missed) and false-accept
// rate (everything-else clips the model wrongly flagged) -- plus overall
// accuracy, then the same two rates across a range of thresholds: the
// trade-off a product picks its operating point from. Pick it on
// validation.csv (`./test model.txt validation.csv`), then confirm it on
// test.csv, so the reported test numbers aren't tuned to the test set.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "data_prep.h"
#include "nn.h"
#include "wake_word_data.h"

// Width of one flattened CSV row (for data_load() below): the whole clip.
#define NUM_INPUTS (SEQUENCE_LEN * NUM_MEL_BINS)

// One clip-level confusion count at a given threshold.
typedef struct {
  int positives, negatives, false_rejects, false_accepts;
} counts_t;

static counts_t count_errors(const float *probability, data_t *data, float threshold)
{
  counts_t c = { 0, 0, 0, 0 };
  for (int i = 0; i < data->num_rows; i++) {
    bool predicted_wake_word = probability[i] >= threshold;
    if (data->target[i][0] >= 0.5f) {
      c.positives++;
      c.false_rejects += !predicted_wake_word;
    } else {
      c.negatives++;
      c.false_accepts += predicted_wake_word;
    }
  }
  return c;
}

static float percent(int n, int d)
{
  return d > 0 ? 100.0f * (float)n / (float)d : 0.0f;
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 4) {
    printf("Usage: %s <model-file> [data.csv] [threshold]\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train_gru.c or train_cnn.c (ascii, binary, or inplace)\n");
    printf("  [data.csv]   : Clips to score (default: test.csv)\n");
    printf("  [threshold]  : Wake-word probability at or above which a clip counts as the wake word (default: 0.5)\n");
    return 1;
  }
  const char *model_path = argv[1];
  const char *data_path = argc > 2 ? argv[2] : "test.csv";
  float threshold = argc > 3 ? strtof(argv[3], NULL) : 0.5f;

  data_t *test_data = data_load((char *)data_path, NUM_INPUTS, 1);
  if (test_data == NULL) {
    fprintf(stderr, "Error: Could not load %s. Run `make` first to build it from the dataset.\n", data_path);
    return 1;
  }

  // See examples/character_recognition/test.c's identical block for why the
  // inplace format needs to be read into a buffer first.
  nn_model_format_t format = nn_model_format(model_path);
  nn_t *model;
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
      data_free(test_data);
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
      data_free(test_data);
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
    data_free(test_data);
    return 1;
  }
  if (wake_word_probability(model, test_data->input[0]) < 0.0f) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match either wake-word model -- is this a train_gru.c or train_cnn.c model?\n",
            model->width[0], model->width[model->depth - 1]);
    nn_free(model);
    free(inplace_buf);
    data_free(test_data);
    return 1;
  }

  float *probability = malloc((size_t)test_data->num_rows * sizeof(float));
  for (int i = 0; i < test_data->num_rows; i++)
    probability[i] = wake_word_probability(model, test_data->input[i]);

  counts_t c = count_errors(probability, test_data, threshold);
  int correct = test_data->num_rows - c.false_rejects - c.false_accepts;
  printf("%s: %d clips (%d wake word, %d other), threshold %.2f\n", data_path, test_data->num_rows,
         c.positives, c.negatives, threshold);
  printf("Accuracy: %d/%d = %.2f%%\n", correct, test_data->num_rows, percent(correct, test_data->num_rows));
  printf("False-reject rate (missed wake word): %d/%d = %.1f%%\n", c.false_rejects, c.positives,
         percent(c.false_rejects, c.positives));
  printf("False-accept rate (wrongly triggered): %d/%d = %.2f%%\n", c.false_accepts, c.negatives,
         percent(c.false_accepts, c.negatives));

  printf("\nthreshold, false-reject %%, false-accept %%\n");
  for (int t = 1; t <= 19; t++) {
    float th = 0.05f * (float)t;
    counts_t ct = count_errors(probability, test_data, th);
    printf("%.2f, %.1f, %.2f\n", th, percent(ct.false_rejects, ct.positives), percent(ct.false_accepts, ct.negatives));
  }
  free(probability);

  data_free(test_data);
  nn_free(model);
  free(inplace_buf);
  return 0;
}
