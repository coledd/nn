/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently re-evaluates a model saved by train.c against test.csv --
// the held-out third split split.py
// carved off from samples.csv, never touched during training or
// early-stopping validation. Architecture-agnostic, like
// examples/character_recognition/test.c and examples/fall_detection/
// test.c: this just reads whatever the model's input/output widths are.
// Reports overall clip-level accuracy plus the standard keyword-spotting
// metrics: false-reject rate (wake-word clips the model missed) and
// false-accept rate (everything-else clips the model wrongly flagged).

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "data_prep.h"
#include "nn.h"
#include "wake_word_data.h"

// Width of one flattened CSV row (for data_load() below) -- NOT the
// network's own INPUT layer width (NUM_MEL_BINS, one frame -- see
// train.c's identical NUM_INPUTS comment for why).
#define NUM_INPUTS (SEQUENCE_LEN * NUM_MEL_BINS)

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c (ascii, binary, or inplace)\n");
    return 1;
  }
  const char *model_path = argv[1];

  data_t *test_data = data_load("test.csv", NUM_INPUTS, 1);
  if (test_data == NULL) {
    fprintf(stderr, "Error: Could not load test data (test.csv). Run `make` first to build it from the dataset.\n");
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
  if ((model->width[0] != (uint32_t)NUM_MEL_BINS) || (model->width[model->depth - 1] != 1)) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match the wake-word task (%d inputs, 1 output) -- is this a train.c model?\n",
            model->width[0], model->width[model->depth - 1], NUM_MEL_BINS);
    nn_free(model);
    free(inplace_buf);
    data_free(test_data);
    return 1;
  }

  int correct = 0, false_accepts = 0, false_rejects = 0, positives = 0, negatives = 0;
  for (int i = 0; i < test_data->num_rows; i++) {
    nn_reset_state(model);
    float *pred = NULL;
    for (int t = 0; t < SEQUENCE_LEN; t++)
      pred = nn_predict(model, test_data->input[i] + t * NUM_MEL_BINS);
    bool predicted_wake_word = pred[0] >= 0.5f;
    bool actual_wake_word = test_data->target[i][0] >= 0.5f;
    if (predicted_wake_word == actual_wake_word)
      correct++;
    if (actual_wake_word) {
      positives++;
      if (!predicted_wake_word)
        false_rejects++;
    } else {
      negatives++;
      if (predicted_wake_word)
        false_accepts++;
    }
  }

  printf("Test clips: %d (%d wake word, %d other), %d frames each\n",
         test_data->num_rows, positives, negatives, SEQUENCE_LEN);
  printf("Accuracy: %d/%d = %.2f%%\n", correct, test_data->num_rows,
         100.0f * (float)correct / (float)test_data->num_rows);
  if (positives > 0)
    printf("False-reject rate (missed wake word): %d/%d = %.1f%%\n", false_rejects, positives,
           100.0f * (float)false_rejects / (float)positives);
  if (negatives > 0)
    printf("False-accept rate (wrongly triggered): %d/%d = %.1f%%\n", false_accepts, negatives,
           100.0f * (float)false_accepts / (float)negatives);

  data_free(test_data);
  nn_free(model);
  free(inplace_buf);
  return 0;
}
