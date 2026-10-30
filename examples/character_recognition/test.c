/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "data_prep.h"
#include "nn.h"

// Scores `model` on every row of `path`: a digit counts as correct when
// the model's most probable class is its label, the usual way MNIST
// results are reported. Prints "<name>: correct/total = accuracy".
static int score(nn_t *model, const char *path, const char *name)
{
  int num_outputs = (int)model->width[model->depth - 1];
  data_t *data = data_load((char *)path, (int)model->width[0], num_outputs);
  if (data == NULL) {
    printf("Error: Could not load %s.\n", path);
    return 1;
  }
  int correct = 0;
  for (int i = 0; i < data->num_rows; i++) {
    // nn_predict() manages its own buffers; don't free the result.
    float *prediction = nn_predict(model, data->input[i]);
    int predicted = 0, label = 0;
    for (int j = 1; j < num_outputs; j++) {
      if (prediction[j] > prediction[predicted])
        predicted = j;
      if (data->target[i][j] > data->target[i][label])
        label = j;
    }
    correct += predicted == label;
  }
  printf("%s: %d/%d = %2.2f%%\n", name, correct, data->num_rows, (correct * 100.0f) / (float)data->num_rows);
  data_free(data);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a saved neural-net model (ascii, binary, or inplace; e.g., model.txt)\n");
    return 1;
  }
  // Load a previously saved model (specified by the user). nn_load_model()
  // already auto-detects ascii vs. binary, but the inplace format has to be
  // read from a memory buffer rather than a path (see nn_load_model_inplace()
  // in nn.h), so read the whole file ourselves first when nn_model_format()
  // reports that's what it is; the buffer must then outlive `model`.
  const char *model_path = argv[1];
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
  if (score(model, "train.csv", "Train") != 0 || score(model, "test.csv", "Test ") != 0) {
    nn_free(model);
    free(inplace_buf);
    return 1;
  }
  nn_free(model);
  free(inplace_buf);
  return 0;
}
