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
#include "nn.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <model-file> <count>\n", argv[0]);
    printf("  <model-file> : Path to the neural-net model to prune (e.g., model.txt)\n");
    printf("  <count> : How many neurons to remove.\n");
    return 1;
  }
  const char *model_path = argv[1];
  int count = 1;
  if (argc > 2) {
    count = atoi(argv[2]);
  }

  // Load the model. nn_load_model() already auto-detects ascii vs. binary,
  // but the inplace format needs a mutable model to prune in place
  // (nn_prune_lightest_neuron()/nn_remove_neuron() refuse to run on the
  // read-only model nn_load_model_inplace() itself would produce -- its
  // weight arrays are aliased into the input buffer, not owned), so use
  // nn_load_model_inplace_copy() instead when nn_model_format() reports
  // that's what the input is; that copies the weights into normal owned
  // allocations, so the buffer can be freed right after loading.
  nn_model_format_t format = nn_model_format(model_path);
  nn_t *nn;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t *buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!buf || fread(buf, 1, (size_t)size, file) != (size_t)size) {
      fprintf(stderr, "Error: Could not read model file: %s\n", model_path);
      fclose(file);
      free(buf);
      return 1;
    }
    fclose(file);
    nn = nn_load_model_inplace_copy(buf, (size_t)size);
    free(buf); // copied, not aliased -- safe to free immediately
  } else {
    nn = nn_load_model((char *)model_path);
  }
  if (nn == NULL) {
    fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
    return 1;
  }
  while (count-- > 0) {
    nn_prune_lightest_neuron(nn);
  }
  // Save back in the same format the model was actually loaded as -- not
  // by file extension (nn_save_model() picks ascii vs. binary by whether
  // the path ends in ".bin", which silently changes the format on a
  // binary-formatted file with some other extension; nn_model_format()
  // instead reports what the file actually is).
  nn_error_t save_err;
  switch (format) {
    case NN_MODEL_FORMAT_ASCII:   save_err = nn_save_model_ascii(nn, (char *)model_path); break;
    case NN_MODEL_FORMAT_BINARY:  save_err = nn_save_model_binary(nn, (char *)model_path); break;
    case NN_MODEL_FORMAT_INPLACE: save_err = nn_save_model_inplace(nn, (char *)model_path); break;
    default:                      save_err = NN_ERROR_INVALID_ARGUMENT; break;
  }
  if (save_err != NN_ERROR_NONE) {
    fprintf(stderr, "Error: Failed to save model: %s\n", model_path);
    nn_free(nn);
    return 1;
  }
  nn_free(nn);
  return 0;
}
