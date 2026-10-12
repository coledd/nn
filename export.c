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
#include <string.h>
#include "nn.h"

int main(int argc, char *argv[])
{
  if (argc != 3 && argc != 4) {
    printf("Usage: %s <input_model> <output_model> [--inplace]\n", argv[0]);
    printf("  input_model:  path to an ascii, binary, or inplace neural network model\n");
    printf("  output_model: path to the binary model to create\n");
    printf("  --inplace:    write the zero-copy \"inplace\" format (magic NNP1,\n");
    printf("                see nn_load_model_inplace() in nn.h) instead of the\n");
    printf("                regular binary format (magic NNB1)\n");
    return 1;
  }
  bool inplace_out = false;
  if (argc == 4) {
    if (strcmp(argv[3], "--inplace") != 0) {
      fprintf(stderr, "Unknown option: %s\n", argv[3]);
      return 1;
    }
    inplace_out = true;
  }
  const char *model_path = argv[1];

  // Accept any of the three input formats: nn_load_model() already
  // auto-detects ascii vs. binary, but the inplace format has to be read
  // from a memory buffer rather than a path (see nn_load_model_inplace()
  // in nn.h), so read the whole file ourselves first when that's what
  // nn_model_format() reports -- the buffer must then outlive `network`.
  nn_model_format_t format = nn_model_format(model_path);
  nn_t *network;
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      fprintf(stderr, "Failed to open model: %s\n", model_path);
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    inplace_buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!inplace_buf || fread(inplace_buf, 1, (size_t)size, file) != (size_t)size) {
      fprintf(stderr, "Failed to read model: %s\n", model_path);
      fclose(file);
      free(inplace_buf);
      return 1;
    }
    fclose(file);
    network = nn_load_model_inplace(inplace_buf, (size_t)size);
  } else {
    network = nn_load_model(model_path);
  }
  if (!network) {
    fprintf(stderr, "Failed to load model: %s\n", model_path);
    free(inplace_buf);
    return 1;
  }
  nn_error_t err = inplace_out ? nn_save_model_inplace(network, argv[2]) : nn_save_model_binary(network, argv[2]);
  if (err != NN_ERROR_NONE) {
    fprintf(stderr, "Failed to save model: %s\n", argv[2]);
    nn_free(network);
    free(inplace_buf);
    return 1;
  }
  nn_free(network);
  free(inplace_buf);
  return 0;
}
