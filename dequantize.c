/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

 /*
 * Converts a quantized neural net model into a floating-point neural network.
 */

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <input_model> <output_model>\n", argv[0]);
    printf("  input_model:  path to the quantized neural network model (ascii, binary, or inplace)\n");
    printf("  output_model: path where to save the dequantized model, in the same format as input_model\n");
    return 1;
  }
  const char *input_model = argv[1];
  const char *output_model = argv[2];

  // Load the quantized network. nn_load_model() already auto-detects ascii
  // vs. binary, but the inplace format needs a mutable model to dequantize
  // in place (nn_dequantize() refuses to run on the read-only model
  // nn_load_model_inplace() itself would produce -- its weight arrays are
  // aliased into the input buffer, not owned), so use
  // nn_load_model_inplace_copy() instead when nn_model_format() reports
  // that's what the input is; that copies the weights into normal owned
  // allocations, so the buffer can be freed right after loading.
  nn_model_format_t format = nn_model_format(input_model);
  nn_t *network;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(input_model, "rb");
    if (!file) {
      fprintf(stderr, "Failed to open input model: %s\n", input_model);
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    uint8_t *buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!buf || fread(buf, 1, (size_t)size, file) != (size_t)size) {
      fprintf(stderr, "Failed to read input model: %s\n", input_model);
      fclose(file);
      free(buf);
      return 1;
    }
    fclose(file);
    network = nn_load_model_inplace_copy(buf, (size_t)size);
    free(buf); // copied, not aliased -- safe to free immediately
  } else {
    network = nn_load_model((char *)input_model);
  }
  if (!network) {
    fprintf(stderr, "Failed to load input model: %s\n", input_model);
    return 1;
  }
  // Dequantize the network
  if (!network->quantized) {
    fprintf(stderr, "Network has not been quantized\n");
    nn_free(network);
    return 1;
  }
  if (nn_dequantize(network) != 0) {
    fprintf(stderr, "Failed to dequantize network\n");
    nn_free(network);
    return 1;
  }
  // Save the dequantized network in the same format the input model was in.
  nn_error_t save_err;
  switch (format) {
    case NN_MODEL_FORMAT_ASCII:   save_err = nn_save_model_ascii(network, output_model); break;
    case NN_MODEL_FORMAT_BINARY:  save_err = nn_save_model_binary(network, output_model); break;
    case NN_MODEL_FORMAT_INPLACE: save_err = nn_save_model_inplace(network, output_model); break;
    default:                      save_err = NN_ERROR_INVALID_ARGUMENT; break;
  }
  if (save_err != NN_ERROR_NONE) {
    fprintf(stderr, "Failed to save dequantized model: %s\n", output_model);
    nn_free(network);
    return 1;
  }
  printf("Successfully dequantized model\n");
  printf("  Input:  %s\n", input_model);
  printf("  Output: %s\n", output_model);
  nn_free(network);
  return 0;
}
