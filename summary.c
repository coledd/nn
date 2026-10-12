/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

// Activation function names must exactly match the enum in nn.h
static const char *activation_names[] = {
    "NONE",      "LINEAR",  "RELU",         "LEAKY_RELU", "ELU",
    "THRESHOLD", "SIGMOID", "SIGMOID_FAST", "TANH",       "TANH_FAST",
    "GELU",      "SILU",    "SOFTMAX"};

static const char *layer_types[] = {
    "NONE",        "FC",         "CNN",        "POOL",       "LSTM",
    "GRU",         "RNN",        "ATTENTION",  "TRANSFORMER", "INPUT",
    "OUTPUT",      "DROPOUT"};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a saved neural-net model (e.g., model.txt)\n");
    return 1;
  }
  const char *model_path = argv[1];
  nn_model_format_t format = nn_model_format(model_path);
  const char *format_name;
  switch (format) {
    case NN_MODEL_FORMAT_ASCII:   format_name = "ASCII"; break;
    case NN_MODEL_FORMAT_BINARY:  format_name = "Binary"; break;
    case NN_MODEL_FORMAT_INPLACE: format_name = "Binary (in-place / zero-copy)"; break;
    default:                      format_name = "Unknown"; break;
  }

  nn_t *network;
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    // The inplace format aliases weight/bias data directly out of a
    // caller-supplied buffer (see nn_load_model_inplace() in nn.h) rather
    // than being read as a path, so read the whole file into memory
    // ourselves first; the buffer must then outlive `network`.
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
  printf("Model Format:\t%s\n", format_name);
  printf("Model Version:\t%u.%u.%u.%u\n", (unsigned)network->version_major,
         (unsigned)network->version_minor, (unsigned)network->version_patch,
         (unsigned)network->version_build);
  uint32_t lib_ver = nn_version();
  unsigned lib_major = (lib_ver >> 24) & 0xFF;
  unsigned lib_minor = (lib_ver >> 16) & 0xFF;
  unsigned lib_patch = (lib_ver >> 8) & 0xFF;
  unsigned lib_build = lib_ver & 0xFF;
  printf("NN Lib Version:\t%u.%u.%u.%u\n", lib_major, lib_minor, lib_patch, lib_build);
  if (network->quantized) {
    printf("Model Type:\tQuantized (fixed-point int8)\n");
  } else {
    printf("Model Type:\tFloating-point\n");
  }
  printf("Layer\tType\tWidth\tActivation\n");
  for (int i = 0; i < network->depth; i++) {
    const char *act_name = "UNKNOWN";
    uint8_t act_code = network->activation[i];
    if (act_code < (sizeof(activation_names) / sizeof(activation_names[0]))) {
      act_name = activation_names[act_code];
    }
    printf("%d\t%s\t%u\t%s\n", i, layer_types[network->layer_type[i]], network->width[i], act_name);
  }
  nn_free(network);
  free(inplace_buf);
  return 0;
}
