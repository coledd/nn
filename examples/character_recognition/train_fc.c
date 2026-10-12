/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Trains a plain fully-connected (FC) network -- no convolution, no
// pooling -- on the same MNIST handwritten-digit task and the same
// train.csv/validation.csv data as train_cnn.c in this directory. Every
// pixel is just another input to the first FC layer; there's no notion of
// spatial locality or shared weights the way a convolution has. It's the
// baseline a CNN is normally justified against: expect this to train
// slower and top out at a lower accuracy than train_cnn.c on the same
// data, which is the point of having both examples side by side.

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_prep.h"
#include "nn.h"

// See train_cnn.c's identical constants for the early-stopping rationale.
#define EARLY_STOPPING_PATIENCE 5
#define MAX_EPOCHS 500

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to the neural net model to load or create (e.g., model.txt or model.bin)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Tunable hyperparameters
  int num_inputs = 28 * 28;
  int num_outputs = 10;
  // Softmax + cross-entropy's output-layer gradient is the raw
  // target-prediction (see ACTIVATION_FUNCTION_TYPE_SOFTMAX in nn.h), a
  // larger effective step than sigmoid+MSE's (damped by up to a 0.25x
  // sigmoid-derivative factor) -- see train_cnn.c's identical comment for a
  // specific diagnosed case of this driving a ReLU-family layer's units
  // permanently negative at a higher rate. GELU (not ReLU) for the same
  // reason as train_cnn.c: it dips slightly negative instead of flatlining
  // at zero, so a unit that drifts negative still has a nonzero gradient
  // and can recover.
  float learning_rate = 0.005f;
  float annealing = 1.0f;
  // End of tunable parameters
  data_t *train_data;
  data_t *validation_data;
  nn_t *nn;
  int j;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  float total_train_error = 0.0f;
  float total_validation_err = 0.0f;
  // Set the random seed
  srand((unsigned)time(NULL));
  // Load the training data into memory
  train_data = data_load("train.csv", num_inputs, num_outputs);
  if (train_data == NULL) {
    printf("Error: Could not load training data (train.csv).\n");
    return 1;
  }
  // Load the validation data into memory
  validation_data = data_load("validation.csv", num_inputs, num_outputs);
  if (validation_data == NULL) {
    printf("Error: Could not load validation data (validation.csv).\n");
    data_free(train_data);
    return 1;
  }
  // Attempt to load an existing model. See train_cnn.c's identical block
  // for why the inplace format needs nn_load_model_inplace_copy() instead
  // of plain nn_load_model() to keep training.
  nn_model_format_t existing_format = nn_model_format(model_path);
  if (existing_format == NN_MODEL_FORMAT_INPLACE) {
    nn = NULL;
    FILE *file = fopen(model_path, "rb");
    if (file) {
      fseek(file, 0, SEEK_END);
      long size = ftell(file);
      fseek(file, 0, SEEK_SET);
      uint8_t *buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
      if (buf && fread(buf, 1, (size_t)size, file) == (size_t)size) {
        nn = nn_load_model_inplace_copy(buf, (size_t)size);
      }
      fclose(file);
      free(buf);
    }
  } else {
    nn = nn_load_model((char *)model_path);
  }
  bool resuming = (nn != NULL);
  if (nn == NULL) {
    printf("Creating new model.\n");
    nn = nn_init();
    if (nn == NULL) {
      printf("Error: Failed to initialize new neural network.\n");
      data_free(train_data);
      data_free(validation_data);
      return 1;
    }
    // Construct the neural network, layer by layer -- every pixel feeds
    // directly into the first FC layer, no convolution/pooling ahead of it.
    nn_add_layer(nn, LAYER_TYPE_INPUT, num_inputs, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    nn_add_layer(nn, LAYER_TYPE_FC, 128, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    nn_add_layer(nn, LAYER_TYPE_FC, 64, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    // Softmax + cross-entropy (the loss nn_train()/nn_error() switch to
    // automatically for a softmax output -- see ACTIVATION_FUNCTION_TYPE_SOFTMAX
    // in nn.h) models these 10 digits as mutually exclusive classes.
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, num_outputs, ACTIVATION_FUNCTION_TYPE_SOFTMAX, NULL);
  } else {
    printf("Using existing model file: %s\n", model_path);
    // Verify that model dimensions match expected inputs/outputs
    if ((nn->width[0] != (uint32_t)num_inputs) ||
        (nn->width[nn->depth - 1] != (uint32_t)num_outputs)) {
      printf("Error: Loaded model dimensions do not match expected: %d %d.\n", num_inputs, num_outputs);
      nn_free(nn);
      data_free(train_data);
      data_free(validation_data);
      return 1;
    }
  }
  // See train_cnn.c's identical block for why this baseline matters when
  // resuming an existing model.
  float best_validation_error = FLT_MAX;
  if (resuming) {
    total_validation_err = 0.0f;
    for (j = 0; j < validation_data->num_rows; j++) {
      total_validation_err += nn_error(nn, validation_data->input[j], validation_data->target[j]);
    }
    best_validation_error = total_validation_err / (float)validation_data->num_rows;
    printf("Starting validation error (existing model): %.5f\n", best_validation_error);
  }
  int epochs_since_improvement = 0;
  printf("train error, validation error, learning rate\n");
  while (epochs < MAX_EPOCHS) {
    // It is critical to shuffle training data before each epoch
    data_shuffle(train_data);
    // Train on each row of training data
    total_train_error = 0.0f;
    for (j = 0; j < train_data->num_rows; j++) {
      float *input = train_data->input[j];
      float *target = train_data->target[j];
      total_train_error += nn_train(nn, input, target, learning_rate);
    }
    train_error = total_train_error / (float)train_data->num_rows;
    // Check the model against the validation data
    total_validation_err = 0.0f;
    for (j = 0; j < validation_data->num_rows; j++) {
      float *input = validation_data->input[j];
      float *target = validation_data->target[j];
      total_validation_err += nn_error(nn, input, target);
    }
    validation_error = total_validation_err / (float)validation_data->num_rows;
    epochs++;
    printf("%.5f, %.5f, %.5f\n", train_error, validation_error, learning_rate);
    learning_rate *= annealing;
    if (validation_error < best_validation_error) {
      best_validation_error = validation_error;
      epochs_since_improvement = 0;
      // See train_cnn.c's identical block for why only the best-so-far
      // model is persisted, and in what format.
      if (resuming && existing_format == NN_MODEL_FORMAT_INPLACE) {
        nn_save_model_inplace(nn, (char *)model_path);
      } else if (resuming && existing_format == NN_MODEL_FORMAT_BINARY) {
        nn_save_model_binary(nn, (char *)model_path);
      } else if (resuming && existing_format == NN_MODEL_FORMAT_ASCII) {
        nn_save_model_ascii(nn, (char *)model_path);
      } else {
        nn_save_model(nn, (char *)model_path);
      }
    } else {
      epochs_since_improvement++;
      if (epochs_since_improvement >= EARLY_STOPPING_PATIENCE) {
        printf("No validation improvement for %d epochs (best: %.5f) -- stopping early.\n",
               EARLY_STOPPING_PATIENCE, best_validation_error);
        break;
      }
    }
  }
  data_free(validation_data);
  data_free(train_data);
  nn_free(nn);
  printf("Final (last epoch) train error: %f, validation error: %f\n", train_error, validation_error);
  printf("Best validation error (the model saved to disk): %f\n", best_validation_error);
  printf("Training epochs: %d\n", epochs);
  return 0;
}
