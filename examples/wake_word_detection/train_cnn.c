/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Trains a small CNN to classify one SEQUENCE_LEN-frame log-mel feature
// sequence -- one whole ~1-second utterance -- as the wake word or not,
// against train.csv/validation.csv (see prepare_data.c/README.md for how
// those are built from the real Speech Commands dataset).
//
// Unlike train_gru.c, which reads the clip one frame per call, this reads
// the whole clip at once as a SEQUENCE_LEN x NUM_MEL_BINS image (time down,
// mel bands across), one channel -- prepare_data.c's frame-major row layout
// is exactly that image in row-major order, so the same CSV rows feed both.
// Convolution then learns local time-frequency shapes (a formant's onset, a
// voiced syllable's band pattern) wherever in the second they occur, which
// is how most small keyword-spotting models work.

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_prep.h"
#include "nn.h"
#include "wake_word_data.h"

// See examples/character_recognition/train_cnn.c's identical constants for the early-stopping rationale.
#define EARLY_STOPPING_PATIENCE 5
#define MAX_EPOCHS 500

// One flattened CSV row, and this network's INPUT layer width: the whole clip.
#define NUM_INPUTS (SEQUENCE_LEN * NUM_MEL_BINS)
// Two softmax outputs (other, wake word) rather than one sigmoid: cross-
// entropy instead of MSE (see ACTIVATION_FUNCTION_TYPE_SOFTMAX in nn.h).
// The CSV still carries one 0/1 label; one_hot() expands it.
#define NUM_OUTPUTS 2

static void one_hot(const float *label, float *target)
{
  target[1] = label[0] >= 0.5f ? 1.0f : 0.0f;
  target[0] = 1.0f - target[1];
}

static float run_clip(nn_t *nn, float *input, const float *label, float rate, bool train)
{
  float target[NUM_OUTPUTS];
  one_hot(label, target);
  return train ? nn_train(nn, input, target, rate) : nn_error(nn, input, target);
}

// Adds one "same"-padded 3x3 convolution with GELU, then a 2x2/stride-2 max
// pool, to `nn`, reading an in_h x in_w x in_channels map. Updates
// *in_h/*in_w to the pooled output's size.
static void add_conv_pool(nn_t *nn, int *in_h, int *in_w, int in_channels, int out_channels)
{
  cnn_t cnn = {
    .in_h = (uint16_t)*in_h,
    .in_w = (uint16_t)*in_w,
    .in_channels = (uint8_t)in_channels,
    .out_channels = (uint8_t)out_channels,
    .kernel_size = 3,
    .stride = 1,
    .padding = 1,
    .dilation = 1,
    .weight_init = NN_INIT_HE,
    .bias_init = NN_INIT_ZEROS,
  };
  nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_GELU, &cnn);
  pool_t pool = {
    .in_h = (uint16_t)*in_h,
    .in_w = (uint16_t)*in_w,
    .channels = (uint8_t)out_channels,
    .pool_size = 2,
    .stride = 2,
    .pooling_type = POOLING_TYPE_MAX,
  };
  nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, &pool);
  *in_h = (*in_h - 2) / 2 + 1;
  *in_w = (*in_w - 2) / 2 + 1;
}

// Validation error with each class weighted equally: the mean of the
// wake-word clips' average error and everything else's. validation.csv
// holds ~50 negatives per wake-word clip (see prepare_data.c), so a plain
// average would be almost entirely about false accepts, and early stopping
// would favor a model that rarely fires at all.
static float validation_error_balanced(nn_t *nn, data_t *data)
{
  float total[2] = { 0.0f, 0.0f };
  int count[2] = { 0, 0 };
  for (int i = 0; i < data->num_rows; i++) {
    int c = data->target[i][0] >= 0.5f;
    total[c] += run_clip(nn, data->input[i], data->target[i], 0.0f, false);
    count[c]++;
  }
  return 0.5f * (total[0] / (float)(count[0] > 0 ? count[0] : 1) + total[1] / (float)(count[1] > 0 ? count[1] : 1));
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to the neural net model to load or create (e.g., wake_word_cnn.txt or .bin)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Tunable hyperparameters
  float learning_rate = 0.002f;
  float annealing = 0.95f;
  // End of tunable parameters
  data_t *train_data;
  data_t *validation_data;
  nn_t *nn;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  srand((unsigned)time(NULL));

  train_data = data_load("train.csv", NUM_INPUTS, 1);
  if (train_data == NULL) {
    printf("Error: Could not load training data (train.csv). Run `make` first to build it from the dataset.\n");
    return 1;
  }
  validation_data = data_load("validation.csv", NUM_INPUTS, 1);
  if (validation_data == NULL) {
    printf("Error: Could not load validation data (validation.csv).\n");
    data_free(train_data);
    return 1;
  }

  // Attempt to load an existing model; see examples/character_recognition/
  // train_cnn.c's identical block for why the inplace format needs
  // nn_load_model_inplace_copy() instead of plain nn_load_model() to keep
  // training.
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
    nn_add_layer(nn, LAYER_TYPE_INPUT, NUM_INPUTS, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    // Three conv+pool stages: 49x20x1 -> 24x10x16 -> 12x5x32 -> 6x2x32.
    // Each pool halves the time axis, so the last stage's 3x3 kernels span
    // about a quarter of a second -- most of a syllable.
    int h = SEQUENCE_LEN, w = NUM_MEL_BINS;
    add_conv_pool(nn, &h, &w, 1, 16);
    add_conv_pool(nn, &h, &w, 16, 32);
    add_conv_pool(nn, &h, &w, 32, 32);
    dropout_t dropout = { .rate = 0.2f };
    nn_add_layer(nn, LAYER_TYPE_DROPOUT, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, &dropout);
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, NUM_OUTPUTS, ACTIVATION_FUNCTION_TYPE_SOFTMAX, NULL);
  } else {
    printf("Using existing model file: %s\n", model_path);
    if ((nn->width[0] != (uint32_t)NUM_INPUTS) ||
        (nn->width[nn->depth - 1] != (uint32_t)NUM_OUTPUTS)) {
      printf("Error: Loaded model dimensions do not match expected: %d %d.\n", NUM_INPUTS, NUM_OUTPUTS);
      nn_free(nn);
      data_free(train_data);
      data_free(validation_data);
      return 1;
    }
  }
  // See examples/character_recognition/train_cnn.c's identical block for
  // why this baseline matters when resuming an existing model.
  float best_validation_error = FLT_MAX;
  if (resuming) {
    best_validation_error = validation_error_balanced(nn, validation_data);
    printf("Starting validation error (existing model): %.5f\n", best_validation_error);
  }
  int epochs_since_improvement = 0;
  printf("train error, validation error, learning rate\n");
  while (epochs < MAX_EPOCHS) {
    data_shuffle(train_data);
    float total_train_error = 0.0f;
    for (int j = 0; j < train_data->num_rows; j++)
      total_train_error += run_clip(nn, train_data->input[j], train_data->target[j], learning_rate, true);
    train_error = total_train_error / (float)train_data->num_rows;
    validation_error = validation_error_balanced(nn, validation_data);
    epochs++;
    printf("%.5f, %.5f, %.5f\n", train_error, validation_error, learning_rate);
    fflush(stdout);
    learning_rate *= annealing;
    if (validation_error < best_validation_error) {
      best_validation_error = validation_error;
      epochs_since_improvement = 0;
      // Save back in the same format the model was loaded from when
      // resuming, same rationale as examples/character_recognition/
      // train_cnn.c's identical block.
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
  printf("Final (last epoch) train error: %f, validation error: %f\n", train_error, validation_error);
  printf("Best validation error (the model saved to disk): %f\n", best_validation_error);
  printf("Training epochs: %d\n", epochs);
  nn_free(nn);
  return 0;
}
