/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Trains a fully convolutional network to find red blood cells, white
// blood cells, and platelets in a microscope photo, from the
// train.csv/validation.csv prepare_data.c builds out of the BCCD dataset.
// Same training loop as ../image_classification/train.c (early stopping on
// validation error), but instead of one label per image, the network
// outputs one heatmap per class -- see this example's README for how that
// turns into a detector, and why it suits this library.

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_prep.h"
#include "image_prep.h"
#include "nn.h"

// See ../image_classification/train.c for why early stopping tracks the
// best validation error rather than a fixed target.
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
  int num_inputs = IMG_INPUTS;
  int num_outputs = HEATMAP_OUTPUTS;
  float learning_rate = 0.001f;
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
  // Attempt to load an existing model -- see
  // ../image_classification/train.c's identical block for why the inplace
  // format goes through nn_load_model_inplace_copy().
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
    // Construct the neural network, layer by layer
    nn_add_layer(nn, LAYER_TYPE_INPUT, num_inputs, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    // The first layer is a 5x5 convolution with a stride of 2, instead of
    // a stride-1 convolution followed by 2x2 max pooling (as every conv
    // stage after it, and ../image_classification/train.c's, uses): it
    // goes straight from the 64x48 frame to 32x24x8 without ever holding a
    // full-resolution 64x48x8 map. Not holding that map (or the pooling
    // layer after it) cuts this network's activation RAM by 42%, and
    // measured no less accurate (see this example's README). The 5x5 kernel keeps
    // every input pixel covered despite the stride.
    cnn_t cnn1 = {
      .in_h = IMG_H,
      .in_w = IMG_W,
      .in_channels = IMG_CHANNELS,
      .out_channels = 8,
      .kernel_size = 5,
      .stride = 2,
      .padding = 2,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_GELU, (cnn_t *)&cnn1);
    // One conv+pool stage takes it the rest of the way to the 16x12
    // output grid (GRID_STRIDE 4).
    cnn_t cnn2 = {
      .in_h = IMG_H / 2,
      .in_w = IMG_W / 2,
      .in_channels = 8,
      .out_channels = 16,
      .kernel_size = 3,
      .stride = 1,
      .padding = 1,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_GELU, (cnn_t *)&cnn2);
    pool_t pool2 = {
      .in_h = IMG_H / 2,
      .in_w = IMG_W / 2,
      .channels = 16,
      .pool_size = 2,
      .stride = 2,
      .pooling_type = POOLING_TYPE_MAX,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&pool2);
    // Two more conv layers at the output grid's resolution, with no
    // pooling, so each output cell can still be told apart from its
    // neighbors. They widen what each output cell can see to 27x27 input
    // pixels -- a WBC is ~20 across -- without another pooling stage
    // coarsening the grid.
    cnn_t cnn3 = {
      .in_h = GRID_H,
      .in_w = GRID_W,
      .in_channels = 16,
      .out_channels = 32,
      .kernel_size = 3,
      .stride = 1,
      .padding = 1,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_GELU, (cnn_t *)&cnn3);
    cnn_t cnn4 = {
      .in_h = GRID_H,
      .in_w = GRID_W,
      .in_channels = 32,
      .out_channels = 32,
      .kernel_size = 3,
      .stride = 1,
      .padding = 1,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_GELU, (cnn_t *)&cnn4);
    // The detection head: a 1x1 convolution -- the same small linear
    // classifier applied independently at every grid cell -- with one
    // sigmoid output channel per class. Its output, CLASS_COUNT x GRID_H x
    // GRID_W in CHW order, is exactly the heatmap layout prepare_data.c
    // writes as targets. Being the network's last layer is all it takes to
    // make it the output: nn_train() computes the loss on whatever the
    // final layer is, with no special LAYER_TYPE_OUTPUT needed, and a
    // non-softmax output trains with MSE.
    //
    // A LAYER_TYPE_OUTPUT (fully-connected) layer in its place would
    // connect every one of the 6,144 features above to every one of the 576
    // outputs: 3.5 million weights, versus 99 here, and no built-in notion
    // that a cell's output should depend mostly on what's at that cell.
    cnn_t head = {
      .in_h = GRID_H,
      .in_w = GRID_W,
      .in_channels = 32,
      .out_channels = CLASS_COUNT,
      .kernel_size = 1,
      .stride = 1,
      .padding = 0,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_SIGMOID, (cnn_t *)&head);
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
  // See ../image_classification/train.c for why a resumed model's starting
  // validation error is the baseline to beat.
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
    fflush(stdout);
    learning_rate *= annealing;
    if (validation_error < best_validation_error) {
      best_validation_error = validation_error;
      epochs_since_improvement = 0;
      // Save back in the same format a resumed model was loaded from --
      // see ../image_classification/train.c.
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
