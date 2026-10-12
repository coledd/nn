/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Trains a CNN to tell a cat from a dog in a cropped photo, from the
// train.csv/validation.csv prepare_data.c builds out of the Kaggle Cats and
// Dogs and Oxford-IIIT Pet datasets. Same training loop as ../character_recognition/train_cnn.c
// (softmax output, early stopping on validation error), with conv+pool
// stages scaled to this task's 48x48 color input and a global-average-
// pooling head instead of fully-connected layers -- see this example's
// README for how this task differs from MNIST and what that does to
// accuracy.

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_prep.h"
#include "image_prep.h"
#include "nn.h"

// A fixed validation-error target doesn't mean much across different loss
// functions/architectures (e.g. cross-entropy's scale isn't MSE's), and
// guessing a new magic number every time one of those changes is fragile.
// Early stopping instead tracks the best validation error seen and stops
// once it hasn't improved for this many epochs in a row, saving only the
// best-so-far model (not necessarily the last epoch's, which may already be
// more overfit) -- self-adjusting to whatever the loss actually is.
#define EARLY_STOPPING_PATIENCE 5
// Hard safety cap so training can't run forever even if validation error
// somehow keeps eking out tiny improvements indefinitely.
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
  int num_outputs = CLASS_COUNT;
  // 5x smaller than ../character_recognition/train_cnn.c's 0.005. In an
  // earlier version of this example trained on Oxford-IIIT Pet alone (see
  // this example's README), 0.005 scored the same within run-to-run noise
  // (74.7% vs. 73.9% test accuracy, 2 and 5 runs); 0.001 is kept because
  // it takes smaller steps per epoch, and with every training image
  // augmented into several rows, one epoch is already more than one pass
  // over the same photos.
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
  // Attempt to load an existing model. nn_load_model() already auto-detects
  // ascii vs. binary, but the inplace format needs a mutable model to keep
  // training (nn_train() refuses to run on the read-only model
  // nn_load_model_inplace() itself would produce -- its weight arrays are
  // aliased into the input buffer, not owned), so use
  // nn_load_model_inplace_copy() instead when nn_model_format() reports
  // that's what the file is; that copies the weights into normal owned
  // allocations, so the buffer can be freed right after loading. As with
  // the ascii/binary case below, any load failure here (missing file,
  // corrupt file, whatever) just falls through to creating a new model,
  // rather than being treated as a distinct hard error.
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
    // Four conv+pool stages, two more than
    // ../character_recognition/train_cnn.c's two: a 48x48 crop of a pet
    // on a couch or a lawn is a lot less clean than a centered digit on
    // a blank background, and every extra stage extracts a further level of
    // features. Channels double as the spatial size halves: 48x48x8 ->
    // 24x24x16 -> 12x12x32 -> 6x6x64, pooled to 3x3x64.
    //
    // Every conv layer uses GELU. Without an activation, a conv layer is
    // linear, leaving max pooling as the only nonlinearity between stages
    // (../character_recognition/train_cnn.c still works that way). Adding
    // GELU was the single biggest improvement measured here -- about +3
    // points of test accuracy at no cost in parameters, RAM, or compute --
    // and it's what made the extra fourth stage and the 48x48 input pay off
    // (neither helped without it). See this example's README for the
    // numbers.
    cnn_t cnn1 = {
      .in_h = IMG_SIZE,
      .in_w = IMG_SIZE,
      .in_channels = IMG_CHANNELS,
      .out_channels = 8,
      .kernel_size = 5,
      .stride = 1,
      .padding = 2,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_GELU, (cnn_t *)&cnn1);
    pool_t pool1 = {
      .in_h = IMG_SIZE,
      .in_w = IMG_SIZE,
      .channels = 8,
      .pool_size = 2,
      .stride = 2,
      .pooling_type = POOLING_TYPE_MAX,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&pool1);
    cnn_t cnn2 = {
      .in_h = IMG_SIZE / 2,
      .in_w = IMG_SIZE / 2,
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
      .in_h = IMG_SIZE / 2,
      .in_w = IMG_SIZE / 2,
      .channels = 16,
      .pool_size = 2,
      .stride = 2,
      .pooling_type = POOLING_TYPE_MAX,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&pool2);
    cnn_t cnn3 = {
      .in_h = IMG_SIZE / 4,
      .in_w = IMG_SIZE / 4,
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
    pool_t pool3 = {
      .in_h = IMG_SIZE / 4,
      .in_w = IMG_SIZE / 4,
      .channels = 32,
      .pool_size = 2,
      .stride = 2,
      .pooling_type = POOLING_TYPE_MAX,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&pool3);
    cnn_t cnn4 = {
      .in_h = IMG_SIZE / 8,
      .in_w = IMG_SIZE / 8,
      .in_channels = 32,
      .out_channels = 64,
      .kernel_size = 3,
      .stride = 1,
      .padding = 1,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_GELU, (cnn_t *)&cnn4);
    pool_t pool4 = {
      .in_h = IMG_SIZE / 8,
      .in_w = IMG_SIZE / 8,
      .channels = 64,
      .pool_size = 2,
      .stride = 2,
      .pooling_type = POOLING_TYPE_MAX,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&pool4);
    // Global average pooling: averages each of the 64 3x3 feature maps down
    // to a single value, so the output layer sees 64 inputs -- "how much of
    // each learned feature is anywhere in the crop" -- rather than a
    // flattened 576-wide map. A flatten -> 64-unit GELU FC -> 30% dropout
    // head in its place (as ../character_recognition/train_cnn.c uses) was
    // measured, when this example trained on Oxford-IIIT Pet alone, at the
    // same accuracy within run-to-run noise with 6x the parameters. With no
    // FC layer left to overfit, this head also needs no dropout.
    pool_t gap = {
      .in_h = IMG_SIZE / 16,
      .in_w = IMG_SIZE / 16,
      .channels = 64,
      .pool_size = IMG_SIZE / 16,
      .stride = IMG_SIZE / 16,
      .pooling_type = POOLING_TYPE_AVG,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&gap);
    // Two-way softmax rather than a single sigmoid output, so test.c/
    // predict.c read it the same way as every other multi-class example
    // here -- and so adding a third class (fox, bobcat, ...) later is a
    // CLASS_COUNT change, not a different output layer.
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
  // Establish a validation-error baseline before training: for a resumed
  // model, this is its current validation error, so early stopping (and
  // "only save on improvement" below) can't immediately overwrite an
  // already-good saved model with a worse one just because epoch 1 of this
  // run happens to land above where the previous run left off. For a brand
  // new model, there's nothing to beat yet, so +infinity makes the very
  // first epoch always count as an improvement.
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
      // Only persist the best-so-far model, not every epoch's -- a later
      // epoch may already be more overfit (see the training-log discussion
      // that motivated early stopping) despite training error still
      // falling. When resuming an existing model, save back in the same
      // format it was loaded from (ascii/binary/inplace) rather than
      // picking by file extension, so continuing to train an inplace model
      // doesn't silently convert it to a different format; a brand-new
      // model has no existing format to preserve, so it keeps the
      // extension-based default (nn_save_model()).
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
