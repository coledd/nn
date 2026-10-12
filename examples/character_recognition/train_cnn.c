/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Trains a CNN (convolution + pooling ahead of the fully-connected layers)
// on the MNIST handwritten-digit task. See train_fc.c in this same
// directory for a plain fully-connected network on the identical task/data
// -- the accuracy difference between the two is the point of having both.

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "data_prep.h"
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
  int num_inputs = 28 * 28;
  int num_outputs = 10;
  // Softmax + cross-entropy's output-layer gradient is the raw
  // target-prediction (see ACTIVATION_FUNCTION_TYPE_SOFTMAX in nn.h) with no
  // sigmoid-derivative factor (<=0.25x) damping it the way the old
  // sigmoid+MSE output did -- at the old rate (0.025), that up-to-~4x
  // larger effective step reliably drove every unit of the second FC layer
  // negative within the first epoch. Once every unit in a ReLU layer is
  // simultaneously dead, its derivative is 0 everywhere, permanently
  // blocking all gradient flow to everything before it -- confirmed via a
  // diagnostic run showing that layer frozen at 100% dead output from
  // epoch 0 onward, collapsing training to predicting the class-marginal
  // (log(10) =~ 2.3, the uniform-over-10-classes cross-entropy, which is
  // exactly the plateau this used to get stuck at).
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
    cnn_t cnn = {
      .in_h = 28,
      .in_w = 28,
      .in_channels = 1,
      .out_channels = 8,
      .kernel_size = 5,
      .stride = 1,
      .padding = 2,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (cnn_t *)&cnn);
    // "Same" padding ((28 + 2*2 - 5)/1 + 1 == 28) keeps the CNN output at
    // the full 28x28 instead of shrinking it to 24x24, so the outer couple
    // of pixels of the digit -- where strokes commonly start/end -- still
    // get seen by every kernel position instead of being cropped away
    // before the network looks at them at all. Pool it down 2x2 -> 8x14x14
    // before the first fully-connected layer (vs. 8x12x12 without padding),
    // improving translation invariance.
    pool_t pool = {
      .in_h = 28,
      .in_w = 28,
      .channels = 8,
      .pool_size = 2,
      .stride = 2,
      .pooling_type = POOLING_TYPE_MAX,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&pool);
    // A second conv+pool stage, rather than widening the first one: a
    // wider first stage (e.g. 16 filters instead of 8) would double the
    // flattened width feeding the first FC layer below (3136 instead of
    // 1568), since pooling only shrinks the spatial dims, not the channel
    // count -- and that FC layer already dominates this network's
    // parameter count (see this file's/this example's README's comment on
    // that), so widening the first stage grows the network in exactly the
    // wrong place for comparatively little accuracy (measured: ~191K ->
    // ~379K params for +0.1-0.6 points of test accuracy). A second,
    // smaller-channel-count-relative-to-spatial-size stage instead
    // *shrinks* what reaches the FC layer -- 16 channels over the now-14x14
    // input pools down to 7x7x16 = 784, half of the single-stage
    // network's 1568 -- while still extracting a genuine second level of
    // features (edges-of-edges, not just edges) over the first stage's
    // 8x14x14 output, the same reasoning any deeper CNN stacks conv stages
    // for.
    cnn_t cnn2 = {
      .in_h = 14,
      .in_w = 14,
      .in_channels = 8,
      .out_channels = 16,
      .kernel_size = 5,
      .stride = 1,
      .padding = 2,
      .dilation = 1,
      .weight_init = NN_INIT_XAVIER,
      .bias_init = NN_INIT_ZEROS,
    };
    nn_add_layer(nn, LAYER_TYPE_CNN, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (cnn_t *)&cnn2);
    // Same "same" padding and 2x2/stride-2 max pool as the first stage,
    // just at this stage's 14x14x16 shape: (14 + 2*2 - 5)/1 + 1 == 14, then
    // pooled to 7x7x16.
    pool_t pool2 = {
      .in_h = 14,
      .in_w = 14,
      .channels = 16,
      .pool_size = 2,
      .stride = 2,
      .pooling_type = POOLING_TYPE_MAX,
    };
    nn_add_layer(nn, LAYER_TYPE_POOL, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (pool_t *)&pool2);
    // GELU instead of RELU: smooth and non-monotonic (dips slightly negative
    // before rising) rather than flat-zero for any negative preact, so a
    // unit that drifts negative still has a small, nonzero gradient and can
    // recover instead of dying forever -- directly hardens these layers
    // against the exact permanently-dead-ReLU collapse diagnosed earlier
    // (see the learning-rate comment above), on top of whatever accuracy
    // difference it makes.
    nn_add_layer(nn, LAYER_TYPE_FC, 120, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    // Regularization: the first FC layer (1568*120 weights) has by far the
    // most capacity in this network, and is the one most free to overfit --
    // randomly zeroing 30% of its outputs during training (a no-op at
    // inference, see LAYER_TYPE_DROPOUT's comment in nn.h) keeps the second
    // FC layer from co-adapting to any one of them too tightly.
    dropout_t dropout = { .rate = 0.3f };
    nn_add_layer(nn, LAYER_TYPE_DROPOUT, 0, ACTIVATION_FUNCTION_TYPE_LINEAR, (dropout_t *)&dropout);
    nn_add_layer(nn, LAYER_TYPE_FC, 20, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    // Softmax + cross-entropy (the loss nn_train()/nn_error() switch to
    // automatically for a softmax output -- see ACTIVATION_FUNCTION_TYPE_SOFTMAX
    // in nn.h) models these 10 digits as mutually exclusive classes, unlike
    // the 10 independent per-class sigmoids this used before.
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
