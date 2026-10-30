/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Trains a GRU layer to classify one SEQUENCE_LEN-frame log-mel feature
// sequence -- one whole ~1-second utterance -- as the wake word or not,
// against train.csv/validation.csv (see prepare_data.c/README.md for how
// those are built from the real Speech Commands dataset). This is the
// example's recurrent baseline; train_cnn.c is its main model, and the
// README compares the two. GRU was chosen as the recurrent layer after
// comparing it against a plain RNN and an LSTM on this task, where it won
// clearly. The tuning numbers quoted in this file's comments were all
// measured on this example's earlier random per-clip split, with a
// balanced test set (see git history), not on today's speaker split.
//
// Every frame of one clip here shares the SAME label -- since the whole
// clip either is or isn't the wake word -- so this follows
// examples/gesture_recognition/train.c's "many-to-many, same label
// repeated every timestep" scheme rather than examples/fall_detection's
// per-timestep-varying one.

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

// Width of one flattened CSV row (for data_load() below) -- NOT the
// network's own INPUT layer width. nn_train()/nn_predict()/nn_error()
// alias nn->neuron[0] directly onto whatever pointer they're given (see
// nn.c: `nn->neuron[0] = inputs;`, no copy) and every layer above it reads
// exactly width[0] elements from it, so the INPUT layer has to be
// NUM_MEL_BINS wide (one frame) to match what run_clip() passes per call --
// see nn_add_layer(LAYER_TYPE_INPUT, ...) below.
#define NUM_INPUTS (SEQUENCE_LEN * NUM_MEL_BINS)

// Runs one whole clip's feature sequence through the network, frame by
// frame, training (or, when `train` is false, just measuring error via
// nn_error()) against the SAME target at every frame -- see this file's top
// comment. `input` is one data_prep.[ch] row: SEQUENCE_LEN * NUM_MEL_BINS
// floats, laid out frame-major exactly as prepare_data.c wrote them, so
// `input + t * NUM_MEL_BINS` is frame t's NUM_MEL_BINS-wide feature vector.
// Always starts by resetting the GRU's hidden state (nn_reset_state()):
// each clip is an independent utterance, and the previous clip's final
// hidden state must not leak into it.
static float run_clip(nn_t *nn, float *input, float *target, float rate, bool train)
{
  nn_reset_state(nn);
  float total_err = 0.0f;
  for (int t = 0; t < SEQUENCE_LEN; t++) {
    float *frame = input + t * NUM_MEL_BINS;
    total_err += train ? nn_train(nn, frame, target, rate) : nn_error(nn, frame, target);
  }
  return total_err / (float)SEQUENCE_LEN;
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
    printf("  <model-file> : Path to the neural net model to load or create (e.g., wake_word_model.txt or .bin)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Tunable hyperparameters
  int num_inputs = NUM_INPUTS;
  int num_outputs = 1; // wake-word probability
  int hidden_units = 32;
  // Real audio's noisier gradient made plain SGD at the other RNN
  // examples' 0.05 bounce (validation error moving up and down epoch to
  // epoch) rather than settle. A small sweep (0.003-0.03, two runs each,
  // since random init/shuffling isn't seeded and single-run results on
  // this real dataset vary by several accuracy points) found 0.03 both the
  // highest-scoring and the most consistent across runs (~88% test
  // accuracy both times, vs. lower and more erratic results at 0.01-0.02)
  // -- on the earlier split; see this file's top comment. NN_OPTIMIZER_ADAM was
  // also tried (on a plain RNN, before this example was simplified to GRU
  // only) and made things worse; plain SGD stays the default.
  float learning_rate = 0.03f;
  float annealing = 0.97f;
  // End of tunable parameters
  data_t *train_data;
  data_t *validation_data;
  nn_t *nn;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  srand((unsigned)time(NULL));

  train_data = data_load("train.csv", num_inputs, num_outputs);
  if (train_data == NULL) {
    printf("Error: Could not load training data (train.csv). Run `make` first to build it from the dataset.\n");
    return 1;
  }
  validation_data = data_load("validation.csv", num_inputs, num_outputs);
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
    // NUM_MEL_BINS, not num_inputs (see NUM_INPUTS's comment above) -- one
    // frame per nn_train()/nn_predict() call, not the whole flattened row.
    nn_add_layer(nn, LAYER_TYPE_INPUT, NUM_MEL_BINS, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    // GRU's three gates (reset, update, candidate) each have a fixed
    // nonlinearity -- the activation argument is stored but unused (see
    // LAYER_TYPE_GRU's comment in nn.h) -- a hidden state that should hold
    // a signal across frames without decaying, which is what won this
    // architecture the comparison against a plain RNN and an LSTM (see
    // this file's top comment).
    nn_add_layer(nn, LAYER_TYPE_GRU, hidden_units, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    // LAYER_TYPE_DROPOUT (see examples/character_recognition/train_cnn.c's
    // identical layer for the FC case) was tried here, between the GRU and
    // the output layer, on the theory that training error dropping to
    // near-zero while validation error stayed noisy meant the GRU (nearly
    // all of this network's ~5,121 parameters) was overfitting. Measured
    // across several runs at two rates, it made things worse, not better
    // (0.3: 82-87% test accuracy, avg ~85%; 0.1: 83-86%, avg ~84%; vs. no
    // dropout's 87-89%, avg ~88%, on the earlier split).
    //
    // The first suspect was that nn_dropout_forward() (nn.c) resampled its
    // mask fresh on every call, injecting a different random mask into the
    // GRU's hidden state on every one of a clip's 49 per-timestep frames
    // instead of holding one consistent mask for the whole sequence (the
    // "locked"/variational-dropout technique effective RNN dropout needs).
    // That was real and has since been fixed at the library level: a
    // DROPOUT layer in any network with an RNN/GRU/LSTM layer now holds its
    // mask until nn_reset_state() marks a new sequence (see
    // dropout_mask_valid's comment in nn.h) -- auto-detected, no config
    // needed here. Retested with that fix in place, though, dropout at 0.3
    // still scored 82-83% here, no better than before the fix. So the
    // per-timestep resampling bug was worth fixing on its own correctness
    // merits, but it wasn't (or wasn't the whole) explanation for dropout
    // underperforming on this specific model/dataset -- left out here
    // either way. See git history for the exact layer this tried.
    // Single sigmoid output: wake-word probability, trained with ordinary
    // sigmoid+MSE -- there's only one class here, not several mutually
    // exclusive ones, so no softmax.
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, num_outputs, ACTIVATION_FUNCTION_TYPE_SIGMOID, NULL);
  } else {
    printf("Using existing model file: %s\n", model_path);
    if ((nn->width[0] != (uint32_t)NUM_MEL_BINS) ||
        (nn->width[nn->depth - 1] != (uint32_t)num_outputs)) {
      printf("Error: Loaded model dimensions do not match expected: %d %d.\n", NUM_MEL_BINS, num_outputs);
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
  // Report clip-level accuracy at a 0.5 threshold on the final frame's
  // prediction (the network has seen the whole ~1-second utterance by
  // then), plus the same false-accept/false-reject split test.c reports.
  int correct = 0, false_accepts = 0, false_rejects = 0, positives = 0, negatives = 0;
  for (int i = 0; i < validation_data->num_rows; i++) {
    nn_reset_state(nn);
    float *pred = NULL;
    for (int t = 0; t < SEQUENCE_LEN; t++)
      pred = nn_predict(nn, validation_data->input[i] + t * NUM_MEL_BINS);
    bool predicted_wake_word = pred[0] >= 0.5f;
    bool actual_wake_word = validation_data->target[i][0] >= 0.5f;
    if (predicted_wake_word == actual_wake_word)
      correct++;
    if (actual_wake_word) {
      positives++;
      if (!predicted_wake_word)
        false_rejects++;
    } else {
      negatives++;
      if (predicted_wake_word)
        false_accepts++;
    }
  }
  data_free(validation_data);
  data_free(train_data);
  printf("Final (last epoch) train error: %f, validation error: %f\n", train_error, validation_error);
  printf("Best validation error (the model saved to disk): %f\n", best_validation_error);
  printf("Training epochs: %d\n", epochs);
  printf("Validation accuracy: %d/%d = %.2f%%\n", correct, positives + negatives,
         100.0f * (float)correct / (float)(positives + negatives));
  if (positives > 0)
    printf("False rejects (missed wake word): %d/%d = %.1f%%\n", false_rejects, positives,
           100.0f * (float)false_rejects / (float)positives);
  if (negatives > 0)
    printf("False accepts (wrongly triggered): %d/%d = %.1f%%\n", false_accepts, negatives,
           100.0f * (float)false_accepts / (float)negatives);
  nn_free(nn);
  return 0;
}
