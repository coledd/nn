/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Example: training a GRU layer to continuously monitor a simulated 3-axis
// accelerometer stream and flag a fall as soon as one occurs -- the GRU
// counterpart to train_rnn.c/train_lstm.c in this same directory, on the
// identical task and data, so the three can be compared directly. A fall
// event is a multi-phase pattern (a brief free-fall dip, an impact spike,
// then a long stretch of unusual post-fall stillness) spread across a much
// longer sequence than examples/gesture_recognition's fixed windows, with
// the label VARYING per timestep -- 0 throughout ordinary activity,
// flipping to 1 at the instant a fall begins and staying 1 for the rest of
// the monitoring window.
//
// Measured against train_rnn.c/train_lstm.c on this exact task: all three
// eventually reach the same ceiling (100% per-timestep accuracy, every
// fall detected, zero false alarms), but train_rnn.c consistently takes
// several times as many epochs to get there (60-200+, across several
// runs, vs. well under 50 here) -- gating helps it find that solution
// faster, not just reach a solution the plain hidden state genuinely
// couldn't. GRU gets that gated advantage with only three gates and one
// persistent state instead of LSTM's four gates and two states (see
// LAYER_TYPE_GRU's comment in nn.h), so it's cheaper to run and train --
// the tradeoff that makes GRU worth having alongside LSTM on a
// microcontroller budget: most of the training-efficiency benefit of
// gating, at a fraction of LSTM's per-timestep cost. Every sequence is
// synthesized on the fly (see fall_data.[ch]) -- there is nothing to
// download.

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fall_data.h"
#include "nn.h"

// See examples/character_recognition/train_cnn.c's identical constants for the early-stopping rationale.
#define EARLY_STOPPING_PATIENCE 5
#define MAX_EPOCHS 500

#define SEQUENCES_PER_EPOCH 100          // freshly-synthesized training sequences per epoch
#define VALIDATION_SEQUENCES_PER_CLASS 10 // fixed, generated once, for a stable early-stopping signal

// Prints one freshly-generated example sequence of each kind (normal
// activity, and a fall) to stdout, condensed to acceleration magnitude and
// label every 3rd timestep -- enough to see the free-fall dip, impact
// spike, and post-fall stillness in the fall example, and confirm the
// normal-activity example never looks like one. See
// examples/gesture_recognition/train.c's print_sample_gestures() for the same rationale.
static void print_sample_sequences(void)
{
  static float window[SEQUENCE_LEN][NUM_AXES];
  static float label[SEQUENCE_LEN];
  const char *names[2] = {"NORMAL ACTIVITY", "FALL"};
  for (int kind = 0; kind < 2; kind++) {
    if (kind == 0)
      generate_normal_sequence(window, label);
    else
      generate_fall_sequence(window, label);
    printf("\nSample %s sequence (|accel| in g, every 3rd timestep):\n", names[kind]);
    for (int t = 0; t < SEQUENCE_LEN; t += 3) {
      float mag = sqrtf(window[t][0] * window[t][0] + window[t][1] * window[t][1] + window[t][2] * window[t][2]);
      printf("  t=%3d: |a|=%.2f  label=%.0f\n", t, mag, label[t]);
    }
  }
  printf("\n");
}

// Runs one whole sequence through the network, timestep by timestep,
// training (or, when `train` is false, just measuring error via
// nn_error()) against THAT timestep's own label -- unlike
// examples/gesture_recognition/train.c's run_window(), which repeats one label for the whole
// window, every timestep here can have a different target. Always starts
// by resetting the GRU's hidden state (nn_reset_state()): each
// sequence is an independent monitoring window, and the previous one's
// final state must not leak into it (see nn_reset_state()'s comment in
// nn.h). Returns the average per-timestep error for this sequence.
static float run_sequence(nn_t *nn, float window[SEQUENCE_LEN][NUM_AXES], float label[SEQUENCE_LEN], float rate, bool train)
{
  nn_reset_state(nn);
  float total_err = 0.0f;
  for (int t = 0; t < SEQUENCE_LEN; t++) {
    float target[1] = {label[t]};
    total_err += train ? nn_train(nn, window[t], target, rate) : nn_error(nn, window[t], target);
  }
  return total_err / (float)SEQUENCE_LEN;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to the neural net model to load or create (e.g., fall_model.txt or .bin)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Tunable hyperparameters
  int num_inputs = NUM_AXES;
  int num_outputs = 1; // fall probability at this timestep
  int hidden_units = 16;
  float learning_rate = 0.05f;
  float annealing = 1.0f;
  // End of tunable parameters
  nn_t *nn;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  srand((unsigned)time(NULL));
  print_sample_sequences();

  // A fixed validation set (half normal-activity, half fall sequences),
  // generated once -- unlike the training sequences below, which are
  // freshly synthesized every epoch, as if streaming straight off a live
  // sensor -- so early stopping has a stable signal to compare against
  // from one epoch to the next.
  static float val_window[2 * VALIDATION_SEQUENCES_PER_CLASS][SEQUENCE_LEN][NUM_AXES];
  static float val_label[2 * VALIDATION_SEQUENCES_PER_CLASS][SEQUENCE_LEN];
  static bool val_is_fall[2 * VALIDATION_SEQUENCES_PER_CLASS];
  int num_validation = 0;
  for (int k = 0; k < VALIDATION_SEQUENCES_PER_CLASS; k++) {
    generate_normal_sequence(val_window[num_validation], val_label[num_validation]);
    val_is_fall[num_validation] = false;
    num_validation++;
  }
  for (int k = 0; k < VALIDATION_SEQUENCES_PER_CLASS; k++) {
    generate_fall_sequence(val_window[num_validation], val_label[num_validation]);
    val_is_fall[num_validation] = true;
    num_validation++;
  }

  // Attempt to load an existing model; see examples/character_recognition/train_cnn.c's identical block for
  // why the inplace format needs nn_load_model_inplace_copy() instead of
  // plain nn_load_model() to keep training.
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
      return 1;
    }
    nn_add_layer(nn, LAYER_TYPE_INPUT, num_inputs, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    // GRU (not RNN or LSTM): see this file's top comment for why this is
    // the cheaper middle ground between the two.
    nn_add_layer(nn, LAYER_TYPE_GRU, hidden_units, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    // Single sigmoid output: fall probability at this timestep, trained
    // with ordinary sigmoid+MSE (not softmax -- there's only one class here,
    // not several mutually-exclusive ones).
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, num_outputs, ACTIVATION_FUNCTION_TYPE_SIGMOID, NULL);
  } else {
    printf("Using existing model file: %s\n", model_path);
    if ((nn->width[0] != (uint32_t)num_inputs) ||
        (nn->width[nn->depth - 1] != (uint32_t)num_outputs)) {
      printf("Error: Loaded model dimensions do not match expected: %d %d.\n", num_inputs, num_outputs);
      nn_free(nn);
      return 1;
    }
  }
  // See examples/character_recognition/train_cnn.c's identical block for why this baseline matters when
  // resuming an existing model.
  float best_validation_error = FLT_MAX;
  if (resuming) {
    float total = 0.0f;
    for (int i = 0; i < num_validation; i++)
      total += run_sequence(nn, val_window[i], val_label[i], 0.0f, false);
    best_validation_error = total / (float)num_validation;
    printf("Starting validation error (existing model): %.5f\n", best_validation_error);
  }
  int epochs_since_improvement = 0;
  static float window[SEQUENCE_LEN][NUM_AXES];
  static float label[SEQUENCE_LEN];
  printf("train error, validation error, learning rate\n");
  while (epochs < MAX_EPOCHS) {
    float total_train_error = 0.0f;
    for (int j = 0; j < SEQUENCES_PER_EPOCH; j++) {
      if (rand() % 2)
        generate_fall_sequence(window, label);
      else
        generate_normal_sequence(window, label);
      total_train_error += run_sequence(nn, window, label, learning_rate, true);
    }
    train_error = total_train_error / (float)SEQUENCES_PER_EPOCH;
    float total_validation_err = 0.0f;
    for (int i = 0; i < num_validation; i++)
      total_validation_err += run_sequence(nn, val_window[i], val_label[i], 0.0f, false);
    validation_error = total_validation_err / (float)num_validation;
    epochs++;
    printf("%.5f, %.5f, %.5f\n", train_error, validation_error, learning_rate);
    learning_rate *= annealing;
    if (validation_error < best_validation_error) {
      best_validation_error = validation_error;
      epochs_since_improvement = 0;
      // Save back in the same format the model was loaded from when
      // resuming, same rationale as examples/character_recognition/train_cnn.c's identical block.
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
  // Report metrics that actually matter for a fall detector, not just raw
  // error: per-timestep classification accuracy (threshold 0.5), whether
  // every true fall was ever flagged at all (recall), whether any normal
  // sequence ever triggered a false alarm, and -- for a detected fall --
  // how many timesteps after the true onset the alarm first fired.
  int correct_timesteps = 0, total_timesteps = 0;
  int falls_detected = 0, falls_total = 0;
  int false_alarms = 0, normals_total = 0;
  int latency_sum = 0, latency_count = 0;
  for (int i = 0; i < num_validation; i++) {
    nn_reset_state(nn);
    int onset = -1, detected_at = -1;
    for (int t = 0; t < SEQUENCE_LEN; t++) {
      float pred = nn_predict(nn, val_window[i][t])[0];
      bool predicted_fall = pred >= 0.5f;
      bool actual_fall = val_label[i][t] >= 0.5f;
      if (predicted_fall == actual_fall)
        correct_timesteps++;
      total_timesteps++;
      if (actual_fall && onset < 0)
        onset = t;
      if (predicted_fall && detected_at < 0)
        detected_at = t;
    }
    if (val_is_fall[i]) {
      falls_total++;
      if (detected_at >= onset && onset >= 0) {
        falls_detected++;
        latency_sum += detected_at - onset;
        latency_count++;
      }
    } else {
      normals_total++;
      if (detected_at >= 0)
        false_alarms++;
    }
  }
  printf("Final (last epoch) train error: %f, validation error: %f\n", train_error, validation_error);
  printf("Best validation error (the model saved to disk): %f\n", best_validation_error);
  printf("Training epochs: %d\n", epochs);
  printf("Per-timestep accuracy: %d/%d = %.2f%%\n", correct_timesteps, total_timesteps,
         100.0f * (float)correct_timesteps / (float)total_timesteps);
  printf("Falls detected: %d/%d\n", falls_detected, falls_total);
  printf("False alarms (normal sequences that ever triggered): %d/%d\n", false_alarms, normals_total);
  if (latency_count > 0)
    printf("Average detection latency (timesteps after true onset): %.1f\n", (float)latency_sum / (float)latency_count);
  nn_free(nn);
  return 0;
}
