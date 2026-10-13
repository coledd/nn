/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Example: training an LSTM layer to continuously monitor a simulated
// 3-axis accelerometer stream -- the kind of always-on sensor processing a
// fall-detection wearable or medical-alert pendant does -- and flag a fall
// as soon as one occurs. The LSTM counterpart to train_rnn.c/train_gru.c in
// this same directory, on the identical task and data, so the three can be
// compared directly. Unlike examples/gesture_recognition/train.c's RNN
// (which classifies one short, fixed-length gesture window with a single
// label), a fall event here is a multi-phase pattern (a brief free-fall
// dip, an impact spike, then a long stretch of unusual post-fall
// stillness) spread across a much longer sequence, and the label VARIES
// per timestep -- 0 throughout ordinary activity, flipping to 1 at the
// instant a fall begins and staying 1 for the rest of the monitoring
// window.
//
// Confirming a genuine fall (and not, say, just setting the device down
// hard) benefits from remembering the free-fall dip once the impact arrives
// and then watching what happens afterward -- exactly the kind of "hold
// relevant context, let go of irrelevant context" job LSTM's gated cell
// state is for, more than train_rnn.c's plain hidden state is (see
// LAYER_TYPE_LSTM's comment in nn.h). Measured against train_rnn.c/
// train_gru.c on this exact task, over 20 training runs each: all three
// detect every fall with no false alarms, so this synthetic task turns out
// learnable by a plain RNN too. This is the more expensive of the two gated
// options (four gates and two persistent states vs. GRU's three gates and
// one -- see train_gru.c); it trains in a median of 41 epochs. See this
// directory's README for the measured comparison. Every sequence is
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

// Early stopping keeps the epoch that makes the fewest detection mistakes
// on the validation set (see validate()), and gives up after this many
// epochs without a better one. Mistakes change in whole steps and often sit
// level for a few epochs while the network is still improving, so this is
// longer than examples/character_recognition's.
#define EARLY_STOPPING_PATIENCE 10
#define MAX_EPOCHS 500

#define SEQUENCES_PER_EPOCH 100           // freshly-synthesized training sequences per epoch
#define VALIDATION_SEQUENCES_PER_CLASS 100 // fixed, generated once, for a stable early-stopping signal

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
// by resetting the LSTM's hidden AND cell state (nn_reset_state()): each
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

// Scores the network on the validation set the way test.c does: a fall
// sequence counts as missed unless the first alarm comes at or after the
// fall's onset, and a normal sequence counts as a false alarm if it ever
// alarms. Returns missed falls plus false alarms; *error gets the mean
// per-timestep error, which only breaks ties between epochs with the same
// number of mistakes. Mean error alone is the wrong thing to select on: a
// brief spike above 0.5 during normal activity costs a single timestep's
// worth of error but is a whole false alarm, so the lowest-error epoch can
// be one that false-alarms -- measured over 60 training runs per
// architecture, selecting on error kept such a model in up to a third of
// runs, and selecting on mistakes kept none.
static int validate(nn_t *nn, float window[][SEQUENCE_LEN][NUM_AXES], float label[][SEQUENCE_LEN],
                    const bool *is_fall, int count, float *error)
{
  int mistakes = 0;
  float total_err = 0.0f;
  for (int i = 0; i < count; i++) {
    nn_reset_state(nn);
    int onset = -1, first_alarm = -1;
    for (int t = 0; t < SEQUENCE_LEN; t++) {
      float target[1] = {label[i][t]};
      total_err += nn_error(nn, window[i][t], target);
      if (label[i][t] >= 0.5f && onset < 0)
        onset = t;
      if (nn->neuron[nn->depth - 1][0] >= 0.5f && first_alarm < 0)
        first_alarm = t;
    }
    if (is_fall[i])
      mistakes += !(onset >= 0 && first_alarm >= onset);
    else
      mistakes += first_alarm >= 0;
  }
  *error = total_err / (float)(count * SEQUENCE_LEN);
  return mistakes;
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
    // LSTM (not RNN): confirming a fall means remembering the free-fall dip
    // across the impact, then continuing to watch for post-fall stillness
    // many timesteps later -- exactly the longer-range, selectively-retained
    // memory LSTM's gated cell state is for (see this file's top comment).
    nn_add_layer(nn, LAYER_TYPE_LSTM, hidden_units, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
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
  int best_validation_mistakes = 2 * VALIDATION_SEQUENCES_PER_CLASS + 1;
  if (resuming) {
    best_validation_mistakes = validate(nn, val_window, val_label, val_is_fall, num_validation, &best_validation_error);
    printf("Starting validation (existing model): %d mistakes, error %.5f\n", best_validation_mistakes,
           best_validation_error);
  }
  int epochs_since_improvement = 0;
  static float window[SEQUENCE_LEN][NUM_AXES];
  static float label[SEQUENCE_LEN];
  printf("train error, validation error, validation mistakes, learning rate\n");
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
    int validation_mistakes = validate(nn, val_window, val_label, val_is_fall, num_validation, &validation_error);
    epochs++;
    printf("%.5f, %.5f, %d, %.5f\n", train_error, validation_error, validation_mistakes, learning_rate);
    learning_rate *= annealing;
    if (validation_mistakes < best_validation_mistakes ||
        (validation_mistakes == best_validation_mistakes && validation_error < best_validation_error)) {
      best_validation_mistakes = validation_mistakes;
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
        printf("No validation improvement for %d epochs (best: %d mistakes, error %.5f) -- stopping early.\n",
               EARLY_STOPPING_PATIENCE, best_validation_mistakes, best_validation_error);
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
  printf("Best validation (the model saved to disk): %d mistakes, error %f\n", best_validation_mistakes,
         best_validation_error);
  printf("Training epochs: %d\n", epochs);
  printf("Validation set, last epoch's weights (./test evaluates the saved model on fresh data):\n");
  printf("Per-timestep accuracy: %d/%d = %.2f%%\n", correct_timesteps, total_timesteps,
         100.0f * (float)correct_timesteps / (float)total_timesteps);
  printf("Falls detected: %d/%d\n", falls_detected, falls_total);
  printf("False alarms (normal sequences that ever triggered): %d/%d\n", false_alarms, normals_total);
  if (latency_count > 0)
    printf("Average detection latency (timesteps after true onset): %.1f\n", (float)latency_sum / (float)latency_count);
  nn_free(nn);
  return 0;
}
