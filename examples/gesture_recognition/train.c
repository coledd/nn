/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Example: training an RNN layer to classify short bursts of simulated
// 3-axis accelerometer readings -- the kind of time-series a wearable or
// remote control's IMU streams on an embedded target -- into one of four
// gestures. Unlike train.c's MNIST digits (a fixed, downloaded image
// dataset), there is nothing to download here: every gesture window is
// synthesized on the fly (generate_gesture() below) the same way a live
// sensor would produce one, and fed into the network one timestep per
// nn_train()/nn_predict() call, since LAYER_TYPE_RNN processes a sequence
// by carrying its hidden state across calls rather than seeing a whole
// window at once (see LAYER_TYPE_RNN's comment in nn.h).

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gesture_data.h"
#include "nn.h"

// See train.c's identical constants for the early-stopping rationale.
#define EARLY_STOPPING_PATIENCE 5
#define MAX_EPOCHS 500

#define WINDOWS_PER_EPOCH 200      // freshly-synthesized training windows per epoch
#define VALIDATION_WINDOWS_PER_CLASS 8 // fixed, generated once, for a stable early-stopping signal

// Prints one freshly-generated example window per gesture class to stdout.
// Unlike MNIST (where a digit image is already something you can look at),
// there's no way to eyeball this synthetic sensor data without printing it
// -- this is the "look at your data" step before spending time training on
// it, confirming the four classes actually look distinguishable as raw
// accelerometer numbers.
static void print_sample_gestures(void)
{
  printf("Sample gesture windows (accelerometer x, y, z per timestep):\n");
  for (int g = 0; g < GESTURE_COUNT; g++) {
    float window[WINDOW_LEN][NUM_AXES];
    generate_gesture((gesture_t)g, window);
    printf("\n%s:\n", gesture_names[g]);
    for (int t = 0; t < WINDOW_LEN; t++) {
      printf("  t=%2d: % .3f % .3f % .3f\n", t, window[t][0], window[t][1], window[t][2]);
    }
  }
  printf("\n");
}

// Runs one whole gesture window through the network, timestep by timestep,
// training (or, when `train` is false, just measuring error via
// nn_error()) against the SAME one-hot label at every timestep: a
// "many-to-many" scheme where early timesteps may not yet look
// distinctive, but repeating the label gives the network many chances per
// window to learn the pattern -- and lets a caller watch its confidence in
// the correct class rise across the window. Always starts by resetting the
// RNN's hidden state (nn_reset_state()): each window is an independent
// sequence, and the previous window's final hidden state must not leak
// into this one (see nn_reset_state()'s comment in nn.h). Returns the
// average per-timestep error for this window.
static float run_window(nn_t *nn, float window[WINDOW_LEN][NUM_AXES], float *target, float rate, bool train)
{
  nn_reset_state(nn);
  float total_err = 0.0f;
  for (int t = 0; t < WINDOW_LEN; t++) {
    total_err += train ? nn_train(nn, window[t], target, rate) : nn_error(nn, window[t], target);
  }
  return total_err / (float)WINDOW_LEN;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to the neural net model to load or create (e.g., gesture_model.txt or .bin)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Tunable hyperparameters
  int num_inputs = NUM_AXES;
  int num_outputs = GESTURE_COUNT;
  int hidden_units = 16;
  float learning_rate = 0.05f;
  float annealing = 1.0f;
  // End of tunable parameters
  nn_t *nn;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  srand((unsigned)time(NULL));
  print_sample_gestures();

  // A fixed validation set, generated once (unlike the training windows
  // below, which are freshly synthesized every epoch, as if streaming
  // straight off a live sensor) so early stopping has a stable signal to
  // compare against from one epoch to the next.
  static float val_window[VALIDATION_WINDOWS_PER_CLASS * GESTURE_COUNT][WINDOW_LEN][NUM_AXES];
  static float val_target[VALIDATION_WINDOWS_PER_CLASS * GESTURE_COUNT][GESTURE_COUNT];
  int num_validation = 0;
  for (int g = 0; g < GESTURE_COUNT; g++) {
    for (int k = 0; k < VALIDATION_WINDOWS_PER_CLASS; k++) {
      generate_gesture((gesture_t)g, val_window[num_validation]);
      gesture_one_hot((gesture_t)g, val_target[num_validation]);
      num_validation++;
    }
  }

  // Attempt to load an existing model; see train.c's identical block for
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
    // A single RNN hidden layer is enough to show off the point of this
    // demo: telling gestures apart requires remembering recent samples
    // (a lone accelerometer reading can't distinguish SHAKE from STILL from
    // one instant alone), which is exactly what LAYER_TYPE_RNN's persistent
    // hidden state provides and a plain FC layer looking at a single
    // timestep could not.
    nn_add_layer(nn, LAYER_TYPE_RNN, hidden_units, ACTIVATION_FUNCTION_TYPE_TANH, NULL);
    // Softmax + cross-entropy models the four gestures as mutually
    // exclusive classes (see ACTIVATION_FUNCTION_TYPE_SOFTMAX in nn.h).
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, num_outputs, ACTIVATION_FUNCTION_TYPE_SOFTMAX, NULL);
  } else {
    printf("Using existing model file: %s\n", model_path);
    if ((nn->width[0] != (uint32_t)num_inputs) ||
        (nn->width[nn->depth - 1] != (uint32_t)num_outputs)) {
      printf("Error: Loaded model dimensions do not match expected: %d %d.\n", num_inputs, num_outputs);
      nn_free(nn);
      return 1;
    }
  }
  // See train.c's identical block for why this baseline matters when
  // resuming an existing model.
  float best_validation_error = FLT_MAX;
  if (resuming) {
    float total = 0.0f;
    for (int i = 0; i < num_validation; i++)
      total += run_window(nn, val_window[i], val_target[i], 0.0f, false);
    best_validation_error = total / (float)num_validation;
    printf("Starting validation error (existing model): %.5f\n", best_validation_error);
  }
  int epochs_since_improvement = 0;
  printf("train error, validation error, learning rate\n");
  while (epochs < MAX_EPOCHS) {
    float total_train_error = 0.0f;
    for (int j = 0; j < WINDOWS_PER_EPOCH; j++) {
      gesture_t g = (gesture_t)(rand() % GESTURE_COUNT);
      float window[WINDOW_LEN][NUM_AXES];
      float target[GESTURE_COUNT];
      generate_gesture(g, window);
      gesture_one_hot(g, target);
      total_train_error += run_window(nn, window, target, learning_rate, true);
    }
    train_error = total_train_error / (float)WINDOWS_PER_EPOCH;
    float total_validation_err = 0.0f;
    for (int i = 0; i < num_validation; i++)
      total_validation_err += run_window(nn, val_window[i], val_target[i], 0.0f, false);
    validation_error = total_validation_err / (float)num_validation;
    epochs++;
    printf("%.5f, %.5f, %.5f\n", train_error, validation_error, learning_rate);
    learning_rate *= annealing;
    if (validation_error < best_validation_error) {
      best_validation_error = validation_error;
      epochs_since_improvement = 0;
      // Save back in the same format the model was loaded from when
      // resuming, same rationale as train.c's identical block.
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
  // Report per-class validation accuracy (the gesture with the highest
  // final-timestep output probability) as a more intuitive readout than
  // cross-entropy error alone.
  int correct = 0;
  for (int i = 0; i < num_validation; i++) {
    nn_reset_state(nn);
    float *out = NULL;
    for (int t = 0; t < WINDOW_LEN; t++)
      out = nn_predict(nn, val_window[i][t]);
    int predicted = 0;
    for (int c = 1; c < GESTURE_COUNT; c++)
      if (out[c] > out[predicted])
        predicted = c;
    int actual = i / VALIDATION_WINDOWS_PER_CLASS;
    if (predicted == actual)
      correct++;
  }
  printf("Final (last epoch) train error: %f, validation error: %f\n", train_error, validation_error);
  printf("Best validation error (the model saved to disk): %f\n", best_validation_error);
  printf("Validation accuracy (final-timestep classification): %d/%d\n", correct, num_validation);
  printf("Training epochs: %d\n", epochs);
  printf("Gesture classes: ");
  for (int g = 0; g < GESTURE_COUNT; g++)
    printf("%d=%s%s", g, gesture_names[g], g + 1 < GESTURE_COUNT ? ", " : "\n");
  nn_free(nn);
  return 0;
}
