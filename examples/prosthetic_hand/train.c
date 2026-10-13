/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Example: training the shared model behind a myoelectric prosthetic hand
// and wrist -- a network that reads the electrical activity of the forearm
// muscles through a ring of 8 surface electrodes and decides, every 50ms,
// whether the wearer is trying to open the hand, close it, rotate the
// wrist either way, flex or extend it, or do nothing.
//
// Every person's muscles, and every fit of the socket, produce a somewhat
// different pattern for the same movement, so no prosthesis runs a model
// trained only on other people: it is calibrated to its wearer (see
// intent_decoder.h). What this program trains is the starting point that
// calibration adapts -- a model that has seen 28 people make these
// movements, each standardized with their own calibration, so it has
// learned what is common to all of them. test.c measures how much that
// starting point is worth against training from the wearer's calibration
// alone, and against the linear classifier commercial controllers use.
//
// Training data is real: Ninapro DB2, 40 intact subjects recorded with 12
// electrodes at 2kHz (see ninapro.h), turned into features.csv by
// prepare_data.c through the same emg_features.[ch] code that runs on the
// device. Each row is one 200ms window:
//   input  = EF_FEATURES Hudgins features (4 per electrode), standardized
//            with that subject's calibration
//   target = one-hot intent (see intent_t)

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "feature_set.h"
#include "nn.h"

// See ../character_recognition/train_fc.c's identical constants for the
// early-stopping rationale.
#define EARLY_STOPPING_PATIENCE 10
#define MAX_EPOCHS 200

static float dataset_error(nn_t *nn, const rowset_t *d)
{
  float target[INTENT_COUNT];
  float total = 0.0f;
  for (int i = 0; i < d->rows; i++) {
    memset(target, 0, sizeof(target));
    target[d->label[i]] = 1.0f;
    total += nn_error(nn, d->x + (size_t)i * EF_FEATURES, target);
  }
  return total / (float)d->rows;
}

static float dataset_accuracy(nn_t *nn, const rowset_t *d)
{
  int correct = 0;
  for (int i = 0; i < d->rows; i++) {
    const float *out = nn_predict(nn, d->x + (size_t)i * EF_FEATURES);
    int predicted = 0;
    for (int c = 1; c < INTENT_COUNT; c++)
      if (out[c] > out[predicted])
        predicted = c;
    if (predicted == d->label[i])
      correct++;
  }
  return (float)correct / (float)d->rows;
}

// One validation subject, set up the way a prosthesis would be fitted to
// them: calibration data (repetition 1, raw) and the held-out repetitions
// (standardized with that calibration).
typedef struct {
  intent_decoder_t id;
  rowset_t cal, test;
} fitting_t;

// The score early stopping watches: mean accuracy over the validation
// subjects after each has been calibrated and the model fine-tuned on
// their first repetition -- what the model is for. Validation loss
// without fine-tuning would be the wrong target: it's lowest after the
// first epoch, and rises from there as the model learns the training
// subjects' individual patterns. On this data the fitted accuracy is also
// highest within the first few epochs and roughly flat after: what a
// shared model gives fine-tuning to start from -- which features matter,
// and roughly how -- it learns quickly. `copy_path` is where the current
// weights are saved so each subject can fine-tune a fresh copy.
static float fitted_accuracy(nn_t *nn, fitting_t *fits, int count, const char *copy_path)
{
  float scratch[EF_FEATURES + INTENT_COUNT];
  float total = 0.0f;
  nn_save_model_binary(nn, (char *)copy_path);
  for (int v = 0; v < count; v++) {
    nn_t *copy = nn_load_model_binary((char *)copy_path);
    intent_decoder_fine_tune(&fits[v].id, copy, fits[v].cal.x, fits[v].cal.label, fits[v].cal.rows,
                             ID_FINE_TUNE_EPOCHS, ID_FINE_TUNE_RATE, scratch);
    total += dataset_accuracy(copy, &fits[v].test);
    nn_free(copy);
  }
  remove(copy_path);
  return total / (float)count;
}

// The "look at your data" step: for each intent, how active each electrode
// is on average across the training subjects (mean standardized log MAV;
// 0 is the subject's average over all calibration windows). Each movement
// should light up a different part of the ring -- the flexor side of the
// forearm for a fist and wrist flexion, the extensor side for wrist
// extension -- or the model has nothing to learn from.
static void print_activation_patterns(const rowset_t *d)
{
  double sum[INTENT_COUNT][EF_CHANNELS] = {{0}};
  int count[INTENT_COUNT] = {0};
  for (int i = 0; i < d->rows; i++) {
    count[d->label[i]]++;
    for (int c = 0; c < EF_CHANNELS; c++)
      sum[d->label[i]][c] += d->x[(size_t)i * EF_FEATURES + c * EF_FEATURES_PER_CHANNEL];
  }
  printf("Mean activity (standardized log MAV) per electrode, by intent, across the training subjects:\n");
  printf("  %-17s", "");
  for (int c = 0; c < EF_CHANNELS; c++)
    printf("  ch%-3d", c + 1);
  printf("\n");
  for (int k = 0; k < INTENT_COUNT; k++) {
    printf("  %-17s", intent_names[k]);
    for (int c = 0; c < EF_CHANNELS; c++)
      printf(" %+6.2f", sum[k][c] / (count[k] > 0 ? count[k] : 1));
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to the neural net model to load or create (e.g., model.txt or model.bin)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Tunable hyperparameters
  int num_inputs = EF_FEATURES;
  int num_outputs = INTENT_COUNT;
  float learning_rate = 0.001f;
  // End of tunable parameters
  nn_t *nn;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  srand((unsigned)time(NULL));

  features_t f;
  if (features_load("features.csv", &f) != 0)
    return 1;
  rowset_t train_data = features_pool(&f, 2, TRAIN_FIRST, TRAIN_LAST);
  rowset_t validation_data = features_pool(&f, 2, VALIDATION_FIRST, VALIDATION_LAST);
  fitting_t fits[VALIDATION_LAST - VALIDATION_FIRST + 1];
  int num_fits = VALIDATION_LAST - VALIDATION_FIRST + 1;
  for (int v = 0; v < num_fits; v++) {
    fits[v].cal = features_select(&f, 2, VALIDATION_FIRST + v, 1u << 1);
    fits[v].test = features_select(&f, 2, VALIDATION_FIRST + v, TEST_REPS);
    intent_decoder_init(&fits[v].id);
    intent_decoder_calibrate(&fits[v].id, fits[v].cal.x, fits[v].cal.label, fits[v].cal.rows);
    rowset_standardize(&fits[v].test, &fits[v].id);
  }
  features_free(&f);
  char copy_path[512];
  snprintf(copy_path, sizeof(copy_path), "%s.fitting.bin", model_path);
  printf("Training on %d windows from %d subjects, validating on %d windows from %d others.\n\n", train_data.rows,
         TRAIN_LAST - TRAIN_FIRST + 1, validation_data.rows, VALIDATION_LAST - VALIDATION_FIRST + 1);
  print_activation_patterns(&train_data);

  // Attempt to load an existing model; see ../gesture_recognition/train.c's
  // identical block for why the inplace format needs
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
      return 1;
    }
    nn_add_layer(nn, LAYER_TYPE_INPUT, num_inputs, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
    // A plain fully-connected network, like ../predictive_maintenance's:
    // the features already summarize the signal, and each 200ms window is
    // classified on its own (the decoder's majority vote supplies the
    // smoothing over time). Feeding it several past windows as well was
    // tried and helped no more than it helped the linear baseline, while
    // making the hand slower to respond to a change of intent.
    nn_add_layer(nn, LAYER_TYPE_FC, 64, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    nn_add_layer(nn, LAYER_TYPE_FC, 32, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    // Softmax + cross-entropy: one intent at a time.
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, num_outputs, ACTIVATION_FUNCTION_TYPE_SOFTMAX, NULL);
  } else {
    printf("Using existing model file: %s\n", model_path);
    if ((nn->width[0] != (uint32_t)num_inputs) || (nn->width[nn->depth - 1] != (uint32_t)num_outputs)) {
      printf("Error: Loaded model dimensions do not match expected: %d %d\n", num_inputs, num_outputs);
      nn_free(nn);
      return 1;
    }
  }
  // Adam, as in ../fall_detection: the optimizer state isn't saved with
  // the model, so a resumed run starts it fresh.
  nn_set_optimizer(nn, NN_OPTIMIZER_ADAM, 0.9f, 0.999f, 1e-8f);
  // See ../character_recognition/train_fc.c's identical block for why this
  // baseline matters when resuming an existing model.
  float best_fitted = 0.0f;
  if (resuming) {
    best_fitted = fitted_accuracy(nn, fits, num_fits, copy_path);
    printf("Starting fitted validation accuracy (existing model): %.1f%%\n", 100.0f * best_fitted);
  }
  int *order = (int *)malloc((size_t)train_data.rows * sizeof(int));
  if (order == NULL) {
    fprintf(stderr, "Error: Out of memory.\n");
    return 1;
  }
  for (int i = 0; i < train_data.rows; i++)
    order[i] = i;
  int epochs_since_improvement = 0;
  float target[INTENT_COUNT];
  printf("train error, validation error, validation accuracy (shared), validation accuracy (fine-tuned on 1 rep)\n");
  while (epochs < MAX_EPOCHS) {
    for (int i = train_data.rows - 1; i > 0; i--) {
      int j = rand() % (i + 1);
      int tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    float total_train_error = 0.0f;
    for (int i = 0; i < train_data.rows; i++) {
      int r = order[i];
      memset(target, 0, sizeof(target));
      target[train_data.label[r]] = 1.0f;
      total_train_error += nn_train(nn, train_data.x + (size_t)r * EF_FEATURES, target, learning_rate);
    }
    train_error = total_train_error / (float)train_data.rows;
    validation_error = dataset_error(nn, &validation_data);
    float fitted = fitted_accuracy(nn, fits, num_fits, copy_path);
    epochs++;
    printf("%.5f, %.5f, %.1f%%, %.1f%%\n", train_error, validation_error, 100.0f * dataset_accuracy(nn, &validation_data),
           100.0f * fitted);
    if (fitted > best_fitted) {
      best_fitted = fitted;
      epochs_since_improvement = 0;
      // Save back in the same format the model was loaded from when
      // resuming, same rationale as ../character_recognition/train_fc.c's
      // identical block.
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
        printf("No validation improvement for %d epochs (best: %.1f%%) -- stopping early.\n",
               EARLY_STOPPING_PATIENCE, 100.0f * best_fitted);
        break;
      }
    }
  }
  free(order);
  printf("Final (last epoch) train error: %f, validation error: %f\n", train_error, validation_error);
  printf("Best fitted validation accuracy (the model saved to disk): %.1f%%\n", 100.0f * best_fitted);
  printf("Training epochs: %d\n", epochs);
  rowset_free(&train_data);
  rowset_free(&validation_data);
  for (int v = 0; v < num_fits; v++) {
    rowset_free(&fits[v].cal);
    rowset_free(&fits[v].test);
  }
  nn_free(nn);
  return 0;
}
