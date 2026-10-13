/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Example: training a network to tell, from a few seconds of vibration
// off an accelerometer bolted to an industrial machine, whether the
// machine is healthy -- and if not, which of seven failure modes is
// developing in it -- early enough to schedule a repair before it fails.
//
// There's nothing to download. Public run-to-failure vibration datasets
// exist, but each covers one or two failure modes on one test rig; this
// example wants seven modes, each at every severity from barely-started
// to about-to-fail, on many different machines (so that the value of
// comparing a machine with its own past can be measured -- see
// health_monitor.h). So, like ../gesture_recognition and
// ../fall_detection, it synthesizes its data: machine_sim.[ch] models the
// physics of each fault's vibration signature and produces raw
// accelerometer samples, which then go through exactly the
// vibration_features.[ch]/health_monitor.[ch] code that runs on the STM32.
//
// Each training row is one snapshot from one machine:
//   input  = the model input health_monitor.c builds: the snapshot's
//            spectral fingerprint, and (unless --present-only) how far it
//            has moved from that machine's commissioning baseline
//   target = one-hot machine condition (see machine_condition_t)
// The fault severity of each row is drawn uniformly from
// [MIN_TRAINING_SEVERITY, 1], so most of the training set is faults well
// short of failure -- the stage at which catching them is worth something.

#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "machine_sim.h"
#include "nn.h"

// See ../character_recognition/train_fc.c's identical constants for the
// early-stopping rationale.
#define EARLY_STOPPING_PATIENCE 10
#define MAX_EPOCHS 500

#define TRAIN_MACHINES 400
#define VALIDATION_MACHINES 80
#define SNAPSHOTS_PER_MACHINE 16
#define MIN_TRAINING_SEVERITY 0.05f

typedef struct {
  int rows, num_inputs;
  float *input;   // rows x num_inputs
  float *target;  // rows x CONDITION_COUNT
  int *label;
} dataset_t;

// Builds a dataset of `machines` distinct simulated machines: each is
// commissioned (while healthy) and then observed SNAPSHOTS_PER_MACHINE
// times, each time in a randomly chosen condition at a random severity.
static dataset_t generate_dataset(int machines, int num_inputs)
{
  dataset_t d;
  d.rows = machines * SNAPSHOTS_PER_MACHINE;
  d.num_inputs = num_inputs;
  d.input = (float *)malloc((size_t)d.rows * (size_t)num_inputs * sizeof(float));
  d.target = (float *)calloc((size_t)d.rows * CONDITION_COUNT, sizeof(float));
  d.label = (int *)malloc((size_t)d.rows * sizeof(int));
  if (d.input == NULL || d.target == NULL || d.label == NULL) {
    fprintf(stderr, "Error: Out of memory generating dataset.\n");
    exit(1);
  }
  static health_monitor_t hm; // ~70KB: static rather than on the stack
  int row = 0;
  for (int i = 0; i < machines; i++) {
    machine_t m;
    machine_create(&m);
    health_monitor_init(&hm, num_inputs);
    machine_commission(&hm, &m);
    for (int j = 0; j < SNAPSHOTS_PER_MACHINE; j++) {
      machine_condition_t c = (machine_condition_t)(rand() % CONDITION_COUNT);
      health_report_t report;
      machine_observe(&hm, NULL, &m, c, sim_uniform(MIN_TRAINING_SEVERITY, 1.0f), &report);
      memcpy(d.input + (size_t)row * num_inputs, report.model_input, (size_t)num_inputs * sizeof(float));
      d.target[(size_t)row * CONDITION_COUNT + c] = 1.0f;
      d.label[row] = c;
      row++;
    }
  }
  return d;
}

static void dataset_free(dataset_t *d)
{
  free(d->input);
  free(d->target);
  free(d->label);
}

static float dataset_error(nn_t *nn, const dataset_t *d)
{
  float total = 0.0f;
  for (int i = 0; i < d->rows; i++)
    total += nn_error(nn, d->input + (size_t)i * d->num_inputs, d->target + (size_t)i * CONDITION_COUNT);
  return total / (float)d->rows;
}

// The "look at your data" step: for one machine, which fingerprint bands
// each condition moves most, relative to that machine's own baseline, at
// a moderate severity. These should line up with the physics described in
// health_monitor.h's condition list -- 1x for imbalance, 2x/3x for
// misalignment, the defect frequencies in the envelope spectrum for the
// bearing faults -- or the model has nothing sensible to learn.
static void print_sample_fingerprints(void)
{
  static health_monitor_t hm;
  vibration_features_t *vf = &hm.vf;
  machine_t m;
  machine_create(&m);
  health_monitor_init(&hm, HM_INPUTS_WITH_BASELINE);
  machine_commission(&hm, &m);
  printf("Largest fingerprint changes from one machine's baseline, at severity 0.5\n");
  printf("(spectrum = vibration spectrum band, envelope = envelope spectrum band; change in log10 energy):\n");
  for (int c = 0; c < CONDITION_COUNT; c++) {
    health_report_t report;
    machine_observe(&hm, NULL, &m, (machine_condition_t)c, 0.5f, &report);
    const float *delta = report.model_input + VF_FINGERPRINT_LEN;
    bool used[VF_FINGERPRINT_LEN] = {false};
    printf("  %-14s (%.2f mm/s):", condition_names[c], report.velocity_rms);
    for (int rank = 0; rank < 3; rank++) {
      int best = 0;
      while (used[best])
        best++;
      for (int i = best + 1; i < VF_FINGERPRINT_LEN; i++)
        if (!used[i] && delta[i] > delta[best])
          best = i;
      used[best] = true;
      bool envelope = best >= VF_SPECTRUM_BANDS;
      const uint16_t *edges = envelope ? vf->envelope_edges : vf->spectrum_edges;
      int b = envelope ? best - VF_SPECTRUM_BANDS : best;
      float bin_hz = VF_SAMPLE_RATE / (float)VF_FRAME_LEN;
      printf("  %s %.0f-%.0fHz %+.2f", envelope ? "envelope" : "spectrum",
             edges[b] * bin_hz, edges[b + 1] * bin_hz, delta[best]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  bool present_only = argc == 3 && strcmp(argv[1], "--present-only") == 0;
  if (argc != 2 && !present_only) {
    printf("Usage: %s [--present-only] <model-file>\n", argv[0]);
    printf("  --present-only : train on each snapshot's own fingerprint alone, without the machine's baseline (for comparison)\n");
    printf("  <model-file>   : Path to the neural net model to load or create (e.g., model.txt or model.bin)\n");
    return 1;
  }
  const char *model_path = argv[argc - 1];
  // Tunable hyperparameters
  int num_inputs = present_only ? HM_INPUTS_PRESENT_ONLY : HM_INPUTS_WITH_BASELINE;
  int num_outputs = CONDITION_COUNT;
  float learning_rate = 0.01f;
  float annealing = 1.0f;
  // End of tunable parameters
  nn_t *nn;
  int epochs = 0;
  float train_error = 1.0f;
  float validation_error = 1.0f;
  srand((unsigned)time(NULL));
  print_sample_fingerprints();

  printf("Simulating %d training machines and %d validation machines (%d snapshots each)...\n",
         TRAIN_MACHINES, VALIDATION_MACHINES, HM_COMMISSIONING_SNAPSHOTS + SNAPSHOTS_PER_MACHINE);
  fflush(stdout);
  dataset_t train_data = generate_dataset(TRAIN_MACHINES, num_inputs);
  dataset_t validation_data = generate_dataset(VALIDATION_MACHINES, num_inputs);

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
    // A plain fully-connected network is enough here: the hard part --
    // turning seconds of raw vibration into a few dozen numbers that
    // capture the machine's frequency content -- is the FFT front end's
    // job, and each snapshot is classified on its own (the baseline
    // supplies the history), so there's no sequence for a recurrent layer
    // to remember.
    nn_add_layer(nn, LAYER_TYPE_FC, 32, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    nn_add_layer(nn, LAYER_TYPE_FC, 16, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
    // Softmax + cross-entropy: one condition per machine at a time.
    nn_add_layer(nn, LAYER_TYPE_OUTPUT, num_outputs, ACTIVATION_FUNCTION_TYPE_SOFTMAX, NULL);
  } else {
    printf("Using existing model file: %s\n", model_path);
    if ((nn->width[0] != (uint32_t)num_inputs) ||
        (nn->width[nn->depth - 1] != (uint32_t)num_outputs)) {
      printf("Error: Loaded model dimensions do not match expected: %d %d.%s\n", num_inputs, num_outputs,
             present_only ? "" : " (Was it trained with --present-only?)");
      nn_free(nn);
      return 1;
    }
  }
  // See ../character_recognition/train_fc.c's identical block for why this
  // baseline matters when resuming an existing model.
  float best_validation_error = FLT_MAX;
  if (resuming) {
    best_validation_error = dataset_error(nn, &validation_data);
    printf("Starting validation error (existing model): %.5f\n", best_validation_error);
  }
  int *order = (int *)malloc((size_t)train_data.rows * sizeof(int));
  if (order == NULL) {
    fprintf(stderr, "Error: Out of memory.\n");
    return 1;
  }
  for (int i = 0; i < train_data.rows; i++)
    order[i] = i;
  int epochs_since_improvement = 0;
  printf("train error, validation error, learning rate\n");
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
      total_train_error += nn_train(nn, train_data.input + (size_t)r * num_inputs,
                                    train_data.target + (size_t)r * CONDITION_COUNT, learning_rate);
    }
    train_error = total_train_error / (float)train_data.rows;
    validation_error = dataset_error(nn, &validation_data);
    epochs++;
    printf("%.5f, %.5f, %.5f\n", train_error, validation_error, learning_rate);
    learning_rate *= annealing;
    if (validation_error < best_validation_error) {
      best_validation_error = validation_error;
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
        printf("No validation improvement for %d epochs (best: %.5f) -- stopping early.\n",
               EARLY_STOPPING_PATIENCE, best_validation_error);
        break;
      }
    }
  }
  free(order);
  // Validation accuracy of the final (last-epoch) weights, as a more
  // intuitive readout than cross-entropy alone. test.c re-evaluates the
  // saved (best) model independently, on machines it has never seen.
  int correct = 0;
  for (int i = 0; i < validation_data.rows; i++) {
    float *out = nn_predict(nn, validation_data.input + (size_t)i * num_inputs);
    int predicted = 0;
    for (int c = 1; c < CONDITION_COUNT; c++)
      if (out[c] > out[predicted])
        predicted = c;
    if (predicted == validation_data.label[i])
      correct++;
  }
  printf("Final (last epoch) train error: %f, validation error: %f\n", train_error, validation_error);
  printf("Best validation error (the model saved to disk): %f\n", best_validation_error);
  printf("Validation accuracy (last epoch): %d/%d = %.1f%%\n", correct, validation_data.rows,
         100.0f * correct / (float)validation_data.rows);
  printf("Training epochs: %d\n", epochs);
  dataset_free(&train_data);
  dataset_free(&validation_data);
  nn_free(nn);
  return 0;
}
