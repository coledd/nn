/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently re-evaluates a model saved by train_rnn.c/train_gru.c/
// train_lstm.c against a freshly-synthesized batch of monitoring sequences
// (architecture-agnostic, like examples/character_recognition/test.c is for
// its two training programs -- this just reads whatever the model's
// input/output widths are) -- same rationale as
// examples/gesture_recognition/test.c: nothing about any of their training
// data is ever persisted to disk, so every sequence generated here is new,
// unseen by construction, whether or not it happens to resemble something
// the model trained on.
// Reports the metrics that actually matter for a fall detector (not just
// raw error): per-timestep accuracy, recall (falls ever detected), false
// alarms (normal sequences that ever triggered), and average detection
// latency after the true onset.

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fall_data.h"
#include "nn.h"

#define SEQUENCES_PER_CLASS 50 // freshly-synthesized test sequences per class (normal / fall)

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train_rnn.c/train_gru.c/train_lstm.c (ascii, binary, or inplace)\n");
    return 1;
  }
  const char *model_path = argv[1];
  // Load a previously saved model. See examples/character_recognition/test.c's identical block for why the
  // inplace format needs to be read into a buffer first.
  nn_model_format_t format = nn_model_format(model_path);
  nn_t *model;
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    inplace_buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!inplace_buf || fread(inplace_buf, 1, (size_t)size, file) != (size_t)size) {
      fprintf(stderr, "Error: Could not read model file: %s\n", model_path);
      fclose(file);
      free(inplace_buf);
      return 1;
    }
    fclose(file);
    model = nn_load_model_inplace(inplace_buf, (size_t)size);
  } else {
    model = nn_load_model((char *)model_path);
  }
  if (model == NULL) {
    fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
    free(inplace_buf);
    return 1;
  }
  if ((model->width[0] != (uint32_t)NUM_AXES) || (model->width[model->depth - 1] != 1)) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match the fall-detection task (%d inputs, 1 output) -- is this a train_rnn.c/train_gru.c/train_lstm.c model?\n",
            model->width[0], model->width[model->depth - 1], NUM_AXES);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }
  srand((unsigned)time(NULL));

  static float window[SEQUENCE_LEN][NUM_AXES];
  static float label[SEQUENCE_LEN];
  int correct_timesteps = 0, total_timesteps = 0;
  int falls_detected = 0, falls_total = 0;
  int false_alarms = 0, normals_total = 0;
  int latency_sum = 0, latency_count = 0;

  for (int kind = 0; kind < 2; kind++) {
    bool is_fall = (kind == 1);
    for (int k = 0; k < SEQUENCES_PER_CLASS; k++) {
      if (is_fall)
        generate_fall_sequence(window, label);
      else
        generate_normal_sequence(window, label);

      nn_reset_state(model);
      int onset = -1, detected_at = -1;
      for (int t = 0; t < SEQUENCE_LEN; t++) {
        float pred = nn_predict(model, window[t])[0];
        bool predicted_fall = pred >= 0.5f;
        bool actual_fall = label[t] >= 0.5f;
        if (predicted_fall == actual_fall)
          correct_timesteps++;
        total_timesteps++;
        if (actual_fall && onset < 0)
          onset = t;
        if (predicted_fall && detected_at < 0)
          detected_at = t;
      }
      if (is_fall) {
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
  }

  printf("Test sequences: %d normal, %d fall (%d timesteps each, freshly synthesized)\n",
         normals_total, falls_total, SEQUENCE_LEN);
  printf("Per-timestep accuracy: %d/%d = %.2f%%\n", correct_timesteps, total_timesteps,
         100.0f * (float)correct_timesteps / (float)total_timesteps);
  printf("Falls detected (recall): %d/%d = %.1f%%\n", falls_detected, falls_total,
         100.0f * (float)falls_detected / (float)falls_total);
  printf("False alarms (normal sequences that ever triggered): %d/%d = %.1f%%\n", false_alarms, normals_total,
         100.0f * (float)false_alarms / (float)normals_total);
  if (latency_count > 0)
    printf("Average detection latency (timesteps after true onset): %.1f\n", (float)latency_sum / (float)latency_count);
  else
    printf("Average detection latency: n/a (no falls were detected)\n");

  nn_free(model);
  free(inplace_buf);
  return 0;
}
