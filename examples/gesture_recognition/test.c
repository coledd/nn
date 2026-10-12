/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Independently re-evaluates a model saved by train.c against a
// freshly-synthesized batch of gesture windows -- unlike examples/character_recognition/test.c
// (which re-loads train.csv/test.csv, fixed files carved out once by split.py),
// there is nothing to load here: every window generate_gesture() produces
// is new, so it's unseen by construction, whether or not it happens to
// overlap in spirit with whatever the model trained on. Reports a
// confusion matrix and per-class/overall accuracy, a more independent
// readout than train.c's own final validation accuracy (which is
// measured against the same fixed validation set early stopping used to
// pick the saved model).

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gesture_data.h"
#include "nn.h"

#define WINDOWS_PER_CLASS 50 // freshly-synthesized test windows per gesture class

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c (ascii, binary, or inplace)\n");
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
  if ((model->width[0] != (uint32_t)NUM_AXES) || (model->width[model->depth - 1] != (uint32_t)GESTURE_COUNT)) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match the gesture task (%d inputs, %d outputs) -- is this a train.c model?\n",
            model->width[0], model->width[model->depth - 1], NUM_AXES, GESTURE_COUNT);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }
  srand((unsigned)time(NULL));

  // confusion[actual][predicted]
  int confusion[GESTURE_COUNT][GESTURE_COUNT] = {{0}};
  for (int actual = 0; actual < GESTURE_COUNT; actual++) {
    for (int k = 0; k < WINDOWS_PER_CLASS; k++) {
      float window[WINDOW_LEN][NUM_AXES];
      generate_gesture((gesture_t)actual, window);
      // Classify off the final timestep's output, same convention
      // train.c's own end-of-run accuracy readout uses: the RNN's
      // hidden state has by then seen the whole window.
      nn_reset_state(model);
      float *out = NULL;
      for (int t = 0; t < WINDOW_LEN; t++)
        out = nn_predict(model, window[t]);
      int predicted = 0;
      for (int c = 1; c < GESTURE_COUNT; c++)
        if (out[c] > out[predicted])
          predicted = c;
      confusion[actual][predicted]++;
    }
  }

  printf("Confusion matrix (rows = actual, columns = predicted), %d windows per class:\n", WINDOWS_PER_CLASS);
  printf("%-12s", "");
  for (int p = 0; p < GESTURE_COUNT; p++)
    printf("%-12s", gesture_names[p]);
  printf("\n");
  int total_correct = 0, total = 0;
  for (int a = 0; a < GESTURE_COUNT; a++) {
    printf("%-12s", gesture_names[a]);
    for (int p = 0; p < GESTURE_COUNT; p++) {
      printf("%-12d", confusion[a][p]);
      total += confusion[a][p];
      if (a == p)
        total_correct += confusion[a][p];
    }
    int class_total = 0;
    for (int p = 0; p < GESTURE_COUNT; p++)
      class_total += confusion[a][p];
    printf("  (%d/%d = %.1f%%)\n", confusion[a][a], class_total, 100.0f * confusion[a][a] / (float)class_total);
  }
  printf("\nOverall accuracy: %d/%d = %.2f%%\n", total_correct, total, 100.0f * total_correct / (float)total);

  nn_free(model);
  free(inplace_buf);
  return 0;
}
