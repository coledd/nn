/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Evaluates the shared model saved by train.c on people it has never
// seen -- 8 intact subjects (Ninapro DB2) and 11 transradial amputees
// (Ninapro DB3) -- the way a prosthesis would be fitted to each of them:
// calibrate on some repetitions of each movement, then decode the
// repetitions held back (2 and 5).
//
// For each person and each amount of calibration data (1, 2 or 4
// repetitions of every movement, about 50 seconds each), four decoders,
// all fed the same standardized Hudgins features:
//
//   LDA          linear discriminant analysis trained on the calibration
//                data -- what commercial pattern-recognition controllers run
//   NN, scratch  the same network as train.c's, trained from random
//                weights on the calibration data alone
//   NN, shared   train.c's model as is: only the feature standardization
//                comes from the calibration
//   NN, tuned    train.c's model fine-tuned on the calibration data with
//                intent_decoder_fine_tune() -- what the firmware does
//
// Accuracy is per 200ms window, over the windows prepare_data.c kept (see
// its comment). replay.c measures the decoder the way the wearer would
// feel it instead: continuously, transitions included, after voting.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_set.h"
#include "lda.h"
#include "ninapro.h"
#include "nn.h"

// Training from random weights needs more passes than fine-tuning does
// (chosen the same way, on the validation subjects).
#define SCRATCH_EPOCHS 40
#define SCRATCH_RATE 0.001f

#define NUM_CALIBRATIONS 3
static const unsigned calibration_reps[NUM_CALIBRATIONS] = {1u << 1, 1u << 1 | 1u << 3, 1u << 1 | 1u << 3 | 1u << 4 | 1u << 6};
static const char *calibration_names[NUM_CALIBRATIONS] = {"1 rep", "2 reps", "4 reps"};

enum { METHOD_LDA, METHOD_SCRATCH, METHOD_SHARED, METHOD_TUNED, NUM_METHODS };
static const char *method_names[NUM_METHODS] = {"LDA", "NN, scratch", "NN, shared", "NN, tuned"};

typedef struct {
  const char *name;
  int db, first, last;
} group_t;

static const group_t groups[] = {
  {"Intact (DB2 subjects 33-40)", 2, TEST_INTACT_FIRST, TEST_INTACT_LAST},
  {"Amputee (DB3 subjects 1-11)", 3, 1, NINAPRO_DB3_SUBJECTS},
};
#define NUM_GROUPS ((int)(sizeof(groups) / sizeof(groups[0])))

static int argmax(const float *out)
{
  int best = 0;
  for (int c = 1; c < INTENT_COUNT; c++)
    if (out[c] > out[best])
      best = c;
  return best;
}

// Fraction of `test` (standardized) classified correctly; adds each
// decision to `confusion` when it isn't NULL.
static float nn_accuracy(nn_t *nn, const rowset_t *test, int confusion[INTENT_COUNT][INTENT_COUNT])
{
  int correct = 0;
  for (int r = 0; r < test->rows; r++) {
    int p = argmax(nn_predict(nn, test->x + (size_t)r * EF_FEATURES));
    correct += p == test->label[r];
    if (confusion)
      confusion[test->label[r]][p]++;
  }
  return (float)correct / (float)test->rows;
}

static float lda_accuracy(const rowset_t *cal, const rowset_t *test)
{
  lda_t lda;
  float accuracy = 0.0f;
  if (lda_train(&lda, cal->x, cal->label, cal->rows, EF_FEATURES, INTENT_COUNT, 0.05f) == 0) {
    int correct = 0;
    for (int r = 0; r < test->rows; r++)
      correct += lda_predict(&lda, test->x + (size_t)r * EF_FEATURES) == test->label[r];
    accuracy = (float)correct / (float)test->rows;
  }
  lda_free(&lda);
  return accuracy;
}

static nn_t *new_network(void)
{
  nn_t *nn = nn_init();
  nn_add_layer(nn, LAYER_TYPE_INPUT, EF_FEATURES, ACTIVATION_FUNCTION_TYPE_NONE, NULL);
  nn_add_layer(nn, LAYER_TYPE_FC, 64, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
  nn_add_layer(nn, LAYER_TYPE_FC, 32, ACTIVATION_FUNCTION_TYPE_GELU, NULL);
  nn_add_layer(nn, LAYER_TYPE_OUTPUT, INTENT_COUNT, ACTIVATION_FUNCTION_TYPE_SOFTMAX, NULL);
  return nn;
}

static void print_confusion(const char *title, int confusion[INTENT_COUNT][INTENT_COUNT])
{
  printf("%s (rows: actual, columns: decoded, %% of row):\n", title);
  printf("  %-17s", "");
  for (int p = 0; p < INTENT_COUNT; p++)
    printf(" %6.6s", intent_names[p] + (p >= INTENT_WRIST_PRONATION ? 6 : 0));
  printf("\n");
  for (int a = 0; a < INTENT_COUNT; a++) {
    int total = 0;
    for (int p = 0; p < INTENT_COUNT; p++)
      total += confusion[a][p];
    printf("  %-17s", intent_names[a]);
    for (int p = 0; p < INTENT_COUNT; p++)
      printf(" %6.1f", total > 0 ? 100.0f * confusion[a][p] / total : 0.0f);
    printf("\n");
  }
  printf("\n");
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <model-file>\n", argv[0]);
    return 1;
  }
  const char *model_path = argv[1];
  nn_t *shared = nn_load_model(model_path);
  if (shared == NULL) {
    printf("Error: Could not load %s (run ./train first; inplace models can't be fine-tuned)\n", model_path);
    return 1;
  }
  if (shared->width[0] != EF_FEATURES || shared->width[shared->depth - 1] != INTENT_COUNT) {
    printf("Error: %s has %u inputs and %u outputs, expected %d and %d\n", model_path, shared->width[0],
           shared->width[shared->depth - 1], EF_FEATURES, INTENT_COUNT);
    return 1;
  }
  nn_free(shared);
  // Fixed, so the from-scratch networks' random starting weights -- and
  // so the numbers below -- are the same on every run for a given model.
  srand(1);
  features_t f;
  if (features_load("features.csv", &f) != 0)
    return 1;

  float scratch[EF_FEATURES + INTENT_COUNT];
  for (int g = 0; g < NUM_GROUPS; g++) {
    const group_t *grp = &groups[g];
    int subjects = grp->last - grp->first + 1;
    float mean[NUM_METHODS][NUM_CALIBRATIONS] = {{0}};
    float per_subject[NUM_CALIBRATIONS][NINAPRO_DB3_SUBJECTS];
    int confusion[INTENT_COUNT][INTENT_COUNT] = {{0}};
    for (int s = 0; s < subjects; s++) {
      int subject = grp->first + s;
      for (int c = 0; c < NUM_CALIBRATIONS; c++) {
        rowset_t cal = features_select(&f, grp->db, subject, calibration_reps[c]);
        rowset_t test = features_select(&f, grp->db, subject, TEST_REPS);
        intent_decoder_t id;
        intent_decoder_init(&id);
        intent_decoder_calibrate(&id, cal.x, cal.label, cal.rows);
        // LDA takes standardized rows; intent_decoder_fine_tune() takes raw
        // ones and standardizes them itself, as on the device.
        rowset_t cal_std = features_select(&f, grp->db, subject, calibration_reps[c]);
        rowset_standardize(&cal_std, &id);
        rowset_standardize(&test, &id);

        mean[METHOD_LDA][c] += lda_accuracy(&cal_std, &test);

        nn_t *nn = new_network();
        intent_decoder_fine_tune(&id, nn, cal.x, cal.label, cal.rows, SCRATCH_EPOCHS, SCRATCH_RATE, scratch);
        mean[METHOD_SCRATCH][c] += nn_accuracy(nn, &test, NULL);
        nn_free(nn);

        nn = nn_load_model(model_path);
        mean[METHOD_SHARED][c] += nn_accuracy(nn, &test, NULL);
        intent_decoder_fine_tune(&id, nn, cal.x, cal.label, cal.rows, ID_FINE_TUNE_EPOCHS, ID_FINE_TUNE_RATE, scratch);
        float tuned = nn_accuracy(nn, &test, c == 0 ? confusion : NULL);
        mean[METHOD_TUNED][c] += tuned;
        per_subject[c][s] = tuned;
        nn_free(nn);

        rowset_free(&cal);
        rowset_free(&cal_std);
        rowset_free(&test);
      }
      printf(".");
      fflush(stdout);
    }
    printf("\n%s: mean window accuracy on repetitions 2 and 5, by calibration data\n", grp->name);
    printf("  %-12s", "");
    for (int c = 0; c < NUM_CALIBRATIONS; c++)
      printf(" %8s", calibration_names[c]);
    printf("\n");
    for (int m = 0; m < NUM_METHODS; m++) {
      printf("  %-12s", method_names[m]);
      for (int c = 0; c < NUM_CALIBRATIONS; c++)
        printf("   %5.1f%%", 100.0f * mean[m][c] / subjects);
      printf("\n");
    }
    printf("  NN, tuned, per subject (1 / 2 / 4 reps):");
    for (int s = 0; s < subjects; s++)
      printf("%s s%d %.0f/%.0f/%.0f", s % 4 == 0 ? "\n   " : "  ", grp->first + s, 100.0f * per_subject[0][s],
             100.0f * per_subject[1][s], 100.0f * per_subject[2][s]);
    printf("\n\n");
    char title[128];
    snprintf(title, sizeof(title), "%s, NN tuned on 1 rep: confusion matrix", grp->name);
    print_confusion(title, confusion);
  }
  features_free(&f);
  return 0;
}
