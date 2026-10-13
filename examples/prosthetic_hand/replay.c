/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Replays recorded sessions through intent_decoder.[ch] sample by sample,
// exactly as the prosthesis would run them, and measures the decoder the
// way its wearer would feel it.
//
// For each person: the model from train.c is loaded, the decoder is
// calibrated on their first repetition of every movement (about a minute
// of prompted movements) and the model fine-tuned on the same data with
// intent_decoder_fine_tune() -- the firmware's calibration routine. Then
// the whole recording streams through intent_decoder_add_sample() at
// 2kHz, and every decision during repetitions 2 and 5 is scored. Unlike
// test.c's window accuracy, this includes every window: the onset and end
// of each movement, the ones that straddle a change, and rest right after
// a movement. Four numbers, for each of several ways of turning the
// model's per-window outputs into motor commands (see stream_names below) --
// the last of them is what intent_decoder.[ch] actually does:
//
//   response   time from a movement's onset (as Ninapro's refined labels
//              mark it) to the first decision naming it
//   correct    share of decisions during the movement that name it
//   wrong      share of decisions during the movement that name a
//              *different* movement -- the hand doing something the wearer
//              didn't ask for, which is worse than doing nothing
//   drift      share of decisions during rest that name a movement
//
// Usage:
//   ./replay model.bin           all 19 test subjects, summarized by group
//   ./replay model.bin 3 1       one subject (DB3 subject 1): their numbers
//                                plus a timeline of each movement

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "feature_set.h"
#include "intent_decoder.h"
#include "ninapro.h"
#include "nn.h"

#define DATASET_DIR "dataset"
#define CALIBRATION_REPETITION 1
#define TIMELINE_REPETITION 2
#define TIMELINE_LEAD 20 // decisions shown before each movement's onset (1s)

static const char intent_chars[INTENT_COUNT] = {'.', 'O', 'C', 'P', 'S', 'F', 'E'};

typedef struct {
  double response_ms;   // summed over movements where the intent was named at all
  int responded, movements;
  int correct, wrong, moving_decisions;
  int drift, rest_decisions;
} stats_t;

// The decision streams scored, all from the same model outputs: the raw
// classification, voting alone, rejection alone at three thresholds, and
// the decoder's own output (rejection at ID_REJECT_BELOW, then voting).
enum { STREAM_RAW, STREAM_VOTED, STREAM_REJECT_70, STREAM_REJECT_80, STREAM_REJECT_90, STREAM_DECODER, NUM_STREAMS };
static const char *stream_names[NUM_STREAMS] = {
  "raw", "voted", "rejected < 0.7", "rejected < 0.8", "rejected < 0.9", "decoder (rejected < 0.8, voted)"
};
static const float stream_threshold[NUM_STREAMS] = {0.0f, 0.0f, 0.7f, 0.8f, 0.9f, 0.0f};

typedef struct {
  stats_t stream[NUM_STREAMS];
} result_t;

static void add_stats(stats_t *a, const stats_t *b)
{
  a->response_ms += b->response_ms;
  a->responded += b->responded;
  a->movements += b->movements;
  a->correct += b->correct;
  a->wrong += b->wrong;
  a->moving_decisions += b->moving_decisions;
  a->drift += b->drift;
  a->rest_decisions += b->rest_decisions;
}

static void print_result(const result_t *r)
{
  printf("  %-32s %9s %7s %8s %6s %6s\n", "", "response", "named", "correct", "wrong", "drift");
  for (int k = 0; k < NUM_STREAMS; k++) {
    const stats_t *s = &r->stream[k];
    char named[16];
    snprintf(named, sizeof(named), "%d/%d", s->responded, s->movements);
    printf("  %-32s %7.0fms %7s %7.1f%% %5.1f%% %5.1f%%\n", stream_names[k],
           s->responded > 0 ? s->response_ms / s->responded : 0.0, named,
           100.0 * s->correct / (s->moving_decisions > 0 ? s->moving_decisions : 1),
           100.0 * s->wrong / (s->moving_decisions > 0 ? s->moving_decisions : 1),
           100.0 * s->drift / (s->rest_decisions > 0 ? s->rest_decisions : 1));
  }
}

// Scores one decision made at sample `i`. `onset` is the sample at which
// the current movement began (or -1 during rest), and `named` whether it
// has been named yet; both are kept per decision stream.
static void score(stats_t *s, const ninapro_recording_t *rec, int i, int decision, int *named, int onset)
{
  int intent = rec->intent[i];
  if (intent > INTENT_REST) {
    s->moving_decisions++;
    if (decision == intent) {
      s->correct++;
      if (!*named) {
        *named = 1;
        s->responded++;
        s->response_ms += (i - onset) * 1000.0 / EF_SAMPLE_RATE;
      }
    } else if (decision != INTENT_REST) {
      s->wrong++;
    }
  } else if (intent == INTENT_REST && rec->next_movement[i] > INTENT_REST) {
    s->rest_decisions++;
    if (decision != INTENT_REST)
      s->drift++;
  }
}

static int replay_subject(const char *model_path, int db, int subject, bool timeline, result_t *result)
{
  char path[512];
  ninapro_path(path, sizeof(path), DATASET_DIR, db, subject);
  ninapro_recording_t rec;
  if (ninapro_load(path, &rec) != 0)
    return -1;

  // Calibration: the windows of repetition 1 that prepare_data.c would
  // have kept, through the same streaming feature extractor.
  int capacity = 4096, n = 0, rest_windows = 0;
  float *cal_x = (float *)malloc((size_t)capacity * EF_FEATURES * sizeof(float));
  int *cal_label = (int *)malloc((size_t)capacity * sizeof(int));
  emg_features_t ef;
  emg_features_init(&ef);
  float features[EF_FEATURES];
  for (int i = 0; i < rec.samples; i++) {
    if (!emg_features_add_sample(&ef, rec.emg + (size_t)i * EF_CHANNELS, features))
      continue;
    int intent = ninapro_window_intent(&rec, i);
    if (intent < 0 || rec.repetition[i] != CALIBRATION_REPETITION ||
        (intent == INTENT_REST && (rest_windows++ % NINAPRO_REST_THINNING) != 0))
      continue;
    if (n == capacity) {
      capacity *= 2;
      cal_x = (float *)realloc(cal_x, (size_t)capacity * EF_FEATURES * sizeof(float));
      cal_label = (int *)realloc(cal_label, (size_t)capacity * sizeof(int));
    }
    if (cal_x == NULL || cal_label == NULL) {
      fprintf(stderr, "Error: Out of memory.\n");
      exit(1);
    }
    memcpy(cal_x + (size_t)n * EF_FEATURES, features, sizeof(features));
    cal_label[n++] = intent;
  }
  nn_t *nn = nn_load_model(model_path);
  if (nn == NULL) {
    fprintf(stderr, "Error: Could not load %s (run ./train first; inplace models can't be fine-tuned)\n", model_path);
    exit(1);
  }
  static intent_decoder_t id;
  intent_decoder_init(&id);
  intent_decoder_calibrate(&id, cal_x, cal_label, n);
  float scratch[EF_FEATURES + INTENT_COUNT];
  intent_decoder_fine_tune(&id, nn, cal_x, cal_label, n, ID_FINE_TUNE_EPOCHS, ID_FINE_TUNE_RATE, scratch);
  free(cal_x);
  free(cal_label);

  // Control: the whole recording, continuously.
  memset(result, 0, sizeof(*result));
  int onset = -1, named[NUM_STREAMS] = {0};
  id_vote_t raw_vote;
  memset(&raw_vote, 0, sizeof(raw_vote));
  // Timeline: one character per decision for each movement in the
  // timeline repetition, from TIMELINE_LEAD decisions before its onset.
  char actual_line[INTENT_COUNT][256], decoded_line[INTENT_COUNT][256];
  int line_len[INTENT_COUNT] = {0};
  char history_actual[TIMELINE_LEAD], history_decoded[TIMELINE_LEAD];
  int history = 0;
  int recording = -1; // intent whose timeline is being recorded
  int tail = 0;        // decisions recorded since that movement ended
  for (int i = 0; i < rec.samples; i++) {
    int intent = rec.intent[i];
    if (intent > INTENT_REST && (i == 0 || rec.intent[i - 1] != intent)) {
      onset = i;
      memset(named, 0, sizeof(named));
      if (timeline && rec.repetition[i] == TIMELINE_REPETITION && line_len[intent] == 0) {
        recording = intent;
        int k = history < TIMELINE_LEAD ? history : TIMELINE_LEAD;
        for (int j = 0; j < k; j++) {
          actual_line[intent][j] = history_actual[(history - k + j) % TIMELINE_LEAD];
          decoded_line[intent][j] = history_decoded[(history - k + j) % TIMELINE_LEAD];
        }
        line_len[intent] = k;
        tail = 0;
      }
    }
    id_report_t report;
    if (!intent_decoder_add_sample(&id, nn, rec.emg + (size_t)i * EF_CHANNELS, &report))
      continue;
    int decision[NUM_STREAMS];
    decision[STREAM_RAW] = report.raw;
    decision[STREAM_VOTED] = id_vote_add(&raw_vote, report.raw);
    for (int k = STREAM_REJECT_70; k <= STREAM_REJECT_90; k++)
      decision[k] = report.confidence >= stream_threshold[k] ? report.raw : INTENT_REST;
    decision[STREAM_DECODER] = report.intent;
    if ((TEST_REPS >> rec.repetition[i]) & 1u)
      for (int k = 0; k < NUM_STREAMS; k++)
        score(&result->stream[k], &rec, i, decision[k], &named[k], onset);
    char a = intent >= 0 ? intent_chars[intent] : ' ';
    char d = intent_chars[report.intent];
    history_actual[history % TIMELINE_LEAD] = a;
    history_decoded[history % TIMELINE_LEAD] = d;
    history++;
    if (recording >= 0) {
      // Through the movement, then TIMELINE_LEAD decisions (1s) of what follows.
      if (intent != recording)
        tail++;
      int len = line_len[recording];
      if (len < 255 && tail <= TIMELINE_LEAD) {
        actual_line[recording][len] = a;
        decoded_line[recording][len] = d;
        line_len[recording]++;
      } else {
        recording = -1;
      }
    }
  }
  // Every movement in the scored repetitions counts, named or not.
  for (int i = 1; i < rec.samples; i++) {
    if (rec.intent[i] > INTENT_REST && rec.intent[i - 1] != rec.intent[i] && ((TEST_REPS >> rec.repetition[i]) & 1u))
      for (int k = 0; k < NUM_STREAMS; k++)
        result->stream[k].movements++;
  }

  if (timeline) {
    printf("Repetition %d of each movement, one character per 50ms decoder output, from 1s before onset:\n",
           TIMELINE_REPETITION);
    printf("  . rest  O open  C close  P pronation  S supination  F flexion  E extension\n\n");
    for (int k = INTENT_HAND_OPEN; k < INTENT_COUNT; k++) {
      printf("  %-17s actual   %.*s\n", intent_names[k], line_len[k], actual_line[k]);
      printf("  %-17s decoded  %.*s\n\n", "", line_len[k], decoded_line[k]);
    }
  }
  nn_free(nn);
  ninapro_free(&rec);
  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 2 && argc != 4) {
    printf("Usage: %s <model-file> [<db> <subject>]\n", argv[0]);
    printf("  <db> <subject> : replay one subject (e.g. 3 1 for DB3 subject 1) with a timeline\n");
    return 1;
  }
  srand(1);
  if (argc == 4) {
    result_t r;
    if (replay_subject(argv[1], atoi(argv[2]), atoi(argv[3]), true, &r) != 0)
      return 1;
    printf("Repetitions 2 and 5:\n");
    print_result(&r);
    return 0;
  }
  const struct { const char *name; int db, first, last; } groups[] = {
    {"Intact (DB2 subjects 33-40)", 2, TEST_INTACT_FIRST, TEST_INTACT_LAST},
    {"Amputee (DB3 subjects 1-11)", 3, 1, NINAPRO_DB3_SUBJECTS},
  };
  for (int g = 0; g < 2; g++) {
    result_t total;
    memset(&total, 0, sizeof(total));
    for (int s = groups[g].first; s <= groups[g].last; s++) {
      result_t r;
      if (replay_subject(argv[1], groups[g].db, s, false, &r) != 0)
        return 1;
      for (int k = 0; k < NUM_STREAMS; k++)
        add_stats(&total.stream[k], &r.stream[k]);
      printf(".");
      fflush(stdout);
    }
    printf("\n%s, calibrated on repetition %d, continuous decoding of repetitions 2 and 5:\n", groups[g].name,
           CALIBRATION_REPETITION);
    print_result(&total);
    printf("\n");
  }
  return 0;
}
