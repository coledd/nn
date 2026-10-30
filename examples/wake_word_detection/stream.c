/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Tests trained wake-word models the way a listener actually runs: on a
// continuous stream, not on pre-cut clips. test.c scores each one-second
// clip on its own, so it can't say how often a device wakes up per hour,
// or whether a wake word that falls across two windows still gets caught.
//
// The stream is every clip in one of the dataset's speaker-separated
// split lists -- testing_list.txt (195 of its clips "marvin") or, for
// choosing a threshold without touching the test speakers,
// validation_list.txt -- shuffled and laid end to end with a random gap of
// GAP_MIN_MS-GAP_MAX_MS of background between each pair, over a background
// bed looped from the same split's stretch of the dataset's
// _background_noise_ recordings (as prepare_data.c divides them: the last
// 10% of each for test, the 10% before that for validation) mixed in at a
// chosen signal-to-noise ratio. A one-second window slides along it every STEP_MS; each window
// gets exactly the features a clip gets (audio_features_extract_sequence(),
// normalization included) and goes through wake_word_probability().
//
// A window at or above the threshold triggers, and then the detector holds
// off for REFRACTORY_MS, as a real one does so that one utterance wakes it
// once. A trigger whose window overlaps a "marvin" clip at all detects it
// -- the device woke while, or just after, the word was spoken -- and any
// other trigger is a false wake-up. Every model named on the command line
// is scored on the same stream, at thresholds from 0.05 to 0.995 (finer
// near the top, where a detector that must rarely wake has to sit).
//
// Two second-stage checks are scored alongside, both from the same
// per-window probabilities. Each model is scored three times: waking on
// one window over the threshold, and only once two or three consecutive
// windows (100 ms apart) are -- a spoken word keeps scoring high as the
// window slides across it, a chance match often doesn't. And a model
// argument of the form "a.txt+b.txt" means two models that must both
// agree: its score for a window is the smaller of their two, which on a
// device is the first model waking the second to check.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_features.h"
#include "nn.h"
#include "wav_reader.h"
#include "wake_word_data.h"

#define WAKE_WORD "marvin"
#define GAP_MIN_MS 250
#define GAP_MAX_MS 1500
#define STEP_MS 100        // window hop: 10 windows per second
#define REFRACTORY_MS 1000 // no second trigger within a second of the first
#define MAX_MODELS 16
#define MAX_CONSECUTIVE 3 // score 1, 2 and 3 consecutive windows
#define MAX_PATH_LEN 1024

#define STEP_SAMPLES (SAMPLE_RATE * STEP_MS / 1000)
#define REFRACTORY_STEPS (REFRACTORY_MS / STEP_MS)

// A small, seeded generator, so a given seed always builds the same stream
// (rand() differs between C libraries).
static unsigned long long rng_state = 1;
static unsigned rng_next(void)
{
  rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (unsigned)(rng_state >> 33);
}

typedef struct {
  long start, end; // sample range in the stream
} interval_t;

static const float thresholds[] = {0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f, 0.45f, 0.50f,
                                   0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f,
                                   0.96f, 0.97f, 0.98f, 0.99f, 0.995f};
#define NUM_THRESHOLDS ((int)(sizeof(thresholds) / sizeof(thresholds[0])))

// Appends the test (or validation) stretch of every background recording
// to `bed` (growing it as needed); returns its new length in samples.
static long load_background(const char *dataset_dir, int validation, float **bed)
{
  static const char *names[] = {"doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav",
                                "pink_noise.wav", "running_tap.wav", "white_noise.wav"};
  long len = 0;
  for (size_t n = 0; n < sizeof(names) / sizeof(names[0]); n++) {
    char path[MAX_PATH_LEN];
    snprintf(path, sizeof(path), "%s/_background_noise_/%s", dataset_dir, names[n]);
    wav_t *wav = wav_load(path);
    if (wav == NULL) {
      fprintf(stderr, "Error: could not load %s.\n", path);
      exit(1);
    }
    // The same stretches prepare_data.c's write_background_noise() puts in
    // validation.csv and test.csv.
    int num_segments = wav->num_samples / CLIP_SAMPLES;
    long from = (long)(num_segments * (validation ? 8 : 9) / 10) * CLIP_SAMPLES;
    long to = (long)(validation ? num_segments * 9 / 10 : num_segments) * CLIP_SAMPLES;
    *bed = realloc(*bed, (size_t)(len + to - from) * sizeof(float));
    memcpy(*bed + len, wav->samples + from, (size_t)(to - from) * sizeof(float));
    len += to - from;
    wav_free(wav);
  }
  return len;
}

int main(int argc, char *argv[])
{
  if (argc < 5 || (strcmp(argv[2], "test") != 0 && strcmp(argv[2], "validation") != 0)) {
    printf("Usage: %s <dataset-dir> <test|validation> <snr-db> <model>[+<model>] [...]\n", argv[0]);
    printf("  Builds a continuous stream from one split's clips over that split's background noise,\n");
    printf("  mixed at <snr-db> (speech power over background power), and reports each model's detection\n");
    printf("  rate and false wake-ups per hour across thresholds, waking on 1, 2 or 3 consecutive windows.\n");
    printf("  a.txt+b.txt scores two models that must both agree. Set SEED in the environment to vary the stream.\n");
    return 1;
  }
  const char *dataset_dir = argv[1];
  const int validation = strcmp(argv[2], "validation") == 0;
  const float snr_db = strtof(argv[3], NULL);
  int num_models = argc - 4;
  if (num_models > MAX_MODELS) {
    fprintf(stderr, "Error: at most %d models.\n", MAX_MODELS);
    return 1;
  }
  // models[m][1] is NULL unless argument m names a second, confirming model.
  nn_t *models[MAX_MODELS][2] = { { NULL } };
  for (int m = 0; m < num_models; m++) {
    char spec[MAX_PATH_LEN];
    snprintf(spec, sizeof(spec), "%s", argv[4 + m]);
    char *second = strchr(spec, '+');
    if (second)
      *second++ = '\0';
    models[m][0] = nn_load_model(spec);
    if (second)
      models[m][1] = nn_load_model(second);
    if (models[m][0] == NULL || (second && models[m][1] == NULL)) {
      fprintf(stderr, "Error: could not load model %s.\n", argv[4 + m]);
      return 1;
    }
  }
  if (getenv("SEED"))
    rng_state = strtoull(getenv("SEED"), NULL, 10);

  // The test clips, in a shuffled order.
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "%s/%s_list.txt", dataset_dir, validation ? "validation" : "testing");
  FILE *list = fopen(path, "r");
  if (list == NULL) {
    fprintf(stderr, "Error: could not open %s.\n", path);
    return 1;
  }
  int num_clips = 0, capacity = 1024;
  char **clips = malloc((size_t)capacity * sizeof(char *));
  char line[MAX_PATH_LEN];
  while (fgets(line, sizeof(line), list) != NULL) {
    line[strcspn(line, "\r\n")] = '\0';
    if (line[0] == '\0')
      continue;
    if (num_clips == capacity)
      clips = realloc(clips, (size_t)(capacity *= 2) * sizeof(char *));
    clips[num_clips++] = strdup(line);
  }
  fclose(list);
  for (int i = num_clips - 1; i > 0; i--) {
    int j = (int)(rng_next() % (unsigned)(i + 1));
    char *t = clips[i];
    clips[i] = clips[j];
    clips[j] = t;
  }

  // Lay the clips end to end with random gaps, starting and ending with a
  // second of background, and note where each "marvin" lands.
  long stream_len = SAMPLE_RATE, stream_cap = 1L << 24;
  float *stream = calloc((size_t)stream_cap, sizeof(float));
  interval_t *wake = malloc((size_t)num_clips * sizeof(interval_t));
  int num_wake = 0;
  double speech_power = 0.0;
  long speech_samples = 0;
  for (int i = 0; i < num_clips; i++) {
    snprintf(path, sizeof(path), "%s/%s", dataset_dir, clips[i]);
    wav_t *wav = wav_load(path);
    if (wav == NULL)
      continue;
    long gap = (long)(GAP_MIN_MS + rng_next() % (GAP_MAX_MS - GAP_MIN_MS + 1)) * SAMPLE_RATE / 1000;
    long start = stream_len + (i > 0 ? gap : 0);
    long end = start + wav->num_samples;
    while (end + SAMPLE_RATE > stream_cap) {
      stream = realloc(stream, (size_t)(stream_cap * 2) * sizeof(float));
      memset(stream + stream_cap, 0, (size_t)stream_cap * sizeof(float));
      stream_cap *= 2;
    }
    memcpy(stream + start, wav->samples, (size_t)wav->num_samples * sizeof(float));
    for (int k = 0; k < wav->num_samples; k++)
      speech_power += (double)wav->samples[k] * wav->samples[k];
    speech_samples += wav->num_samples;
    if (strncmp(clips[i], WAKE_WORD "/", strlen(WAKE_WORD) + 1) == 0)
      wake[num_wake++] = (interval_t){start, end};
    stream_len = end;
    wav_free(wav);
  }
  stream_len += SAMPLE_RATE;
  speech_power /= (double)speech_samples;

  // Mix in the background bed, looped, at the requested SNR.
  float *bed = NULL;
  long bed_len = load_background(dataset_dir, validation, &bed);
  double bed_power = 0.0;
  for (long k = 0; k < bed_len; k++)
    bed_power += (double)bed[k] * bed[k];
  bed_power /= (double)bed_len;
  const float gain = (float)sqrt(speech_power / bed_power / pow(10.0, snr_db / 10.0));
  for (long k = 0; k < stream_len; k++)
    stream[k] += gain * bed[k % bed_len];

  const double hours = (double)stream_len / SAMPLE_RATE / 3600.0;
  printf("Stream: %.2f hours, %d %s clips (%d \"%s\"), gaps %d-%d ms, background at %.0f dB SNR",
         hours, num_clips, argv[2], num_wake, WAKE_WORD, GAP_MIN_MS, GAP_MAX_MS, snr_db);
  printf(" (%.1f s of %s background, looped), seed %llu\n", (double)bed_len / SAMPLE_RATE, argv[2],
         getenv("SEED") ? strtoull(getenv("SEED"), NULL, 10) : 1ULL);
  printf("Window: 1 s every %d ms, %d ms hold-off after each trigger\n\n", STEP_MS, REFRACTORY_MS);

  // Every model's probability for every window.
  audio_features_t *af = audio_features_init(SAMPLE_RATE, FRAME_LEN, NUM_MEL_BINS);
  long num_windows = (stream_len - CLIP_SAMPLES) / STEP_SAMPLES + 1;
  float *prob = malloc((size_t)num_models * num_windows * sizeof(float));
  static float features[SEQUENCE_LEN * NUM_MEL_BINS];
  for (long w = 0; w < num_windows; w++) {
    audio_features_extract_sequence(af, stream + w * STEP_SAMPLES, CLIP_SAMPLES, HOP_LEN, features);
    for (int m = 0; m < num_models; m++) {
      float p = wake_word_probability(models[m][0], features);
      if (models[m][1]) {
        float q = wake_word_probability(models[m][1], features);
        p = q < p ? q : p;
      }
      prob[(size_t)m * num_windows + w] = p;
    }
  }

  // Score every model at every threshold.
  int *detected = malloc((size_t)num_wake * sizeof(int));
  for (int scored = 0; scored < num_models * MAX_CONSECUTIVE; scored++) {
    const int m = scored / MAX_CONSECUTIVE, consecutive = scored % MAX_CONSECUTIVE + 1;
    const float *p = prob + (size_t)m * num_windows;
    printf("%s, %d consecutive window%s\n", argv[4 + m], consecutive, consecutive > 1 ? "s" : "");
    printf("threshold, detected %%, missed %%, false wake-ups, per hour\n");
    for (int t = 0; t < NUM_THRESHOLDS; t++) {
      const float threshold = thresholds[t];
      memset(detected, 0, (size_t)num_wake * sizeof(int));
      int false_wakeups = 0, next_wake = 0;
      wake_word_detector_t detector;
      wake_word_detector_init(&detector, threshold, consecutive, REFRACTORY_STEPS - 1);
      for (long w = 0; w < num_windows; w++) {
        if (!wake_word_detector_step(&detector, p[w]))
          continue;
        long ws = w * STEP_SAMPLES, we = ws + CLIP_SAMPLES;
        // The wake-word clips are in stream order: skip those that ended
        // before this window, then check the ones it overlaps.
        while (next_wake < num_wake && wake[next_wake].end <= ws)
          next_wake++;
        int overlaps_any = 0;
        for (int k = next_wake; k < num_wake && wake[k].start < we; k++) {
          detected[k] = 1;
          overlaps_any = 1;
        }
        if (!overlaps_any)
          false_wakeups++;
      }
      int hits = 0;
      for (int k = 0; k < num_wake; k++)
        hits += detected[k];
      printf("%.3f, %.1f, %.1f, %d, %.2f\n", threshold, 100.0 * hits / num_wake,
             100.0 * (num_wake - hits) / num_wake, false_wakeups, false_wakeups / hours);
    }
    printf("\n");
  }

  free(detected);
  free(prob);
  audio_features_free(af);
  free(bed);
  free(wake);
  free(stream);
  for (int i = 0; i < num_clips; i++)
    free(clips[i]);
  free(clips);
  for (int m = 0; m < num_models; m++) {
    nn_free(models[m][0]);
    nn_free(models[m][1]);
  }
  return 0;
}
