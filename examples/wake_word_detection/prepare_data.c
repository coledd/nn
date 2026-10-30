/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Walks an extracted copy of the Google Speech Commands dataset
// (https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz,
// CC BY 4.0 -- one subdirectory per spoken word, plus a
// "_background_noise_" directory of longer ambient recordings) and writes
// one flattened row per clip: SEQUENCE_LEN * NUM_MEL_BINS log-mel energies
// (see wake_word_data.h and ../../audio_features.h), followed by a single
// 0/1 label -- 1 for the wake word, 0 for everything else (other spoken
// words, plus 1-second chunks cut from the dataset's own background-noise
// recordings, so the model also learns to stay quiet through
// silence/ambient noise, not just other speech).
// train.csv also gets one-second windows cut from short runs of training
// clips laid end to end over background noise (see
// write_stream_windows()), so the model learns what a sliding window on a
// live stream sees, not just one word alone in its second.
//
// Rows go straight into train.csv, validation.csv and test.csv, following
// the dataset's own validation_list.txt and testing_list.txt rather than a
// random shuffle (../../split.py). Those lists assign every clip from one
// speaker to the same split (by a hash of the speaker ID; see the dataset's
// README), so test.csv measures the model on voices it never trained on --
// what a product shipped to new users actually faces. A random per-clip
// split lets the same speaker's "marvin" land in both train and test, and
// scores optimistically.
//
// This is the one genuinely new tool in this example. Unlike
// examples/character_recognition's samples.csv (MNIST's pixels, flattened
// and thresholded -- which has no bearing on what a deployed target
// does), flattening audio into feature rows via audio_features.[ch] IS
// exactly what a deployed target's firmware has to do against its own live
// mic buffer, so that step has to be real, working code here rather than a
// downloaded shortcut. Everything downstream of this file -- data_prep.[ch],
// the CSV loading in train_cnn.c/train_gru.c/test.c -- is the same
// general-purpose CSV machinery examples/character_recognition already
// established.
//
// The wake word is a command-line argument, not hardcoded: "marvin" (this
// example's default) is one of two words the dataset's own authors
// included specifically so they could double as a pretend wake word
// (Warden, 2018, "Speech Commands: A Dataset for Limited-Vocabulary Speech
// Recognition", https://arxiv.org/abs/1804.03209), but swapping to a
// different word later needs no code changes, only a different argument.

#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_features.h"
#include "wake_word_data.h"
#include "wav_reader.h"

#define MAX_PATH_LEN 4096
#define BACKGROUND_NOISE_DIR "_background_noise_"

// train.csv gets this many times the negatives validation.csv/test.csv do
// (relative to each split's wake-word count): more of the ~84,000
// other-word clips the training speakers recorded, and overlapping windows
// over the background noise. A wake-word detector's hardest job is staying
// quiet through everything else, and the balanced splits use under 2% of
// what's available. To keep train.csv balanced anyway, each training
// wake-word clip is written once as recorded plus TRAIN_NEGATIVE_RATIO - 1
// times shifted in time (see MAX_SHIFT_SAMPLES).
#define TRAIN_NEGATIVE_RATIO 5
// Largest time shift, either way, of an augmented wake-word copy: 100 ms.
// Speech Commands clips aren't aligned -- the word sits wherever the
// speaker said it within the second -- and a live microphone window
// won't be either.
#define MAX_SHIFT_SAMPLES (SAMPLE_RATE / 10)

// Streaming windows (see write_stream_windows()): train.csv also gets
// one-second windows cut from short runs of training clips laid end to
// end, the way a listener's sliding window sees speech, rather than one
// word sitting alone in its second. STREAM_POSITIVES_PER_CLIP windows per
// training wake-word clip, each containing all of the spoken word, and
// STREAM_NEGATIVE_RATIO times as many windows from runs of other words.
// The gaps between clips match stream.c's, and the background is the
// training stretch of the noise recordings at STREAM_SNR_MIN_DB to
// STREAM_SNR_MAX_DB, from noisy to nearly silent.
#define STREAM_POSITIVES_PER_CLIP 2
#define STREAM_NEGATIVE_RATIO 5
#define STREAM_GAP_MIN_MS 250
#define STREAM_GAP_MAX_MS 1500
#define STREAM_SNR_MIN_DB 5
#define STREAM_SNR_MAX_DB 60
// Each window is cut from three clips and two gaps: at most 3 s of clips
// and 3 s of gaps.
#define STREAM_RUN_SAMPLES (6 * SAMPLE_RATE)

typedef enum { SPLIT_TRAIN = 0, SPLIT_VALIDATION, SPLIT_TEST, NUM_SPLITS } split_t;
static const char *split_paths[NUM_SPLITS] = { "train.csv", "validation.csv", "test.csv" };

// One of the dataset's split lists (validation_list.txt or
// testing_list.txt): every "word/file.wav" path in it, sorted for bsearch().
typedef struct {
  char **paths;
  int count;
} split_list_t;

static int compare_strings(const void *a, const void *b)
{
  return strcmp(*(char *const *)a, *(char *const *)b);
}

// Loads one split list, returning 1 on success. A missing list is an
// error, not an empty split: without it every clip would silently land in
// train.csv.
static int split_list_load(split_list_t *list, const char *dataset_dir, const char *name)
{
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "%s/%s", dataset_dir, name);
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "Error: could not open %s.\n", path);
    return 0;
  }
  int capacity = 1024;
  list->paths = malloc((size_t)capacity * sizeof(char *));
  list->count = 0;
  char line[MAX_PATH_LEN];
  while (fgets(line, sizeof(line), file) != NULL) {
    line[strcspn(line, "\r\n")] = '\0';
    if (line[0] == '\0')
      continue;
    if (list->count == capacity) {
      capacity *= 2;
      list->paths = realloc(list->paths, (size_t)capacity * sizeof(char *));
    }
    list->paths[list->count++] = strdup(line);
  }
  fclose(file);
  qsort(list->paths, (size_t)list->count, sizeof(char *), compare_strings);
  return 1;
}

static int split_list_contains(const split_list_t *list, const char *key)
{
  return bsearch(&key, list->paths, (size_t)list->count, sizeof(char *), compare_strings) != NULL;
}

static void split_list_free(split_list_t *list)
{
  for (int i = 0; i < list->count; i++)
    free(list->paths[i]);
  free(list->paths);
}

static split_list_t validation_list, testing_list;

// Which split the clip "word/file.wav" belongs to, per the dataset's lists.
static split_t split_of(const char *word, const char *file_name)
{
  char key[MAX_PATH_LEN];
  snprintf(key, sizeof(key), "%s/%s", word, file_name);
  if (split_list_contains(&testing_list, key))
    return SPLIT_TEST;
  if (split_list_contains(&validation_list, key))
    return SPLIT_VALIDATION;
  return SPLIT_TRAIN;
}

// Copies `wav`'s samples into a fixed CLIP_SAMPLES-length buffer, padding
// with digital silence if the clip is shorter, or truncating if longer --
// every row this tool writes needs the same SEQUENCE_LEN, which
// audio_features_extract_sequence() only guarantees for a fixed-length
// input.
// `shift` moves the audio later (positive) or earlier (negative) by that
// many samples, filling the gap with silence.
static void fit_to_clip_length(const wav_t *wav, float *clip, int shift)
{
  for (int i = 0; i < CLIP_SAMPLES; i++) {
    int src = i - shift;
    clip[i] = src >= 0 && src < wav->num_samples ? wav->samples[src] : 0.0f;
  }
}

// Extracts one CLIP_SAMPLES-long clip's log-mel feature sequence and
// appends it to `file` as one CSV row (SEQUENCE_LEN * NUM_MEL_BINS inputs,
// then `label`).
static void write_row(FILE *file, audio_features_t *af, const float *clip, int label)
{
  static float sequence[SEQUENCE_LEN * NUM_MEL_BINS];
  audio_features_extract_sequence(af, clip, CLIP_SAMPLES, HOP_LEN, sequence);
  for (int i = 0; i < SEQUENCE_LEN * NUM_MEL_BINS; i++)
    fprintf(file, "%.6f,", sequence[i]);
  fprintf(file, "%d\n", label);
}

// Loads one WAV file and writes its feature row, plus `extra_copies` more
// rows of it shifted by a random amount up to MAX_SHIFT_SAMPLES either way.
// Returns the number of rows written. Silently skips (returns 0 for) a file
// that fails to load -- real-world datasets this size have a handful of
// malformed/truncated clips, not worth aborting the whole pass over.
static int write_wav_row(FILE *file, audio_features_t *af, const char *path, int label, int extra_copies)
{
  wav_t *wav = wav_load(path);
  if (wav == NULL)
    return 0;
  static float clip[CLIP_SAMPLES];
  for (int k = 0; k <= extra_copies; k++) {
    int shift = k == 0 ? 0 : rand() % (2 * MAX_SHIFT_SAMPLES + 1) - MAX_SHIFT_SAMPLES;
    fit_to_clip_length(wav, clip, shift);
    write_row(file, af, clip, label);
  }
  wav_free(wav);
  return 1 + extra_copies;
}

// Writes the ".wav" files directly inside `dataset_dir`/`word`, each to its
// own split's file in `files`, with the given `label`, up to
// `max_count[split]` clips per split (each training clip with
// `train_extra_copies` time-shifted copies; see write_wav_row()), in whatever order readdir() returns
// them -- good enough for a roughly-balanced negative class, since these
// are already independent per-utterance recordings with no ordering to
// correct for. Adds the number of rows written to each split to
// `written[split]`.
static void write_word_dir(FILE *files[NUM_SPLITS], audio_features_t *af, const char *dataset_dir,
                           const char *word, const int max_count[NUM_SPLITS], int label,
                           int train_extra_copies, int written[NUM_SPLITS])
{
  char dir_path[MAX_PATH_LEN];
  snprintf(dir_path, sizeof(dir_path), "%s/%s", dataset_dir, word);
  DIR *dir = opendir(dir_path);
  if (dir == NULL) {
    fprintf(stderr, "Warning: could not open %s, skipping.\n", dir_path);
    return;
  }
  int count[NUM_SPLITS] = { 0 };
  int rows[NUM_SPLITS] = { 0 };
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    size_t len = strlen(entry->d_name);
    if (len < 5 || strcmp(entry->d_name + len - 4, ".wav") != 0)
      continue;
    split_t split = split_of(word, entry->d_name);
    if (count[split] >= max_count[split])
      continue;
    char path[MAX_PATH_LEN];
    snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
    int extra_copies = split == SPLIT_TRAIN ? train_extra_copies : 0;
    int n = write_wav_row(files[split], af, path, label, extra_copies);
    count[split] += n > 0;
    rows[split] += n;
  }
  closedir(dir);
  for (int i = 0; i < NUM_SPLITS; i++)
    written[i] += rows[i];
}

// Chops every "*.wav" file in `dir_path` (the dataset's
// "_background_noise_" directory -- a handful of longer ambient-noise
// recordings, not per-word utterances) into non-overlapping CLIP_SAMPLES
// segments, each written as its own label-0 row: silence/noise negatives,
// standard keyword-spotting practice alongside the "other words" negatives
// written by write_word_dir(). The dataset's split lists don't cover these
// recordings, so each one is split by position instead: its first 80% of
// segments to train, the next 10% to validation, the last 10% to test --
// contiguous stretches, so no test segment sits right beside a training
// one. The training stretch is cut into TRAIN_NEGATIVE_RATIO times as many
// overlapping windows. Adds the number of rows written to each split to
// `written[split]`.
static void write_background_noise(FILE *files[NUM_SPLITS], audio_features_t *af, const char *dir_path,
                                   int written[NUM_SPLITS])
{
  DIR *dir = opendir(dir_path);
  if (dir == NULL) {
    fprintf(stderr, "Warning: could not open %s, skipping background noise.\n", dir_path);
    return;
  }
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    size_t len = strlen(entry->d_name);
    if (len < 5 || strcmp(entry->d_name + len - 4, ".wav") != 0)
      continue;
    char path[MAX_PATH_LEN];
    snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
    wav_t *wav = wav_load(path);
    if (wav == NULL)
      continue;
    int num_segments = wav->num_samples / CLIP_SAMPLES;
    int train_end = num_segments * 8 / 10, validation_end = num_segments * 9 / 10;
    static float clip[CLIP_SAMPLES];
    for (int k = 0; k < train_end * TRAIN_NEGATIVE_RATIO; k++) {
      size_t offset = (size_t)k * CLIP_SAMPLES / TRAIN_NEGATIVE_RATIO;
      if (offset + CLIP_SAMPLES > (size_t)train_end * CLIP_SAMPLES)
        break;
      memcpy(clip, wav->samples + offset, CLIP_SAMPLES * sizeof(float));
      write_row(files[SPLIT_TRAIN], af, clip, 0);
      written[SPLIT_TRAIN]++;
    }
    for (int k = train_end; k < num_segments; k++) {
      split_t split = k < validation_end ? SPLIT_VALIDATION : SPLIT_TEST;
      memcpy(clip, wav->samples + (size_t)k * CLIP_SAMPLES, CLIP_SAMPLES * sizeof(float));
      write_row(files[split], af, clip, 0);
      written[split]++;
    }
    wav_free(wav);
  }
  closedir(dir);
}

// Every training-split ".wav" path under the dataset's word directories:
// the wake word's in `wake`, every other word's in `other`.
typedef struct {
  char **paths;
  int count, capacity;
} path_list_t;

static void path_list_add(path_list_t *list, const char *path)
{
  if (list->count == list->capacity) {
    list->capacity = list->capacity ? 2 * list->capacity : 1024;
    list->paths = realloc(list->paths, (size_t)list->capacity * sizeof(char *));
  }
  list->paths[list->count++] = strdup(path);
}

static void path_list_free(path_list_t *list)
{
  for (int i = 0; i < list->count; i++)
    free(list->paths[i]);
  free(list->paths);
}

static int is_other_word_dir(const char *name, const char *wake_word);

static void collect_training_clips(const char *dataset_dir, const char *wake_word, path_list_t *wake, path_list_t *other)
{
  DIR *top = opendir(dataset_dir);
  if (top == NULL)
    return;
  struct dirent *word;
  while ((word = readdir(top)) != NULL) {
    int is_wake = strcmp(word->d_name, wake_word) == 0;
    if (!is_wake && !is_other_word_dir(word->d_name, wake_word))
      continue;
    char dir_path[MAX_PATH_LEN];
    snprintf(dir_path, sizeof(dir_path), "%s/%s", dataset_dir, word->d_name);
    DIR *dir = opendir(dir_path);
    if (dir == NULL)
      continue;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
      size_t len = strlen(entry->d_name);
      if (len < 5 || strcmp(entry->d_name + len - 4, ".wav") != 0 || split_of(word->d_name, entry->d_name) != SPLIT_TRAIN)
        continue;
      char path[MAX_PATH_LEN];
      snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
      path_list_add(is_wake ? wake : other, path);
    }
    closedir(dir);
  }
  closedir(top);
}

// The training stretch (the first 80%, as write_background_noise() splits
// them) of every background recording, end to end. Returns its length.
static long load_training_background(const char *noise_dir, float **bed)
{
  DIR *dir = opendir(noise_dir);
  if (dir == NULL)
    return 0;
  long len = 0;
  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    size_t name_len = strlen(entry->d_name);
    if (name_len < 5 || strcmp(entry->d_name + name_len - 4, ".wav") != 0)
      continue;
    char path[MAX_PATH_LEN];
    snprintf(path, sizeof(path), "%s/%s", noise_dir, entry->d_name);
    wav_t *wav = wav_load(path);
    if (wav == NULL)
      continue;
    long n = (long)(wav->num_samples / CLIP_SAMPLES * 8 / 10) * CLIP_SAMPLES;
    *bed = realloc(*bed, (size_t)(len + n) * sizeof(float));
    memcpy(*bed + len, wav->samples, (size_t)n * sizeof(float));
    len += n;
    wav_free(wav);
  }
  closedir(dir);
  return len;
}

// Adds `wav` to `run` at `at`, returning the sample just past it.
static long place_clip(float *run, long at, const wav_t *wav)
{
  memcpy(run + at, wav->samples, (size_t)wav->num_samples * sizeof(float));
  return at + wav->num_samples;
}

static long random_gap(void)
{
  return (long)(STREAM_GAP_MIN_MS + rand() % (STREAM_GAP_MAX_MS - STREAM_GAP_MIN_MS + 1)) * SAMPLE_RATE / 1000;
}

// The sample range of the spoken word inside a clip: the first and last
// 10 ms block whose energy is within 20 dB of the clip's loudest block.
static void spoken_extent(const wav_t *wav, long *first, long *last)
{
  const int block = SAMPLE_RATE / 100;
  int num_blocks = wav->num_samples / block;
  float peak = 0.0f;
  for (int b = 0; b < num_blocks; b++) {
    float e = 0.0f;
    for (int k = 0; k < block; k++)
      e += wav->samples[b * block + k] * wav->samples[b * block + k];
    if (e > peak)
      peak = e;
  }
  *first = 0;
  *last = wav->num_samples;
  int found = 0;
  for (int b = 0; b < num_blocks; b++) {
    float e = 0.0f;
    for (int k = 0; k < block; k++)
      e += wav->samples[b * block + k] * wav->samples[b * block + k];
    if (e >= 0.01f * peak) {
      if (!found)
        *first = (long)b * block;
      found = 1;
      *last = (long)(b + 1) * block;
    }
  }
}

// Writes train.csv's streaming windows (see STREAM_POSITIVES_PER_CLIP):
// each is one second cut from three training clips laid end to end with
// random gaps, over the training background at a random SNR. A positive
// window has a wake-word clip in the middle and is placed anywhere that
// still holds all of its spoken word; a negative has three other words
// and is placed anywhere at all, so it usually straddles two of them or
// holds part of one -- the windows a sliding listener sees most, and that
// whole, centered clips never show it. Returns the rows written per label.
static void write_stream_windows(FILE *file, audio_features_t *af, const char *dataset_dir, const char *wake_word,
                                 int written[2])
{
  path_list_t wake = { 0 }, other = { 0 };
  collect_training_clips(dataset_dir, wake_word, &wake, &other);
  char noise_dir[MAX_PATH_LEN];
  snprintf(noise_dir, sizeof(noise_dir), "%s/%s", dataset_dir, BACKGROUND_NOISE_DIR);
  float *bed = NULL;
  long bed_len = load_training_background(noise_dir, &bed);
  if (wake.count == 0 || other.count < 3 || bed_len < STREAM_RUN_SAMPLES) {
    fprintf(stderr, "Warning: not enough training clips or background noise for streaming windows.\n");
    path_list_free(&wake);
    path_list_free(&other);
    free(bed);
    return;
  }
  static float run[STREAM_RUN_SAMPLES], clip[CLIP_SAMPLES];
  const int num_positive = STREAM_POSITIVES_PER_CLIP * wake.count;
  const int total = num_positive * (1 + STREAM_NEGATIVE_RATIO);
  for (int n = 0; n < total; n++) {
    const int label = n < num_positive;
    wav_t *wav[3];
    wav[0] = wav_load(other.paths[rand() % other.count]);
    wav[1] = wav_load(label ? wake.paths[n / STREAM_POSITIVES_PER_CLIP] : other.paths[rand() % other.count]);
    wav[2] = wav_load(other.paths[rand() % other.count]);
    if (!wav[0] || !wav[1] || !wav[2]) {
      for (int k = 0; k < 3; k++)
        wav_free(wav[k]);
      continue;
    }
    memset(run, 0, sizeof(run));
    long middle = place_clip(run, 0, wav[0]) + random_gap();
    long run_len = place_clip(run, place_clip(run, middle, wav[1]) + random_gap(), wav[2]);
    // Background at a random SNR, relative to the three clips' power.
    double speech = 0.0, noise = 0.0;
    long samples = 0;
    for (int k = 0; k < 3; k++) {
      for (int i = 0; i < wav[k]->num_samples; i++)
        speech += (double)wav[k]->samples[i] * wav[k]->samples[i];
      samples += wav[k]->num_samples;
    }
    long bed_at = (long)(((unsigned long)rand() * RAND_MAX + rand()) % (unsigned long)(bed_len - run_len));
    for (long i = 0; i < run_len; i++)
      noise += (double)bed[bed_at + i] * bed[bed_at + i];
    double snr_db = STREAM_SNR_MIN_DB + (STREAM_SNR_MAX_DB - STREAM_SNR_MIN_DB) * ((double)rand() / RAND_MAX);
    float gain = noise > 0.0 ? (float)sqrt(speech / samples / (noise / run_len) / pow(10.0, snr_db / 10.0)) : 0.0f;
    for (long i = 0; i < run_len; i++)
      run[i] += gain * bed[bed_at + i];
    // Where the window may start.
    long lo = 0, hi = run_len - CLIP_SAMPLES;
    if (label) {
      long first, last;
      spoken_extent(wav[1], &first, &last);
      lo = middle + last - CLIP_SAMPLES;
      hi = middle + first;
      if (lo < 0)
        lo = 0;
      if (hi > run_len - CLIP_SAMPLES)
        hi = run_len - CLIP_SAMPLES;
    }
    if (hi >= lo) {
      long start = lo + rand() % (hi - lo + 1);
      memcpy(clip, run + start, sizeof(clip));
      write_row(file, af, clip, label);
      written[label]++;
    }
    for (int k = 0; k < 3; k++)
      wav_free(wav[k]);
  }
  path_list_free(&wake);
  path_list_free(&other);
  free(bed);
}

// True if `name` is a word directory worth walking: not "." / "..", not the
// wake word itself (already handled separately), and not the background
// noise directory (also handled separately).
static int is_other_word_dir(const char *name, const char *wake_word)
{
  return name[0] != '.' && strcmp(name, wake_word) != 0 && strcmp(name, BACKGROUND_NOISE_DIR) != 0;
}

static void print_counts(const char *what, const int counts[NUM_SPLITS])
{
  printf("Wrote %d %s rows (train %d, validation %d, test %d).\n",
         counts[0] + counts[1] + counts[2], what, counts[0], counts[1], counts[2]);
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3) {
    printf("Usage: %s <dataset-dir> [wake-word]\n", argv[0]);
    printf("  <dataset-dir> : Path to an extracted Speech Commands directory (one subdirectory per word,\n");
    printf("                  plus validation_list.txt and testing_list.txt)\n");
    printf("  [wake-word]   : Word to treat as the positive class (default: marvin)\n");
    printf("Writes train.csv, validation.csv and test.csv to the current directory.\n");
    return 1;
  }
  const char *dataset_dir = argv[1];
  const char *wake_word = argc > 2 ? argv[2] : "marvin";

  if (!split_list_load(&validation_list, dataset_dir, "validation_list.txt") ||
      !split_list_load(&testing_list, dataset_dir, "testing_list.txt"))
    return 1;
  audio_features_t *af = audio_features_init(SAMPLE_RATE, FRAME_LEN, NUM_MEL_BINS);
  if (af == NULL) {
    fprintf(stderr, "Error: could not initialize audio feature extractor.\n");
    return 1;
  }
  FILE *files[NUM_SPLITS];
  for (int i = 0; i < NUM_SPLITS; i++) {
    files[i] = fopen(split_paths[i], "w");
    if (files[i] == NULL) {
      fprintf(stderr, "Error: could not open %s for writing.\n", split_paths[i]);
      return 1;
    }
  }

  const int no_cap[NUM_SPLITS] = { INT_MAX, INT_MAX, INT_MAX };
  // Fixed seed, so rerunning this produces the same augmented train.csv.
  srand(1);
  int num_positive[NUM_SPLITS] = { 0 };
  write_word_dir(files, af, dataset_dir, wake_word, no_cap, 1, TRAIN_NEGATIVE_RATIO - 1, num_positive);
  print_counts("positive", num_positive);
  if (num_positive[SPLIT_TRAIN] == 0 || num_positive[SPLIT_VALIDATION] == 0 || num_positive[SPLIT_TEST] == 0) {
    fprintf(stderr, "Error: no wake-word clips found under %s/%s for every split -- is <dataset-dir> correct?\n",
            dataset_dir, wake_word);
    return 1;
  }

  // Roughly balance train.csv's negative class against its wake-word rows:
  // ~90% sampled evenly across the dataset's other word directories, ~10%
  // chopped from its background noise recordings -- both "other speech"
  // and "silence/noise" need to be told apart from the wake word.
  // num_positive already counts the shifted copies, so the
  // TRAIN_NEGATIVE_RATIO times as many distinct negatives still balance it.
  //
  // validation.csv and test.csv aren't balanced: they get every other-word
  // clip their speakers recorded (~10,000 each). A deployed detector hears
  // far more of everything else than of its wake word, and a false-accept
  // rate measured on a couple of hundred negatives moves by half a point
  // per clip. train_*.c and test.c report the two classes' errors
  // separately, so the imbalance doesn't swamp the wake word's.
  DIR *dataset_dir_handle = opendir(dataset_dir);
  if (dataset_dir_handle == NULL) {
    fprintf(stderr, "Error: could not open dataset directory %s.\n", dataset_dir);
    return 1;
  }
  int num_other_dirs = 0;
  struct dirent *entry;
  while ((entry = readdir(dataset_dir_handle)) != NULL) {
    if (!is_other_word_dir(entry->d_name, wake_word))
      continue;
    char probe_path[MAX_PATH_LEN];
    snprintf(probe_path, sizeof(probe_path), "%s/%s", dataset_dir, entry->d_name);
    DIR *probe = opendir(probe_path);
    if (probe != NULL) {
      closedir(probe);
      num_other_dirs++;
    }
  }
  int per_dir_cap[NUM_SPLITS];
  for (int i = 0; i < NUM_SPLITS; i++) {
    int target_other_words = (int)(0.9f * (float)num_positive[i]);
    per_dir_cap[i] = i != SPLIT_TRAIN ? INT_MAX
                   : num_other_dirs > 0 ? (target_other_words + num_other_dirs - 1) / num_other_dirs : 0;
  }

  rewinddir(dataset_dir_handle);
  int num_negative[NUM_SPLITS] = { 0 };
  while ((entry = readdir(dataset_dir_handle)) != NULL) {
    if (!is_other_word_dir(entry->d_name, wake_word))
      continue;
    char probe_path[MAX_PATH_LEN];
    snprintf(probe_path, sizeof(probe_path), "%s/%s", dataset_dir, entry->d_name);
    DIR *probe = opendir(probe_path);
    if (probe == NULL)
      continue;
    closedir(probe);
    write_word_dir(files, af, dataset_dir, entry->d_name, per_dir_cap, 0, 0, num_negative);
  }
  closedir(dataset_dir_handle);
  print_counts("negative other-word", num_negative);

  char noise_dir[MAX_PATH_LEN];
  snprintf(noise_dir, sizeof(noise_dir), "%s/%s", dataset_dir, BACKGROUND_NOISE_DIR);
  int num_noise[NUM_SPLITS] = { 0 };
  write_background_noise(files, af, noise_dir, num_noise);
  print_counts("negative background-noise", num_noise);

  int num_stream[2] = { 0 };
  write_stream_windows(files[SPLIT_TRAIN], af, dataset_dir, wake_word, num_stream);
  printf("Wrote %d streaming-window training rows (%d positive, %d negative).\n",
         num_stream[0] + num_stream[1], num_stream[1], num_stream[0]);

  for (int i = 0; i < NUM_SPLITS; i++)
    fclose(files[i]);
  audio_features_free(af);
  split_list_free(&validation_list);
  split_list_free(&testing_list);
  return 0;
}
