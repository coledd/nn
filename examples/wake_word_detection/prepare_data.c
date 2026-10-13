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
// one flattened row per clip to a CSV: SEQUENCE_LEN * NUM_MEL_BINS log-mel
// energies (see wake_word_data.h and ../../audio_features.h), followed by a
// single 0/1 label -- 1 for the wake word, 0 for everything else (other
// spoken words, plus 1-second chunks cut from the dataset's own
// background-noise recordings, so the model also learns to stay quiet
// through silence/ambient noise, not just other speech).
//
// This is the one genuinely new tool in this example. Unlike
// examples/character_recognition's samples.csv (MNIST's pixels, flattened
// and thresholded -- which has no bearing on what a deployed target
// does), flattening audio into feature rows via audio_features.[ch] IS
// exactly what a deployed target's firmware has to do against its own live
// mic buffer, so that step has to be real, working code here rather than a
// downloaded shortcut. Everything downstream of this file -- split.py,
// data_prep.[ch], the CSV loading in train.c/
// test.c -- is the same general-purpose CSV machinery
// examples/character_recognition already established.
//
// The wake word is a command-line argument, not hardcoded: "marvin" (this
// example's default) is one of two words the dataset's own authors
// included specifically so they could double as a pretend wake word
// (Warden, 2018, "Speech Commands: A Dataset for Limited-Vocabulary Speech
// Recognition", https://arxiv.org/abs/1804.03209), but swapping to a
// different word later needs no code changes, only a different argument.

#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_features.h"
#include "wake_word_data.h"
#include "wav_reader.h"

#define MAX_PATH_LEN 4096
#define BACKGROUND_NOISE_DIR "_background_noise_"

// Copies `wav`'s samples into a fixed CLIP_SAMPLES-length buffer, padding
// with digital silence if the clip is shorter, or truncating if longer --
// every row this tool writes needs the same SEQUENCE_LEN, which
// audio_features_extract_sequence() only guarantees for a fixed-length
// input.
static void fit_to_clip_length(const wav_t *wav, float *clip)
{
  int n = wav->num_samples < CLIP_SAMPLES ? wav->num_samples : CLIP_SAMPLES;
  memcpy(clip, wav->samples, (size_t)n * sizeof(float));
  if (n < CLIP_SAMPLES)
    memset(clip + n, 0, (size_t)(CLIP_SAMPLES - n) * sizeof(float));
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

// Loads one WAV file and writes its feature row, returning 1 on success.
// Silently skips (returns 0 for) a file that fails to load -- real-world
// datasets this size have a handful of malformed/truncated clips, not worth
// aborting the whole pass over.
static int write_wav_row(FILE *file, audio_features_t *af, const char *path, int label)
{
  wav_t *wav = wav_load(path);
  if (wav == NULL)
    return 0;
  static float clip[CLIP_SAMPLES];
  fit_to_clip_length(wav, clip);
  wav_free(wav);
  write_row(file, af, clip, label);
  return 1;
}

// Writes up to `max_count` rows (with the given `label`) from every ".wav"
// file directly inside `dir_path`, in whatever order readdir() returns
// them -- good enough for a roughly-balanced negative class, since these
// are already independent per-utterance recordings with no ordering to
// correct for. Returns the number of rows actually written.
static int write_word_dir(FILE *file, audio_features_t *af, const char *dir_path, int max_count, int label)
{
  DIR *dir = opendir(dir_path);
  if (dir == NULL) {
    fprintf(stderr, "Warning: could not open %s, skipping.\n", dir_path);
    return 0;
  }
  int written = 0;
  struct dirent *entry;
  while (written < max_count && (entry = readdir(dir)) != NULL) {
    size_t len = strlen(entry->d_name);
    if (len < 5 || strcmp(entry->d_name + len - 4, ".wav") != 0)
      continue;
    char path[MAX_PATH_LEN];
    snprintf(path, sizeof(path), "%s/%s", dir_path, entry->d_name);
    written += write_wav_row(file, af, path, label);
  }
  closedir(dir);
  return written;
}

// Chops every "*.wav" file in `dir_path` (the dataset's
// "_background_noise_" directory -- a handful of longer ambient-noise
// recordings, not per-word utterances) into non-overlapping CLIP_SAMPLES
// segments, each written as its own label-0 row: silence/noise negatives,
// standard keyword-spotting practice alongside the "other words" negatives
// written by write_word_dir(). Returns the number of rows written.
static int write_background_noise(FILE *file, audio_features_t *af, const char *dir_path)
{
  DIR *dir = opendir(dir_path);
  if (dir == NULL) {
    fprintf(stderr, "Warning: could not open %s, skipping background noise.\n", dir_path);
    return 0;
  }
  int written = 0;
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
    static float clip[CLIP_SAMPLES];
    for (int offset = 0; offset + CLIP_SAMPLES <= wav->num_samples; offset += CLIP_SAMPLES) {
      memcpy(clip, wav->samples + offset, CLIP_SAMPLES * sizeof(float));
      write_row(file, af, clip, 0);
      written++;
    }
    wav_free(wav);
  }
  closedir(dir);
  return written;
}

// True if `name` is a word directory worth walking: not "." / "..", not the
// wake word itself (already handled separately), and not the background
// noise directory (also handled separately).
static int is_other_word_dir(const char *name, const char *wake_word)
{
  return name[0] != '.' && strcmp(name, wake_word) != 0 && strcmp(name, BACKGROUND_NOISE_DIR) != 0;
}

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 4) {
    printf("Usage: %s <dataset-dir> <output.csv> [wake-word]\n", argv[0]);
    printf("  <dataset-dir> : Path to an extracted Speech Commands directory (one subdirectory per word)\n");
    printf("  <output.csv>  : Path to write the flattened feature/label CSV to\n");
    printf("  [wake-word]   : Word to treat as the positive class (default: marvin)\n");
    return 1;
  }
  const char *dataset_dir = argv[1];
  const char *output_path = argv[2];
  const char *wake_word = argc > 3 ? argv[3] : "marvin";

  audio_features_t *af = audio_features_init(SAMPLE_RATE, FRAME_LEN, NUM_MEL_BINS);
  if (af == NULL) {
    fprintf(stderr, "Error: could not initialize audio feature extractor.\n");
    return 1;
  }
  FILE *out = fopen(output_path, "w");
  if (out == NULL) {
    fprintf(stderr, "Error: could not open %s for writing.\n", output_path);
    audio_features_free(af);
    return 1;
  }

  char wake_word_dir[MAX_PATH_LEN];
  snprintf(wake_word_dir, sizeof(wake_word_dir), "%s/%s", dataset_dir, wake_word);
  int num_positive = write_word_dir(out, af, wake_word_dir, INT_MAX, 1);
  printf("Wrote %d positive (\"%s\") rows.\n", num_positive, wake_word);
  if (num_positive == 0) {
    fprintf(stderr, "Error: no wake-word clips found under %s -- is <dataset-dir> correct?\n", wake_word_dir);
    fclose(out);
    audio_features_free(af);
    return 1;
  }

  // Roughly balance the negative class: ~90% sampled evenly across the
  // dataset's other word directories, ~10% chopped from its background
  // noise recordings -- both "other speech" and "silence/noise" need to be
  // told apart from the wake word.
  DIR *dataset_dir_handle = opendir(dataset_dir);
  if (dataset_dir_handle == NULL) {
    fprintf(stderr, "Error: could not open dataset directory %s.\n", dataset_dir);
    fclose(out);
    audio_features_free(af);
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
  int target_other_words = (int)(0.9f * (float)num_positive);
  int per_dir_cap = num_other_dirs > 0 ? (target_other_words + num_other_dirs - 1) / num_other_dirs : 0;

  rewinddir(dataset_dir_handle);
  int num_negative = 0;
  while ((entry = readdir(dataset_dir_handle)) != NULL) {
    if (!is_other_word_dir(entry->d_name, wake_word))
      continue;
    char word_dir[MAX_PATH_LEN];
    snprintf(word_dir, sizeof(word_dir), "%s/%s", dataset_dir, entry->d_name);
    num_negative += write_word_dir(out, af, word_dir, per_dir_cap, 0);
  }
  closedir(dataset_dir_handle);
  printf("Wrote %d negative rows from other words.\n", num_negative);

  char noise_dir[MAX_PATH_LEN];
  snprintf(noise_dir, sizeof(noise_dir), "%s/%s", dataset_dir, BACKGROUND_NOISE_DIR);
  int num_noise = write_background_noise(out, af, noise_dir);
  printf("Wrote %d negative rows from background noise.\n", num_noise);

  fclose(out);
  audio_features_free(af);
  printf("Total: %d positive, %d negative rows written to %s.\n",
         num_positive, num_negative + num_noise, output_path);
  return 0;
}
