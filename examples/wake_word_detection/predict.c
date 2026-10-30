/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Demonstrates using a trained wake-word model in an application: loads one
// WAV file and extracts its log-mel feature sequence via
// audio_features_extract_sequence() -- the EXACT function a deployed
// firmware would call against its own live mic buffer, not a
// desktop-specific shortcut -- then runs it through the model.
// wav_reader.[ch] (desktop-only, since firmware never reads a .wav file) is
// the only piece of this file that wouldn't also appear, largely verbatim,
// in an embedded build: swap it for whatever hands you a CLIP_SAMPLES-long
// PCM buffer from the mic/ADC driver, and the audio_features_extract_sequence()
// + nn_predict() calls below are unchanged. See this example's README for
// the library-vs-application split this file is meant to illustrate.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "audio_features.h"
#include "nn.h"
#include "wake_word_data.h"
#include "wav_reader.h"

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <model-file> <sample.wav>\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train_gru.c or train_cnn.c\n");
    printf("  <sample.wav> : A 16-bit PCM mono WAV file at %d Hz, e.g. one from the Speech Commands dataset\n", SAMPLE_RATE);
    return 1;
  }
  const char *model_path = argv[1];
  const char *wav_path = argv[2];

  wav_t *wav = wav_load(wav_path);
  if (wav == NULL) {
    fprintf(stderr, "Error: could not load %s (must be 16-bit PCM mono WAV).\n", wav_path);
    return 1;
  }
  if (wav->sample_rate != SAMPLE_RATE) {
    fprintf(stderr, "Error: %s is %d Hz, expected %d Hz.\n", wav_path, wav->sample_rate, SAMPLE_RATE);
    wav_free(wav);
    return 1;
  }
  // Pad or truncate to the same fixed CLIP_SAMPLES length every training
  // clip was fit to (see prepare_data.c's fit_to_clip_length()) -- real
  // firmware would instead maintain a rolling CLIP_SAMPLES-sample buffer
  // fed continuously from its mic driver.
  static float clip[CLIP_SAMPLES];
  int n = wav->num_samples < CLIP_SAMPLES ? wav->num_samples : CLIP_SAMPLES;
  memcpy(clip, wav->samples, (size_t)n * sizeof(float));
  if (n < CLIP_SAMPLES)
    memset(clip + n, 0, (size_t)(CLIP_SAMPLES - n) * sizeof(float));
  wav_free(wav);

  audio_features_t *af = audio_features_init(SAMPLE_RATE, FRAME_LEN, NUM_MEL_BINS);
  if (af == NULL) {
    fprintf(stderr, "Error: could not initialize audio feature extractor.\n");
    return 1;
  }
  static float sequence[SEQUENCE_LEN * NUM_MEL_BINS];
  audio_features_extract_sequence(af, clip, CLIP_SAMPLES, HOP_LEN, sequence);
  audio_features_free(af);

  // Recall a previously trained neural network model (ascii, binary, or
  // inplace, auto-detected). See examples/character_recognition/predict.c's
  // identical block for why the inplace format needs to be read into a
  // buffer first.
  nn_model_format_t format = nn_model_format(model_path);
  nn_t *model;
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      printf("Error: Missing or invalid model file.\n");
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    inplace_buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!inplace_buf || fread(inplace_buf, 1, (size_t)size, file) != (size_t)size) {
      printf("Error: Could not read model file.\n");
      fclose(file);
      free(inplace_buf);
      return 1;
    }
    fclose(file);
    model = nn_load_model_inplace(inplace_buf, (size_t)size);
  } else {
    model = nn_load_model(model_path);
  }
  if (NULL == model) {
    printf("Error: Missing or invalid model file.\n");
    free(inplace_buf);
    return 1;
  }
  // The GRU model reads the sequence one frame per call, the CNN model the
  // whole clip at once; wake_word_probability() (wake_word_data.h) handles
  // both, and is the call a firmware would make once per window.
  float probability = wake_word_probability(model, sequence);
  if (probability < 0.0f) {
    fprintf(stderr, "Error: model's shape (%u inputs, %u outputs) matches neither wake-word model.\n",
            model->width[0], model->width[model->depth - 1]);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }
  printf("Wake-word probability: %.5f\n", probability);

  nn_free(model);
  free(inplace_buf);
  return 0;
}
