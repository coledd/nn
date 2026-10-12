/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef WAV_READER_H
#define WAV_READER_H

// Decodes a 16-bit PCM mono WAV file into a float sample buffer -- desktop
// training-data-prep tooling, same category as data_prep.[ch]: general
// enough (any 16-bit PCM mono WAV, nothing dataset-specific) to earn root
// placement even with a single consumer today, but not something an
// embedded target would want, since firmware gets PCM straight from its own
// mic/ADC driver, never a .wav file. Not part of libnn.a for that reason --
// see the top-level Makefile's wav_reader.o rule.

typedef struct {
  int sample_rate;
  int num_samples;
  float *samples; // num_samples floats, normalized to [-1, 1]
} wav_t;

// Reads and decodes the WAV file at `path`. Returns NULL on any I/O or
// format error, including a WAV file that isn't 16-bit PCM mono (the only
// format this decoder supports).
wav_t *wav_load(const char *path);
void wav_free(wav_t *wav);

#endif /* WAV_READER_H */
