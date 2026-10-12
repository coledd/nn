/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wav_reader.h"

static uint32_t read_le32(const uint8_t *p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t read_le16(const uint8_t *p)
{
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

wav_t *wav_load(const char *path)
{
  FILE *file = fopen(path, "rb");
  if (file == NULL)
    return NULL;

  uint8_t riff_header[12];
  if (fread(riff_header, 1, 12, file) != 12 ||
      memcmp(riff_header, "RIFF", 4) != 0 || memcmp(riff_header + 8, "WAVE", 4) != 0) {
    fclose(file);
    return NULL;
  }

  int have_fmt = 0, audio_format = 0, num_channels = 0, sample_rate = 0, bits_per_sample = 0;
  uint8_t *data = NULL;
  uint32_t data_size = 0;

  // Walk the WAVE file's subchunks in whatever order they appear -- "fmt "
  // isn't guaranteed to come immediately before "data", and other chunks
  // (e.g. "LIST", "fact") are simply skipped.
  uint8_t chunk_header[8];
  while (fread(chunk_header, 1, 8, file) == 8) {
    uint32_t chunk_size = read_le32(chunk_header + 4);
    if (memcmp(chunk_header, "fmt ", 4) == 0 && chunk_size >= 16) {
      uint8_t fmt[16];
      if (fread(fmt, 1, 16, file) != 16) {
        free(data);
        fclose(file);
        return NULL;
      }
      audio_format = read_le16(fmt);
      num_channels = read_le16(fmt + 2);
      sample_rate = (int)read_le32(fmt + 4);
      bits_per_sample = read_le16(fmt + 14);
      have_fmt = 1;
      if (chunk_size > 16)
        fseek(file, (long)(chunk_size - 16), SEEK_CUR);
    } else if (memcmp(chunk_header, "data", 4) == 0) {
      free(data); // in case an earlier, malformed "data" chunk preceded this one
      data = (uint8_t *)malloc(chunk_size);
      if (data == NULL || fread(data, 1, chunk_size, file) != chunk_size) {
        free(data);
        fclose(file);
        return NULL;
      }
      data_size = chunk_size;
    } else {
      fseek(file, (long)chunk_size, SEEK_CUR);
    }
    if (chunk_size & 1) // subchunks are word-aligned; skip the pad byte if the size was odd
      fseek(file, 1, SEEK_CUR);
  }
  fclose(file);

  if (!have_fmt || data == NULL || audio_format != 1 || num_channels != 1 || bits_per_sample != 16) {
    free(data);
    return NULL;
  }

  int num_samples = (int)(data_size / 2);
  wav_t *wav = (wav_t *)malloc(sizeof(wav_t));
  if (wav == NULL) {
    free(data);
    return NULL;
  }
  wav->samples = (float *)malloc((size_t)num_samples * sizeof(float));
  if (wav->samples == NULL) {
    free(wav);
    free(data);
    return NULL;
  }
  for (int i = 0; i < num_samples; i++) {
    int16_t sample = (int16_t)((uint16_t)data[2 * i] | ((uint16_t)data[2 * i + 1] << 8));
    wav->samples[i] = (float)sample / 32768.0f;
  }
  free(data);
  wav->sample_rate = sample_rate;
  wav->num_samples = num_samples;
  return wav;
}

void wav_free(wav_t *wav)
{
  if (wav == NULL)
    return;
  free(wav->samples);
  free(wav);
}
