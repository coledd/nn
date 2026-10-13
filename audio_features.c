/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "audio_features.h"
#include "fft.h"

// Local, not M_PI -- <math.h> doesn't portably define it (it's a glibc
// extension gated on feature-test macros this project doesn't set), and
// fall_data.c/gesture_data.c already establish the pattern of a local
// float pi constant instead of depending on it.
static const float PI = 3.14159265358979323846f;

static int is_power_of_two(int n)
{
  return n > 0 && (n & (n - 1)) == 0;
}

static float hz_to_mel(float hz)
{
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel)
{
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Builds the num_mel_bins x num_fft_bins triangular mel filterbank (the
// standard HTK/librosa-style construction): num_mel_bins+2 points spaced
// evenly in mel scale from 0Hz to Nyquist, mapped back to FFT bin indices,
// with filter m rising linearly from bin[m-1] to bin[m] and falling back to
// 0 by bin[m+1].
static int build_mel_filterbank(audio_features_t *af)
{
  int num_points = af->num_mel_bins + 2;
  float *mel_points = (float *)malloc((size_t)num_points * sizeof(float));
  int *bin = (int *)malloc((size_t)num_points * sizeof(int));
  if (mel_points == NULL || bin == NULL) {
    free(mel_points);
    free(bin);
    return 0;
  }
  float mel_min = hz_to_mel(0.0f);
  float mel_max = hz_to_mel((float)af->sample_rate / 2.0f);
  for (int i = 0; i < num_points; i++) {
    mel_points[i] = mel_min + (mel_max - mel_min) * (float)i / (float)(num_points - 1);
    float hz = mel_to_hz(mel_points[i]);
    bin[i] = (int)floorf((float)(af->frame_len + 1) * hz / (float)af->sample_rate);
  }
  for (int m = 1; m <= af->num_mel_bins; m++) {
    int left = bin[m - 1], center = bin[m], right = bin[m + 1];
    float *row = af->mel_weights + (m - 1) * af->num_fft_bins;
    for (int k = 0; k < af->num_fft_bins; k++) {
      float w = 0.0f;
      if (k >= left && k <= center && center > left)
        w = (float)(k - left) / (float)(center - left);
      else if (k >= center && k <= right && right > center)
        w = (float)(right - k) / (float)(right - center);
      row[k] = w;
    }
  }
  free(mel_points);
  free(bin);
  return 1;
}

audio_features_t *audio_features_init(int sample_rate, int frame_len, int num_mel_bins)
{
  if (sample_rate <= 0 || num_mel_bins <= 0 || !is_power_of_two(frame_len))
    return NULL;

  audio_features_t *af = (audio_features_t *)malloc(sizeof(audio_features_t));
  if (af == NULL)
    return NULL;
  af->sample_rate = sample_rate;
  af->frame_len = frame_len;
  af->num_fft_bins = frame_len / 2 + 1;
  af->num_mel_bins = num_mel_bins;
  af->window = (float *)malloc((size_t)frame_len * sizeof(float));
  af->mel_weights = (float *)malloc((size_t)num_mel_bins * (size_t)af->num_fft_bins * sizeof(float));
  af->scratch_real = (float *)malloc((size_t)frame_len * sizeof(float));
  af->scratch_imag = (float *)malloc((size_t)frame_len * sizeof(float));
  af->power = (float *)malloc((size_t)af->num_fft_bins * sizeof(float));
  if (af->window == NULL || af->mel_weights == NULL || af->scratch_real == NULL ||
      af->scratch_imag == NULL || af->power == NULL || !build_mel_filterbank(af)) {
    audio_features_free(af);
    return NULL;
  }
  // Hann window: tapers each frame's edges to zero, reducing the spectral
  // leakage a hard (rectangular) frame boundary would otherwise introduce.
  for (int n = 0; n < frame_len; n++)
    af->window[n] = 0.5f - 0.5f * cosf(2.0f * PI * (float)n / (float)(frame_len - 1));
  return af;
}

void audio_features_free(audio_features_t *af)
{
  if (af == NULL)
    return;
  free(af->window);
  free(af->mel_weights);
  free(af->scratch_real);
  free(af->scratch_imag);
  free(af->power);
  free(af);
}

void audio_features_frame(audio_features_t *af, const float *frame, float *out)
{
  for (int i = 0; i < af->frame_len; i++) {
    af->scratch_real[i] = frame[i] * af->window[i];
    af->scratch_imag[i] = 0.0f;
  }
  fft(af->scratch_real, af->scratch_imag, af->frame_len);
  for (int k = 0; k < af->num_fft_bins; k++)
    af->power[k] = af->scratch_real[k] * af->scratch_real[k] + af->scratch_imag[k] * af->scratch_imag[k];
  for (int m = 0; m < af->num_mel_bins; m++) {
    const float *row = af->mel_weights + m * af->num_fft_bins;
    float energy = 0.0f;
    for (int k = 0; k < af->num_fft_bins; k++)
      energy += row[k] * af->power[k];
    // +1e-6f: log(0) is undefined, and a silent/zero-padded frame (see
    // wake_word_data.h's CLIP_SAMPLES padding) is exactly the case where
    // `energy` can legitimately be 0.
    out[m] = logf(energy + 1e-6f);
  }
}

int audio_features_num_frames(const audio_features_t *af, int num_samples, int hop)
{
  if (num_samples < af->frame_len)
    return 0;
  return 1 + (num_samples - af->frame_len) / hop;
}

void audio_features_extract_sequence(audio_features_t *af, const float *pcm, int num_samples, int hop, float *out)
{
  int num_frames = audio_features_num_frames(af, num_samples, hop);
  for (int t = 0; t < num_frames; t++)
    audio_features_frame(af, pcm + t * hop, out + t * af->num_mel_bins);

  // Per-clip mean/variance normalization: real recordings vary widely in
  // overall loudness and spectral tilt (a quiet vs. a loud speaker, a
  // near-field vs. far-field mic), which otherwise shows up as a
  // per-clip-varying offset and scale in the raw log-mel energies above --
  // exactly the kind of inconsistent input that makes a downstream model's
  // training noisy or slow to converge. Doing this here, over the whole
  // sequence just computed, needs no corpus-wide statistics that training
  // and a deployed target would otherwise have to keep in sync -- it's
  // self-contained, and applies identically whether `out` ends up in a
  // training CSV row or fed straight into nn_predict().
  int n = num_frames * af->num_mel_bins;
  if (n > 0) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++)
      mean += out[i];
    mean /= (float)n;
    float variance = 0.0f;
    for (int i = 0; i < n; i++) {
      float d = out[i] - mean;
      variance += d * d;
    }
    variance /= (float)n;
    float inv_std = 1.0f / sqrtf(variance + 1e-6f);
    for (int i = 0; i < n; i++)
      out[i] = (out[i] - mean) * inv_std;
  }
}
