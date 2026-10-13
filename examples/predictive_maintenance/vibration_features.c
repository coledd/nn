/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <string.h>
#include "fft.h"
#include "vibration_features.h"

// Local, not M_PI -- see ../../audio_features.c's identical constant.
static const float PI = 3.14159265358979323846f;
static const float STANDARD_GRAVITY = 9.80665f; // m/s^2 per g

static int hz_to_bin(float hz)
{
  return (int)lroundf(hz * (float)VF_FRAME_LEN / VF_SAMPLE_RATE);
}

// Logarithmically spaced band edges between `low_hz` and `high_hz`, in FFT
// bins. Down at the low end, log spacing asks for bands narrower than one
// bin; those are widened to exactly one bin (and everything above shifted
// up accordingly) so no band is ever empty.
static void log_band_edges(uint16_t *edges, int num_bands, float low_hz, float high_hz)
{
  int low = hz_to_bin(low_hz), high = hz_to_bin(high_hz);
  float ratio = (float)high / (float)low;
  edges[0] = (uint16_t)low;
  for (int b = 1; b <= num_bands; b++) {
    int edge = (int)lroundf((float)low * powf(ratio, (float)b / (float)num_bands));
    if (edge <= edges[b - 1])
      edge = edges[b - 1] + 1;
    edges[b] = (uint16_t)edge;
  }
}

static void reset_snapshot(vibration_features_t *vf)
{
  memset(vf->spectrum, 0, sizeof(vf->spectrum));
  memset(vf->envelope, 0, sizeof(vf->envelope));
  vf->frames = 0;
}

void vibration_features_init(vibration_features_t *vf)
{
  vf->window_sum = 0.0f;
  vf->window_power = 0.0f;
  for (int n = 0; n < VF_FRAME_LEN; n++) {
    vf->window[n] = 0.5f - 0.5f * cosf(2.0f * PI * (float)n / (float)(VF_FRAME_LEN - 1));
    vf->window_sum += vf->window[n];
    vf->window_power += vf->window[n] * vf->window[n];
  }
  log_band_edges(vf->spectrum_edges, VF_SPECTRUM_BANDS, VF_SPECTRUM_LOW_HZ, VF_SPECTRUM_HIGH_HZ);
  log_band_edges(vf->envelope_edges, VF_ENVELOPE_BANDS, VF_ENVELOPE_SPECTRUM_LOW_HZ, VF_ENVELOPE_SPECTRUM_HIGH_HZ);
  vf->envelope_low_bin = (uint16_t)hz_to_bin(VF_ENVELOPE_LOW_HZ);
  vf->envelope_high_bin = (uint16_t)hz_to_bin(VF_ENVELOPE_HIGH_HZ);
  reset_snapshot(vf);
}

void vibration_features_add_frame(vibration_features_t *vf, const float *frame)
{
  float *re = vf->real, *im = vf->imag;
  const float *w = vf->window;

  // 1. Vibration spectrum of the windowed frame.
  for (int n = 0; n < VF_FRAME_LEN; n++) {
    re[n] = frame[n] * w[n];
    im[n] = 0.0f;
  }
  fft(re, im, VF_FRAME_LEN);
  for (int k = 0; k < VF_NUM_BINS; k++)
    vf->spectrum[k] += re[k] * re[k] + im[k] * im[k];

  // 2. Envelope of the resonance band, from the same transform: keep only
  // the positive-frequency bins inside the band, doubled, and zero the
  // rest (including every negative-frequency bin). The inverse FFT of that
  // is the band-passed analytic signal, whose magnitude is the envelope.
  for (int k = 0; k < VF_FRAME_LEN; k++) {
    if (k >= vf->envelope_low_bin && k <= vf->envelope_high_bin) {
      re[k] *= 2.0f;
      im[k] *= 2.0f;
    } else {
      re[k] = 0.0f;
      im[k] = 0.0f;
    }
  }
  ifft(re, im, VF_FRAME_LEN);
  // The frame was already Hann-windowed above, so the envelope comes out
  // windowed too. Its DC level, though, comes out as a Hann-shaped bump
  // that would leak into the lowest envelope bands -- subtract the
  // windowed mean (envelope mean times the window) rather than a flat one.
  float envelope_sum = 0.0f;
  for (int n = 0; n < VF_FRAME_LEN; n++) {
    re[n] = sqrtf(re[n] * re[n] + im[n] * im[n]);
    envelope_sum += re[n];
  }
  float envelope_mean = envelope_sum / vf->window_sum;
  for (int n = 0; n < VF_FRAME_LEN; n++) {
    re[n] -= envelope_mean * w[n];
    im[n] = 0.0f;
  }

  // 3. Envelope spectrum.
  fft(re, im, VF_FRAME_LEN);
  for (int k = 0; k < VF_NUM_BINS; k++)
    vf->envelope[k] += re[k] * re[k] + im[k] * im[k];
  vf->frames++;
}

// Log10 energy of each band, then shifted to zero mean across the bands.
// The shift cancels any overall scale on the spectrum -- in particular the
// accelerometer's mounting gain, which varies from one installation to
// the next and isn't a property of the machine's health.
static void band_log_energies(const float *power, const uint16_t *edges, int num_bands, float *out)
{
  float mean = 0.0f;
  for (int b = 0; b < num_bands; b++) {
    float energy = 0.0f;
    for (int k = edges[b]; k < edges[b + 1]; k++)
      energy += power[k];
    // +1e-12f: log(0) is undefined. Real sensor noise keeps every band
    // well above this.
    out[b] = log10f(energy + 1e-12f);
    mean += out[b];
  }
  mean /= (float)num_bands;
  for (int b = 0; b < num_bands; b++)
    out[b] -= mean;
}

float vibration_features_finish(vibration_features_t *vf, float *fingerprint)
{
  if (vf->frames == 0)
    return -1.0f;

  // RMS velocity: each bin's acceleration power divided by (2*pi*f)^2
  // (integrating acceleration to velocity), summed over 10Hz..1kHz. The
  // 2/(N*sum(w^2)) factor turns a one-sided Hann-windowed power spectrum
  // into mean-square amplitude (Parseval), so a pure 1g-peak sine reads
  // as 1/sqrt(2) g RMS before integration.
  float scale = 2.0f / ((float)VF_FRAME_LEN * vf->window_power * (float)vf->frames);
  float velocity_ms = 0.0f;
  for (int k = hz_to_bin(VF_VELOCITY_LOW_HZ); k <= hz_to_bin(VF_VELOCITY_HIGH_HZ); k++) {
    float omega = 2.0f * PI * (float)k * VF_SAMPLE_RATE / (float)VF_FRAME_LEN;
    velocity_ms += vf->spectrum[k] * scale / (omega * omega);
  }
  float velocity_mm_s = sqrtf(velocity_ms) * STANDARD_GRAVITY * 1000.0f;

  band_log_energies(vf->spectrum, vf->spectrum_edges, VF_SPECTRUM_BANDS, fingerprint);
  band_log_energies(vf->envelope, vf->envelope_edges, VF_ENVELOPE_BANDS, fingerprint + VF_SPECTRUM_BANDS);
  reset_snapshot(vf);
  return velocity_mm_s;
}
