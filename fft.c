/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include "fft.h"

// Local, not M_PI -- see audio_features.c's identical constant.
static const float PI = 3.14159265358979323846f;

// Iterative radix-2 Cooley-Tukey FFT (decimation in time): standard
// bit-reversal permutation followed by log2(n) butterfly stages; no
// allocation.
void fft(float *real, float *imag, int n)
{
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      float tmp = real[i];
      real[i] = real[j];
      real[j] = tmp;
      tmp = imag[i];
      imag[i] = imag[j];
      imag[j] = tmp;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * PI / (float)len;
    float wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cur_wr = 1.0f, cur_wi = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, b = i + k + len / 2;
        float ur = real[a], ui = imag[a];
        float vr = real[b] * cur_wr - imag[b] * cur_wi;
        float vi = real[b] * cur_wi + imag[b] * cur_wr;
        real[a] = ur + vr;
        imag[a] = ui + vi;
        real[b] = ur - vr;
        imag[b] = ui - vi;
        float next_wr = cur_wr * wr - cur_wi * wi;
        float next_wi = cur_wr * wi + cur_wi * wr;
        cur_wr = next_wr;
        cur_wi = next_wi;
      }
    }
  }
}

// ifft(x) = conj(fft(conj(x))) / n -- reuses the forward transform rather
// than duplicating its butterflies with the twiddle sign flipped.
void ifft(float *real, float *imag, int n)
{
  for (int i = 0; i < n; i++)
    imag[i] = -imag[i];
  fft(real, imag, n);
  float scale = 1.0f / (float)n;
  for (int i = 0; i < n; i++) {
    real[i] *= scale;
    imag[i] = -imag[i] * scale;
  }
}
