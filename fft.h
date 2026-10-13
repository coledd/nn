/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FFT_H
#define FFT_H

// In-place radix-2 FFT shared by the on-device feature extractors
// (audio_features.[ch] and examples/predictive_maintenance's
// vibration_features.[ch]), so the one transform both depend on is
// compiled -- and fixed, if it ever needs fixing -- in exactly one place.
// Like those two, it's meant to ship into firmware next to nn.c/nn.h: no
// allocation, no file I/O, no dependency beyond <math.h>.
//
// It favors portability over speed. On an Arm Cortex-M target with
// CMSIS-DSP available, arm_rfft_fast_f32() computes the same transform
// several times faster and is a drop-in replacement at the call sites.
//
// Not part of libnn.a (it isn't neural-net code) -- see the top-level
// Makefile's fft.o rule.

// Forward DFT of the `n` complex values (`real`[i], `imag`[i]), in place:
// X[k] = sum_t x[t] * exp(-2*pi*i*k*t/n), unscaled. `n` must be a power of
// 2; the caller is responsible for checking that.
void fft(float *real, float *imag, int n);

// Inverse of fft(), including its 1/n scaling, so ifft(fft(x)) == x.
void ifft(float *real, float *imag, int n);

#endif /* FFT_H */
