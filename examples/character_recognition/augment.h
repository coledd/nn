/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AUGMENT_H
#define AUGMENT_H

#include <stdlib.h>

// Training-time augmentation shared by train_cnn.c and train_fc.c. MNIST's
// digits are centered by their center of mass, but real input -- a digit
// cropped from a form, a touch panel, a camera frame -- never sits exactly
// where MNIST's do. Shifting each training digit by a random small amount,
// fresh every epoch, teaches the network that position doesn't matter,
// at no cost to the data or to inference.
#define IMAGE_SIDE 28
#define MAX_SHIFT 2 // pixels, either way, on each axis

// Copies the IMAGE_SIDE x IMAGE_SIDE image `in` to `out`, moved by a random
// whole number of pixels in [-MAX_SHIFT, MAX_SHIFT] on each axis, filling
// the uncovered edge with background (0).
static inline void random_shift(const float *in, float *out)
{
  int dx = rand() % (2 * MAX_SHIFT + 1) - MAX_SHIFT;
  int dy = rand() % (2 * MAX_SHIFT + 1) - MAX_SHIFT;
  for (int y = 0; y < IMAGE_SIDE; y++) {
    for (int x = 0; x < IMAGE_SIDE; x++) {
      int sx = x - dx, sy = y - dy;
      out[y * IMAGE_SIDE + x] =
          (sx >= 0 && sx < IMAGE_SIDE && sy >= 0 && sy < IMAGE_SIDE) ? in[sy * IMAGE_SIDE + sx] : 0.0f;
    }
  }
}

#endif /* AUGMENT_H */
