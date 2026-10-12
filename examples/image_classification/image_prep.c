/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// See image_prep.h for what this is and why it's shared.
//
// Color (IMG_CHANNELS == 3). Measured against grayscale when this example
// trained on Oxford-IIIT Pet alone (see this example's README), the two
// scored the same within run-to-run noise (73.9% vs. 74.1% test accuracy,
// 5 and 2 runs) -- coat color alone
// doesn't separate cats from dogs, since both come in every color. Color
// is kept because it's what a typical camera module delivers anyway; for a
// monochrome (e.g. IR night-vision) sensor, IMG_CHANNELS 1 in image_prep.h
// is the setting, at a third of the input size. Changing it is a one-line
// edit (then rebuild, rerun prepare_data, and train a fresh model).
// Grayscale images decode with one component and are replicated across all
// three channels here, so they need no special case downstream.

#include <math.h>
#include <stddef.h>
#include "image_prep.h"

// The crop is this much wider than the labeled box's longer side, so the
// network sees a little of the animal's surroundings (and all of the
// animal, when a box is drawn tight) rather than the box edge exactly.
#define CROP_MARGIN 1.2f

const char *class_names[CLASS_COUNT] = { "cat", "dog" };

void image_prep_crop(const uint8_t *pixels, int width, int height, int components,
                     float box_x, float box_y, float box_w, float box_h,
                     int flip, float *out)
{
  // Square crop centered on the box, shrunk to fit if it would be larger
  // than the image, then slid (not shrunk further) to stay inside it.
  float side = (box_w > box_h ? box_w : box_h) * CROP_MARGIN;
  float max_side = (float)(width < height ? width : height);
  if (side > max_side)
    side = max_side;
  if (side < 1.0f)
    side = 1.0f;
  float x0 = box_x + box_w * 0.5f - side * 0.5f;
  float y0 = box_y + box_h * 0.5f - side * 0.5f;
  if (x0 < 0.0f)
    x0 = 0.0f;
  if (y0 < 0.0f)
    y0 = 0.0f;
  if (x0 + side > (float)width)
    x0 = (float)width - side;
  if (y0 + side > (float)height)
    y0 = (float)height - side;

  // Area-average every source pixel that falls in each output cell, so a
  // large crop (hundreds of pixels across) downsamples without the
  // aliasing a nearest-neighbor/bilinear sample would have.
  const float cell = side / (float)IMG_SIZE;
  const int plane = IMG_SIZE * IMG_SIZE;
  for (int oy = 0; oy < IMG_SIZE; oy++) {
    int sy0 = (int)(y0 + oy * cell);
    int sy1 = (int)(y0 + (oy + 1) * cell);
    if (sy1 <= sy0)
      sy1 = sy0 + 1;
    if (sy1 > height)
      sy1 = height;
    for (int ox = 0; ox < IMG_SIZE; ox++) {
      int sx0 = (int)(x0 + ox * cell);
      int sx1 = (int)(x0 + (ox + 1) * cell);
      if (sx1 <= sx0)
        sx1 = sx0 + 1;
      if (sx1 > width)
        sx1 = width;
      float sum[3] = { 0.0f, 0.0f, 0.0f };
      int count = 0;
      for (int sy = sy0; sy < sy1; sy++) {
        const uint8_t *row = pixels + ((size_t)sy * width + sx0) * components;
        for (int sx = sx0; sx < sx1; sx++, row += components) {
          if (components >= 3) {
            sum[0] += row[0];
            sum[1] += row[1];
            sum[2] += row[2];
          } else {
            sum[0] += row[0];
            sum[1] += row[0];
            sum[2] += row[0];
          }
          count++;
        }
      }
      int dx = flip ? (IMG_SIZE - 1 - ox) : ox;
      float inv = count > 0 ? 1.0f / (float)count : 0.0f;
      if (IMG_CHANNELS == 1) {
        // ITU-R BT.601 luma
        out[oy * IMG_SIZE + dx] = (0.299f * sum[0] + 0.587f * sum[1] + 0.114f * sum[2]) * inv;
      } else {
        for (int c = 0; c < IMG_CHANNELS; c++)
          out[c * plane + oy * IMG_SIZE + dx] = sum[c] * inv;
      }
    }
  }

  // Per-image zero mean/unit variance -- the image counterpart of
  // audio_features' per-clip normalization in wake_word_detection: a
  // camera's frames swing from a dim hallway at night to a sunlit doorway,
  // and that overall brightness/contrast says nothing about which animal
  // is in the frame.
  float mean = 0.0f;
  for (int i = 0; i < IMG_INPUTS; i++)
    mean += out[i];
  mean /= (float)IMG_INPUTS;
  float var = 0.0f;
  for (int i = 0; i < IMG_INPUTS; i++)
    var += (out[i] - mean) * (out[i] - mean);
  float inv_std = 1.0f / sqrtf(var / (float)IMG_INPUTS + 1e-6f);
  for (int i = 0; i < IMG_INPUTS; i++)
    out[i] = (out[i] - mean) * inv_std;
}
