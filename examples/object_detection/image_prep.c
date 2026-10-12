/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// See image_prep.h for what this is and why it's shared.

#include <math.h>
#include <stddef.h>
#include "image_prep.h"

const char *class_names[CLASS_COUNT] = { "RBC", "WBC", "platelet" };

static int clamp(int v, int lo, int hi)
{
  return v < lo ? lo : (v > hi ? hi : v);
}

void image_prep_frame(const uint8_t *pixels, int width, int height, int components,
                      float src_x, float src_y, float src_w, float src_h,
                      int flip_x, int flip_y, float *out)
{
  // Area-average every source pixel that falls in each output pixel, so a
  // 640x480 frame downsamples ~10x without the aliasing a nearest-
  // neighbor/bilinear sample would have.
  const float cell_w = src_w / (float)IMG_W;
  const float cell_h = src_h / (float)IMG_H;
  const int plane = IMG_W * IMG_H;
  for (int oy = 0; oy < IMG_H; oy++) {
    int sy0 = (int)floorf(src_y + oy * cell_h);
    int sy1 = (int)floorf(src_y + (oy + 1) * cell_h);
    if (sy1 <= sy0)
      sy1 = sy0 + 1;
    for (int ox = 0; ox < IMG_W; ox++) {
      int sx0 = (int)floorf(src_x + ox * cell_w);
      int sx1 = (int)floorf(src_x + (ox + 1) * cell_w);
      if (sx1 <= sx0)
        sx1 = sx0 + 1;
      float sum[3] = { 0.0f, 0.0f, 0.0f };
      int count = 0;
      for (int sy = sy0; sy < sy1; sy++) {
        const uint8_t *row = pixels + (size_t)clamp(sy, 0, height - 1) * width * components;
        for (int sx = sx0; sx < sx1; sx++) {
          const uint8_t *p = row + (size_t)clamp(sx, 0, width - 1) * components;
          if (components >= 3) {
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
          } else {
            sum[0] += p[0];
            sum[1] += p[0];
            sum[2] += p[0];
          }
          count++;
        }
      }
      int dx = flip_x ? (IMG_W - 1 - ox) : ox;
      int dy = flip_y ? (IMG_H - 1 - oy) : oy;
      float inv = 1.0f / (float)count;
      if (IMG_CHANNELS == 1) {
        // ITU-R BT.601 luma
        out[dy * IMG_W + dx] = (0.299f * sum[0] + 0.587f * sum[1] + 0.114f * sum[2]) * inv;
      } else {
        for (int c = 0; c < IMG_CHANNELS; c++)
          out[c * plane + dy * IMG_W + dx] = sum[c] * inv;
      }
    }
  }

  // Per-channel zero mean/unit variance. Per channel rather than over the
  // whole image (as ../image_classification does) because what varies
  // between slides and microscopes is mostly color: how strongly a smear
  // took up its stain, and the lamp's color temperature. Normalizing each
  // channel separately removes that overall cast while keeping what tells
  // cells apart -- a WBC nucleus is still far more purple than the pink
  // RBCs around it, relative to that frame's own average.
  for (int c = 0; c < IMG_CHANNELS; c++) {
    float *ch = out + c * plane;
    float mean = 0.0f;
    for (int i = 0; i < plane; i++)
      mean += ch[i];
    mean /= (float)plane;
    float var = 0.0f;
    for (int i = 0; i < plane; i++)
      var += (ch[i] - mean) * (ch[i] - mean);
    float inv_std = 1.0f / sqrtf(var / (float)plane + 1e-6f);
    for (int i = 0; i < plane; i++)
      ch[i] = (ch[i] - mean) * inv_std;
  }
}
