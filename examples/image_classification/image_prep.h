/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREP_H
#define IMAGE_PREP_H

#include <stdint.h>

// Turns a decoded camera frame plus a region of interest (where the animal
// is) into the fixed-size, normalized network input train.c's model
// expects. Shared by prepare_data.c (building the training CSVs) and
// predict.c (classifying one new image), so the two can't drift out of sync
// -- and, like ../../audio_features.[ch] for wake_word_detection, this is
// the part a deployed camera's firmware would run too, against its own
// frame buffer, not a training-time-only shortcut. No file I/O and no heap
// allocation; decoding a JPEG into the `pixels` buffer is the caller's job
// (stb_image.h here, a camera's own driver/ISP on a real target).

#define IMG_SIZE 48     // network input is IMG_SIZE x IMG_SIZE -- must be a multiple of 16 (train.c halves it four times)
#define IMG_CHANNELS 3  // RGB -- see image_prep.c's top comment for why
#define IMG_INPUTS (IMG_SIZE * IMG_SIZE * IMG_CHANNELS)

typedef enum {
  CLASS_CAT = 0,
  CLASS_DOG,
  CLASS_COUNT
} animal_class_t;

extern const char *class_names[CLASS_COUNT];

// Crops a square region centered on the box (box_x, box_y, box_w, box_h)
// -- in `pixels`' own coordinates, top-left origin -- out of a `width` x
// `height` image with `components` interleaved 8-bit channels per pixel (1 =
// gray, 3 = RGB), resizes it to IMG_SIZE x IMG_SIZE by area averaging,
// converts to IMG_CHANNELS, and normalizes it to zero mean/unit variance.
// `flip` mirrors it left-to-right (training-time augmentation). Writes
// IMG_INPUTS floats to `out`, channel-major (CHW, the layout LAYER_TYPE_CNN
// expects).
void image_prep_crop(const uint8_t *pixels, int width, int height, int components,
                     float box_x, float box_y, float box_w, float box_h,
                     int flip, float *out);

#endif /* IMAGE_PREP_H */
