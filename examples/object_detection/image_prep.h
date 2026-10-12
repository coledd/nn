/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef IMAGE_PREP_H
#define IMAGE_PREP_H

#include <stdint.h>

// Turns a decoded microscope camera frame into the fixed-size, normalized
// network input train.c's model expects. Shared by prepare_data.c
// (building the training CSVs) and predict.c (counting the cells in one
// new image), so the two can't drift out of sync -- and, like
// ../image_classification/image_prep.[ch], this is the part a deployed
// device's firmware would run too, against its own frame buffer. No file
// I/O and no heap allocation; decoding a JPEG into the `pixels` buffer is
// the caller's job (../../stb_image.h here, the camera's own driver on a
// real target).
//
// Unlike ../image_classification, which crops a square around one animal,
// this keeps the whole frame: a detector has to look everywhere, and the
// frame's 4:3 shape is kept so cells aren't squashed.

#define IMG_W 64        // network input is IMG_W x IMG_H -- a multiple of GRID_STRIDE in both directions
#define IMG_H 48
#define IMG_CHANNELS 3  // RGB -- stain color is most of what tells the three cell types apart
#define IMG_INPUTS (IMG_W * IMG_H * IMG_CHANNELS)

// The network's output is one heatmap per class at 1/GRID_STRIDE of the
// input resolution (train.c pools twice): GRID_W x GRID_H cells, each
// covering GRID_STRIDE x GRID_STRIDE input pixels.
#define GRID_STRIDE 4
#define GRID_W (IMG_W / GRID_STRIDE)
#define GRID_H (IMG_H / GRID_STRIDE)
#define GRID_CELLS (GRID_W * GRID_H)

typedef enum {
  CLASS_RBC = 0,   // red blood cell
  CLASS_WBC,       // white blood cell
  CLASS_PLATELET,
  CLASS_COUNT
} cell_class_t;

#define HEATMAP_OUTPUTS (CLASS_COUNT * GRID_CELLS)

extern const char *class_names[CLASS_COUNT];

// Resizes the source rectangle (src_x, src_y, src_w, src_h) -- in `pixels`'
// own coordinates, top-left origin -- of a `width` x `height` image with
// `components` interleaved 8-bit channels per pixel (1 = gray, 3 = RGB) to
// IMG_W x IMG_H by area averaging, converts to IMG_CHANNELS, and normalizes
// each channel to zero mean/unit variance. Any part of the rectangle that
// falls outside the image repeats the nearest edge pixel. `flip_x`/`flip_y`
// mirror the result horizontally/vertically (training-time augmentation).
// Writes IMG_INPUTS floats to `out`, channel-major (CHW, the layout
// LAYER_TYPE_CNN expects). For a whole frame, pass (0, 0, width, height).
void image_prep_frame(const uint8_t *pixels, int width, int height, int components,
                      float src_x, float src_y, float src_w, float src_h,
                      int flip_x, int flip_y, float *out);

#endif /* IMAGE_PREP_H */
