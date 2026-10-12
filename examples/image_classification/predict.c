/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Classifies one image as cat or dog, the way an application would.
//
// Everything here except decoding the image file (stb_image.h -- a camera
// target already has raw pixels from its own sensor driver) is what
// firmware would do: crop the region of interest out of the frame, run it
// through image_prep_crop() (the exact preprocessing train.csv was built
// with), and hand the result to nn_predict(). On a deployed pet-door camera
// the region of interest would come from whatever noticed the animal -- a
// motion detector's changed-pixel bounding box, typically. Here it's given
// on the command line; without one, the whole frame's center square is
// used, which works when the animal fills most of the frame (as it does
// for a camera mounted right at a pet door, and in most of this dataset's
// photos).

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "image_prep.h"
#include "nn.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "stb_image.h"

int main(int argc, char *argv[]) {
  if (argc != 3 && argc != 7) {
    printf("Usage: %s <model-file> <image> [<x> <y> <width> <height>]\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c (ascii, binary, or inplace)\n");
    printf("  <image>      : Image to classify (JPEG, PNG, BMP, or GIF)\n");
    printf("  <x> <y> <width> <height> : Optional box around the animal, in the image's pixel\n");
    printf("                             coordinates (top-left origin); default is the whole image\n");
    return 1;
  }
  const char *model_path = argv[1];
  const char *image_path = argv[2];

  int width, height, components;
  uint8_t *pixels = stbi_load(image_path, &width, &height, &components, 0);
  if (!pixels) {
    fprintf(stderr, "Error: Could not read image %s: %s\n", image_path, stbi_failure_reason());
    return 1;
  }
  float box_x = 0.0f, box_y = 0.0f, box_w = (float)width, box_h = (float)height;
  if (argc == 7) {
    box_x = strtof(argv[3], NULL);
    box_y = strtof(argv[4], NULL);
    box_w = strtof(argv[5], NULL);
    box_h = strtof(argv[6], NULL);
  }
  float inputs[IMG_INPUTS];
  image_prep_crop(pixels, width, height, components, box_x, box_y, box_w, box_h, 0, inputs);
  stbi_image_free(pixels);

  // See examples/character_recognition/predict.c's identical block for why
  // the inplace format needs to be read into a buffer first.
  nn_model_format_t format = nn_model_format(model_path);
  nn_t *model;
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    inplace_buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!inplace_buf || fread(inplace_buf, 1, (size_t)size, file) != (size_t)size) {
      fprintf(stderr, "Error: Could not read model file: %s\n", model_path);
      fclose(file);
      free(inplace_buf);
      return 1;
    }
    fclose(file);
    model = nn_load_model_inplace(inplace_buf, (size_t)size);
  } else {
    model = nn_load_model(model_path);
  }
  if (model == NULL) {
    fprintf(stderr, "Error: Missing or invalid model file: %s\n", model_path);
    free(inplace_buf);
    return 1;
  }
  // nn_predict() has no way to know `inputs` is shorter than
  // model->width[0] and would simply read past its end, so check first.
  if (model->width[0] != IMG_INPUTS || model->width[model->depth - 1] != CLASS_COUNT) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match this task (%d inputs, %d outputs) -- is this a train.c model?\n",
            model->width[0], model->width[model->depth - 1], IMG_INPUTS, CLASS_COUNT);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }

  float *prediction = nn_predict(model, inputs);
  int best = 0;
  for (int c = 0; c < CLASS_COUNT; c++) {
    printf("%s: %.5f\n", class_names[c], prediction[c]);
    if (prediction[c] > prediction[best])
      best = c;
  }
  printf("Prediction: %s\n", class_names[best]);
  nn_free(model);
  free(inplace_buf);
  return 0;
}
