/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Counts the red blood cells, white blood cells, and platelets in one
// microscope photo, the way an application would.
//
// Everything here except decoding the image file (stb_image.h -- a
// microscope camera target already has raw pixels from its own sensor
// driver) is what firmware would do: run the frame through
// image_prep_frame() (the exact preprocessing train.csv was built with),
// hand the result to nn_predict(), and turn the heatmaps it returns into
// detections with heatmap_decode(). Prints each class's count, each
// detection's position in the photo's own pixel coordinates, and a map of
// the output grid, so you can see where in the frame each object was found.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "heatmap.h"
#include "image_prep.h"
#include "nn.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#define MAX_DETECTIONS 256

int main(int argc, char *argv[]) {
  if (argc != 3 && argc != 4) {
    printf("Usage: %s <model-file> <image> [<threshold>]\n", argv[0]);
    printf("  <model-file> : Path to a model saved by train.c (ascii, binary, or inplace)\n");
    printf("  <image>      : Microscope photo to count cells in (JPEG, PNG, or BMP)\n");
    printf("  <threshold>  : Heatmap peak threshold (default %.2f)\n", HEATMAP_THRESHOLD);
    return 1;
  }
  const char *model_path = argv[1];
  const char *image_path = argv[2];
  float threshold = argc == 4 ? strtof(argv[3], NULL) : HEATMAP_THRESHOLD;

  int width, height, components;
  uint8_t *pixels = stbi_load(image_path, &width, &height, &components, 0);
  if (!pixels) {
    fprintf(stderr, "Error: Could not read image %s: %s\n", image_path, stbi_failure_reason());
    return 1;
  }
  static float inputs[IMG_INPUTS];
  image_prep_frame(pixels, width, height, components, 0.0f, 0.0f, (float)width, (float)height, 0, 0, inputs);
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
  if (model->width[0] != IMG_INPUTS || model->width[model->depth - 1] != HEATMAP_OUTPUTS) {
    fprintf(stderr, "Error: Model dimensions (%u inputs, %u outputs) don't match this task (%d inputs, %d outputs) -- is this a train.c model?\n",
            model->width[0], model->width[model->depth - 1], IMG_INPUTS, HEATMAP_OUTPUTS);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }

  float *heatmap = nn_predict(model, inputs);
  static detection_t detections[MAX_DETECTIONS];
  int found = heatmap_decode(heatmap, threshold, detections, MAX_DETECTIONS);
  if (found > MAX_DETECTIONS)
    found = MAX_DETECTIONS;

  int counts[CLASS_COUNT] = { 0 };
  for (int d = 0; d < found; d++)
    counts[detections[d].cls]++;
  for (int c = 0; c < CLASS_COUNT; c++)
    printf("%s: %d\n", class_names[c], counts[c]);
  printf("\n");
  for (int d = 0; d < found; d++)
    printf("%-9s at (%4.0f, %4.0f)  score %.2f\n", class_names[detections[d].cls],
           detections[d].x * width, detections[d].y * height, detections[d].score);

  // One character per output grid cell: the first letter of the class
  // detected there (R, W, or P), or '.' for none.
  char grid[GRID_H][GRID_W];
  for (int gy = 0; gy < GRID_H; gy++)
    for (int gx = 0; gx < GRID_W; gx++)
      grid[gy][gx] = '.';
  static const char letters[CLASS_COUNT] = { 'R', 'W', 'P' };
  for (int d = 0; d < found; d++) {
    int gx = (int)(detections[d].x * GRID_W), gy = (int)(detections[d].y * GRID_H);
    if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H)
      grid[gy][gx] = letters[detections[d].cls];
  }
  printf("\n");
  for (int gy = 0; gy < GRID_H; gy++) {
    for (int gx = 0; gx < GRID_W; gx++)
      printf(gx + 1 < GRID_W ? "%c " : "%c\n", grid[gy][gx]);
  }

  nn_free(model);
  free(inplace_buf);
  return 0;
}
