/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Converts the original MNIST files (LeCun, Cortes & Burges; CC BY-SA 3.0)
// -- four IDX files: 60,000 training and 10,000 test images of 28x28
// 8-bit grayscale digits, and their labels -- into samples.csv, one row
// per image:
//
//   784 pixels, row by row, each 1 if its gray level is 128 or more and 0
//   otherwise, then the label one-hot over 10 columns
//
// Rows end with a trailing comma (data_prep.c ignores it), the format this
// example has always used. All 70,000 images go into one file, training
// set first; ../../split.py shuffles them and makes its own
// train/validation/test split, so MNIST's own split isn't kept.
//
// The pixels are binarized rather than scaled to 0-1 so that predict.c's
// hardcoded sample and any model already trained on this example's data
// keep working unchanged.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_SIZE (28 * 28)
#define NUM_CLASSES 10
#define THRESHOLD 128

// Reads a whole IDX file and checks its magic number (2051 for images,
// 2049 for labels). Returns the buffer and sets *count to the number of
// items, or returns NULL with a message on stderr.
static uint8_t *read_idx(const char *dir, const char *name, uint32_t magic, uint32_t *count, size_t *header_len)
{
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error: Could not open %s\n", path);
    return NULL;
  }
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  uint8_t *buf = size > 8 ? (uint8_t *)malloc((size_t)size) : NULL;
  if (buf == NULL || fread(buf, 1, (size_t)size, file) != (size_t)size) {
    fprintf(stderr, "Error: Could not read %s\n", path);
    fclose(file);
    free(buf);
    return NULL;
  }
  fclose(file);
  // IDX headers are big-endian: magic number, item count, then (for
  // images) row and column counts.
  uint32_t found = (uint32_t)buf[0] << 24 | (uint32_t)buf[1] << 16 | (uint32_t)buf[2] << 8 | buf[3];
  *count = (uint32_t)buf[4] << 24 | (uint32_t)buf[5] << 16 | (uint32_t)buf[6] << 8 | buf[7];
  *header_len = magic == 2051 ? 16 : 8;
  size_t item_len = magic == 2051 ? IMAGE_SIZE : 1;
  if (found != magic || (size_t)size != *header_len + (size_t)*count * item_len) {
    fprintf(stderr, "Error: %s is not an MNIST %s file\n", path, magic == 2051 ? "image" : "label");
    free(buf);
    return NULL;
  }
  return buf;
}

static int write_set(FILE *out, const char *dir, const char *images_name, const char *labels_name)
{
  uint32_t num_images, num_labels;
  size_t images_header, labels_header;
  uint8_t *images = read_idx(dir, images_name, 2051, &num_images, &images_header);
  uint8_t *labels = read_idx(dir, labels_name, 2049, &num_labels, &labels_header);
  if (images == NULL || labels == NULL || num_images != num_labels) {
    if (images != NULL && labels != NULL)
      fprintf(stderr, "Error: %s has %u images but %s has %u labels\n", images_name, num_images, labels_name,
              num_labels);
    free(images);
    free(labels);
    return -1;
  }
  for (uint32_t i = 0; i < num_images; i++) {
    const uint8_t *pixels = images + images_header + (size_t)i * IMAGE_SIZE;
    for (int p = 0; p < IMAGE_SIZE; p++)
      fputs(pixels[p] >= THRESHOLD ? "1," : "0,", out);
    int label = labels[labels_header + i];
    for (int c = 0; c < NUM_CLASSES; c++)
      fputs(c == label ? "1," : "0,", out);
    fputc('\n', out);
  }
  printf("%s: %u images\n", images_name, num_images);
  free(images);
  free(labels);
  return 0;
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <mnist-dir> <output.csv>\n", argv[0]);
    printf("  <mnist-dir> : directory holding the four uncompressed MNIST IDX files\n");
    return 1;
  }
  FILE *out = fopen(argv[2], "w");
  if (out == NULL) {
    fprintf(stderr, "Error: Could not create %s\n", argv[2]);
    return 1;
  }
  if (write_set(out, argv[1], "train-images-idx3-ubyte", "train-labels-idx1-ubyte") != 0 ||
      write_set(out, argv[1], "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte") != 0) {
    fclose(out);
    remove(argv[2]);
    return 1;
  }
  fclose(out);
  return 0;
}
