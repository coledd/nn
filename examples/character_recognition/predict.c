/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"

// Must match train_cnn.c's/train_fc.c's num_inputs (28x28, MNIST-style) --
// both build a model with the same input shape, just a different
// architecture behind it. nn_predict() has no way to know the caller's
// buffer is shorter than model->width[0] and would simply read past its
// end, so it's on the caller to size this correctly.
#define IMG_SIZE 28

// Test data upon which to make a prediction: a rough digit-like blob,
// centered with a symmetric zero margin, the same way a real MNIST digit
// sits within its 28x28 frame. Not a real handwritten sample -- just enough
// structure to exercise the model and see plausible-looking output.
static const float sample[IMG_SIZE][IMG_SIZE] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
};

int main(int argc, char *argv[]) {
  nn_t *model;
  const char *model_path = (argc > 1) ? argv[1] : "model.txt";

  // nn_predict() takes a non-const float*, so this needs to be a mutable
  // copy of `sample` rather than a pointer straight at it (even though
  // nn_predict() never actually writes through it) -- &sample[0][0] would
  // silently discard the const qualifier instead.
  float inputs[IMG_SIZE * IMG_SIZE];
  memcpy(inputs, sample, sizeof(inputs));

  // Recall a previously trained neural network model, inclusive of its
  // weights (ascii, binary, or inplace, auto-detected). The inplace format
  // has to be read from a memory buffer rather than a path (see
  // nn_load_model_inplace() in nn.h), so read the whole file ourselves
  // first when nn_model_format() reports that's what it is; the buffer
  // must then outlive `model`. This tool is read-only (just nn_predict()),
  // so the plain zero-copy loader is enough -- no need for the mutable
  // nn_load_model_inplace_copy() that quantize/dequantize/prune use.
  nn_model_format_t format = nn_model_format(model_path);
  uint8_t *inplace_buf = NULL;
  if (format == NN_MODEL_FORMAT_INPLACE) {
    FILE *file = fopen(model_path, "rb");
    if (!file) {
      printf("Error: Missing or invalid model file.\n");
      return 1;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    inplace_buf = size > 0 ? (uint8_t *)malloc((size_t)size) : NULL;
    if (!inplace_buf || fread(inplace_buf, 1, (size_t)size, file) != (size_t)size) {
      printf("Error: Could not read model file.\n");
      fclose(file);
      free(inplace_buf);
      return 1;
    }
    fclose(file);
    model = nn_load_model_inplace(inplace_buf, (size_t)size);
  } else {
    model = nn_load_model(model_path);
  }
  if (NULL == model) {
    printf("Error: Missing or invalid model file.\n");
    free(inplace_buf);
    return 1;
  }
  // Defensive check: nn_predict() has no way to know `inputs` is shorter
  // than model->width[0] and would simply read past its end, so verify the
  // sizes actually match instead of risking a silent out-of-bounds read if
  // this is ever pointed at a model with a different input width.
  if (model->width[0] != IMG_SIZE * IMG_SIZE) {
    fprintf(stderr, "Error: model expects %u inputs, but this program's sample input is %d.\n",
            model->width[0], IMG_SIZE * IMG_SIZE);
    nn_free(model);
    free(inplace_buf);
    return 1;
  }
  // Make an output prediction based upon new input data
  float *prediction = nn_predict(model, inputs);
  for (int i = 0; i < model->width[model->depth - 1]; i++)
    printf("%d: %.5f\n", i, prediction[i]);
  nn_free(model);
  free(inplace_buf);
  return 0;
}
