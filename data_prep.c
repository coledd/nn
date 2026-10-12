/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_prep.h"

// Allocates a data structure for the sample data
data_t *data_init(int num_rows, int num_inputs, int num_outputs)
{
  data_t *data = NULL;

  data = (data_t *)malloc(sizeof(data_t));
  if (NULL == data)
    return NULL;
  data->num_rows = num_rows;
  data->num_inputs = num_inputs;
  data->num_outputs = num_outputs;
  data->input = (float **)malloc(data->num_rows * sizeof(float *));
  if (NULL == data->input) {
    free(data);
    return NULL;
  }
  data->target = (float **)malloc(data->num_rows * sizeof(float *));
  if (NULL == data->target) {
    free(data->input);
    free(data);
    return NULL;
  }
  for (int i = 0; i < data->num_rows; i++) {
    data->input[i] = (float *)malloc(data->num_inputs * sizeof(float));
    if (NULL == data->input[i]) {
      while (--i >= 0) {
        free(data->input[i]);
        free(data->target[i]);
      }
      free(data->input);
      free(data->target);
      free(data);
      return NULL;
    }
    data->target[i] = (float *)malloc(data->num_outputs * sizeof(float));
    if (NULL == data->target[i]) {
      free(data->input[i]);
      while (--i >= 0) {
        free(data->input[i]);
        free(data->target[i]);
      }
      free(data->input);
      free(data->target);
      free(data);
      return NULL;
    }
  }
  return data;
}

// Splits inputs and outputs into two separate data tables within the data
// structure
void data_parse(data_t *data, char *line, int row)
{
  for (int column = 0; column < (data->num_inputs + data->num_outputs);
       column++) {
    float val = atof(strtok(column == 0 ? line : NULL, ","));
    if (column < data->num_inputs)
      data->input[row][column] = val;
    else
      data->target[row][column - data->num_inputs] = val;
  }
}

// Returns the number of lines in a file
int data_num_lines(FILE *file)
{
  int lines = 0;
  int c = EOF;
  int previous_c = '\n';

  while ((c = getc(file)) != EOF) {
    if (c == '\n')
      lines++;
    previous_c = c;
  }
  if (previous_c != '\n')
    lines++;
  rewind(file);
  return lines;
}

// Parses file from path getting all inputs and outputs for the neural network.
// Returns the data in a data structure. Reads the file in a single pass,
// growing the row arrays (capacity doubling, like a typical dynamic array)
// as needed, rather than first scanning the whole file just to count lines
// (data_num_lines) and then rewinding to re-read it from the start -- for a
// large CSV that halves the I/O and parsing work done at load time.
data_t *data_load(char *path, int num_inputs, int num_outputs)
{
  FILE *file;
  char *line = NULL;
  size_t len = 0;
  data_t *data;
  int capacity;
  int row;

  file = fopen(path, "r");
  if (file == NULL) {
    return NULL;
  }
  data = (data_t *)malloc(sizeof(data_t));
  if (data == NULL) {
    fclose(file);
    return NULL;
  }
  data->num_rows = 0;
  data->num_inputs = num_inputs;
  data->num_outputs = num_outputs;
  capacity = 1024;
  data->input = (float **)malloc(capacity * sizeof(float *));
  data->target = (float **)malloc(capacity * sizeof(float *));
  if (data->input == NULL || data->target == NULL) {
    free(data->input);
    free(data->target);
    free(data);
    fclose(file);
    return NULL;
  }
  row = 0;
  while (getline(&line, &len, file) != -1) {
    if (row == capacity) {
      int new_capacity = capacity * 2;
      // Reassign only on success -- on failure, data->input/data->target
      // still point at their original (still valid, still owned) blocks,
      // so data_free() below can safely tear everything down either way.
      float **grown_input = (float **)realloc(data->input, new_capacity * sizeof(float *));
      if (grown_input != NULL)
        data->input = grown_input;
      float **grown_target = (float **)realloc(data->target, new_capacity * sizeof(float *));
      if (grown_target != NULL)
        data->target = grown_target;
      if (grown_input == NULL || grown_target == NULL) {
        data->num_rows = row;
        free(line);
        fclose(file);
        data_free(data);
        return NULL;
      }
      capacity = new_capacity;
    }
    data->input[row] = (float *)malloc(num_inputs * sizeof(float));
    data->target[row] = (float *)malloc(num_outputs * sizeof(float));
    if (data->input[row] == NULL || data->target[row] == NULL) {
      free(data->input[row]);
      free(data->target[row]);
      data->num_rows = row; // rows [0, row) are fully allocated; row itself just got freed above
      free(line);
      fclose(file);
      data_free(data);
      return NULL;
    }
    data_parse(data, line, row);
    row++;
  }
  free(line);
  fclose(file);
  data->num_rows = row;
  // Shrink the pointer arrays down to the exact row count actually read.
  if (row > 0) {
    float **shrunk_input = (float **)realloc(data->input, row * sizeof(float *));
    if (shrunk_input != NULL)
      data->input = shrunk_input;
    float **shrunk_target = (float **)realloc(data->target, row * sizeof(float *));
    if (shrunk_target != NULL)
      data->target = shrunk_target;
  }
  return data;
}

// Frees a data object
void data_free(data_t *data)
{
  for (int row = 0; row < data->num_rows; row++) {
    free(data->input[row]);
    free(data->target[row]);
  }
  free(data->input);
  free(data->target);
  free(data);
}

// Randomly shuffles the rows of a data object using the Fisher-Yates
// algorithm, which produces a uniformly random permutation. Each index i is
// only ever swapped with an index in [0, i] (its own unshuffled prefix),
// unlike picking j from the full range every time, which is a common but
// biased ("naive shuffle") mistake.
void data_shuffle(data_t *data)
{
  float *input, *output;

  for (int i = data->num_rows - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    // Swap target
    output = data->target[i];
    data->target[i] = data->target[j];
    data->target[j] = output;
    // Swap input
    input = data->input[i];
    data->input[i] = data->input[j];
    data->input[j] = input;
  }
}
