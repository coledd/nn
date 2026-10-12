/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Data structure to hold the sample data
typedef struct {
  int num_rows;    // Number of rows of data
  int num_inputs;  // Number of inputs to neural network
  int num_outputs; // Number of outputs from neural network
  float **input;   // 2D array of inputs
  float **target;  // 2D array of targets
} data_t;

data_t *data_init(int num_rows, int num_inputs, int num_outputs);
void data_parse(data_t *data, char *line, int row);
int data_num_lines(FILE *file);
data_t *data_load(char *path, int num_inputs, int num_outputs);
void data_free(data_t *data);
void data_shuffle(data_t *data);
