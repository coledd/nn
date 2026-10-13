/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MAT_READER_H
#define MAT_READER_H

// Desktop-only reader for the MATLAB files Ninapro publishes its
// recordings in -- the same category as ../../wav_reader.[ch]: a file
// format, not anything a prosthesis would run. Firmware gets samples from
// its own analog front end, never a .mat file.
//
// Handles exactly the subset of the MATLAB Level 5 format those files use:
// top-level numeric matrices (any real numeric class), stored either plain
// or zlib-compressed (the default since MATLAB 7). Compression is undone
// with the zlib decoder in ../../stb_image.h, so there's no new dependency.
// Cell arrays, structs, sparse and complex matrices, and the HDF5-based
// v7.3 format are not supported.

typedef struct {
  char name[64];
  int rows, cols;
  float *data; // rows x cols, converted to float and stored row-major (MATLAB stores column-major)
} mat_var_t;

// Reads the variables listed in `names` (count of them) from `path` into
// `out` (same order). Returns 0 on success, or -1 -- with a message on
// stderr -- if the file can't be read or any variable is missing.
int mat_read(const char *path, const char **names, int count, mat_var_t *out);

void mat_var_free(mat_var_t *var);

#endif /* MAT_READER_H */
