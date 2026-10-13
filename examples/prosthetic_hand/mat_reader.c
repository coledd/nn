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
#include "mat_reader.h"

// Only stb_image's zlib decoder is used, but it can't be built on its own:
// PNG is the smallest image decoder that pulls it in. With no JPEG decoder
// built, a couple of its overflow helpers go unused.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "stb_image.h"
#pragma GCC diagnostic pop

// MATLAB Level 5 data types (the "miXXX" tags) used below.
enum {
  MI_INT8 = 1, MI_UINT8 = 2, MI_INT16 = 3, MI_UINT16 = 4, MI_INT32 = 5, MI_UINT32 = 6,
  MI_SINGLE = 7, MI_DOUBLE = 9, MI_INT64 = 12, MI_UINT64 = 13, MI_MATRIX = 14, MI_COMPRESSED = 15
};
#define MAT_HEADER_LEN 128

static uint32_t read_u32(const uint8_t *p)
{
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

// One data element: its type, payload, and the total bytes it occupies
// (header + payload + padding to 8 bytes). Elements of 4 bytes or less may
// use the "small data element" form, which packs the size into the upper
// half of the type field and the payload into the second word.
typedef struct {
  uint32_t type, size;
  const uint8_t *data;
  size_t total;
} element_t;

static int parse_element(const uint8_t *p, size_t avail, element_t *e)
{
  if (avail < 8)
    return -1;
  uint32_t tag = read_u32(p);
  if (tag >> 16) {
    e->type = tag & 0xffff;
    e->size = tag >> 16;
    e->data = p + 4;
    e->total = 8;
    return e->size <= 4 ? 0 : -1;
  }
  e->type = tag;
  e->size = read_u32(p + 4);
  e->data = p + 8;
  // Compressed elements aren't padded; everything else is, to 8 bytes.
  e->total = 8 + (e->type == MI_COMPRESSED ? e->size : ((size_t)e->size + 7) / 8 * 8);
  return e->total <= avail ? 0 : -1;
}

static float element_value(const element_t *e, size_t i)
{
  const uint8_t *p = e->data;
  switch (e->type) {
  case MI_INT8: return (float)((const int8_t *)p)[i];
  case MI_UINT8: return (float)p[i];
  case MI_INT16: { int16_t v; memcpy(&v, p + 2 * i, 2); return (float)v; }
  case MI_UINT16: { uint16_t v; memcpy(&v, p + 2 * i, 2); return (float)v; }
  case MI_INT32: { int32_t v; memcpy(&v, p + 4 * i, 4); return (float)v; }
  case MI_UINT32: { uint32_t v; memcpy(&v, p + 4 * i, 4); return (float)v; }
  case MI_SINGLE: { float v; memcpy(&v, p + 4 * i, 4); return v; }
  case MI_DOUBLE: { double v; memcpy(&v, p + 8 * i, 8); return (float)v; }
  case MI_INT64: { int64_t v; memcpy(&v, p + 8 * i, 8); return (float)v; }
  case MI_UINT64: { uint64_t v; memcpy(&v, p + 8 * i, 8); return (float)v; }
  default: return 0.0f;
  }
}

static size_t element_count(const element_t *e)
{
  switch (e->type) {
  case MI_INT8: case MI_UINT8: return e->size;
  case MI_INT16: case MI_UINT16: return e->size / 2;
  case MI_INT32: case MI_UINT32: case MI_SINGLE: return e->size / 4;
  case MI_DOUBLE: case MI_INT64: case MI_UINT64: return e->size / 8;
  default: return 0;
  }
}

// Parses one miMATRIX element's payload: array flags, dimensions, name,
// real part. If its name is one of `names`, stores it in `out`.
static void parse_matrix(const uint8_t *p, size_t len, const char **names, int count, mat_var_t *out)
{
  element_t flags, dims, name, real;
  size_t pos = 0;
  if (parse_element(p, len, &flags) != 0)
    return;
  pos += flags.total;
  if (parse_element(p + pos, len - pos, &dims) != 0 || dims.size != 8)
    return; // only 2-D matrices
  pos += dims.total;
  if (parse_element(p + pos, len - pos, &name) != 0)
    return;
  pos += name.total;
  for (int v = 0; v < count; v++) {
    if (out[v].data != NULL || strlen(names[v]) != name.size || memcmp(names[v], name.data, name.size) != 0)
      continue;
    if (parse_element(p + pos, len - pos, &real) != 0)
      return;
    int rows = (int)read_u32(dims.data), cols = (int)read_u32(dims.data + 4);
    if (element_count(&real) != (size_t)rows * (size_t)cols)
      return;
    float *data = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    if (data == NULL)
      return;
    for (int c = 0; c < cols; c++)
      for (int r = 0; r < rows; r++)
        data[(size_t)r * cols + c] = element_value(&real, (size_t)c * rows + r);
    snprintf(out[v].name, sizeof(out[v].name), "%s", names[v]);
    out[v].rows = rows;
    out[v].cols = cols;
    out[v].data = data;
    return;
  }
}

int mat_read(const char *path, const char **names, int count, mat_var_t *out)
{
  memset(out, 0, (size_t)count * sizeof(*out));
  FILE *file = fopen(path, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error: Could not open %s\n", path);
    return -1;
  }
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  uint8_t *buf = size > MAT_HEADER_LEN ? (uint8_t *)malloc((size_t)size) : NULL;
  if (buf == NULL || fread(buf, 1, (size_t)size, file) != (size_t)size) {
    fprintf(stderr, "Error: Could not read %s\n", path);
    fclose(file);
    free(buf);
    return -1;
  }
  fclose(file);
  if (memcmp(buf, "MATLAB 5.0 MAT-file", 19) != 0 || buf[126] != 'I' || buf[127] != 'M') {
    fprintf(stderr, "Error: %s is not a little-endian MATLAB Level 5 file\n", path);
    free(buf);
    return -1;
  }

  size_t pos = MAT_HEADER_LEN;
  element_t e;
  while (pos < (size_t)size && parse_element(buf + pos, (size_t)size - pos, &e) == 0) {
    if (e.type == MI_MATRIX) {
      parse_matrix(e.data, e.size, names, count, out);
    } else if (e.type == MI_COMPRESSED) {
      int inflated_len = 0;
      // A compressed element is exactly one zlib stream holding one plain
      // element (in practice always an miMATRIX).
      uint8_t *inflated = (uint8_t *)stbi_zlib_decode_malloc((const char *)e.data, (int)e.size, &inflated_len);
      element_t inner;
      if (inflated != NULL && parse_element(inflated, (size_t)inflated_len, &inner) == 0 && inner.type == MI_MATRIX)
        parse_matrix(inner.data, inner.size, names, count, out);
      free(inflated);
    }
    pos += e.total;
  }
  free(buf);

  for (int v = 0; v < count; v++) {
    if (out[v].data == NULL) {
      fprintf(stderr, "Error: %s has no numeric matrix named '%s'\n", path, names[v]);
      for (int w = 0; w < count; w++)
        mat_var_free(&out[w]);
      return -1;
    }
  }
  return 0;
}

void mat_var_free(mat_var_t *var)
{
  free(var->data);
  var->data = NULL;
}
