/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Builds this example's training/evaluation CSVs from an extracted copy of
// the BCCD (Blood Cell Count and Detection) dataset
// (https://github.com/Shenggan/BCCD_Dataset; MIT license): 364 640x480
// microscope photos of stained blood smears, with a Pascal VOC XML file per
// photo boxing each red blood cell (RBC), white blood cell (WBC), and
// platelet, and a published train/val/test split.
//
// Each photo becomes one row (or, for train.csv, several augmented rows):
// IMG_INPUTS normalized pixels (see image_prep.[ch] -- the whole frame,
// resized and normalized exactly the way predict.c and a deployed device
// would), followed by HEATMAP_OUTPUTS targets: CLASS_COUNT heatmaps of
// GRID_W x GRID_H cells, 1 in the cell containing each labeled object's
// center and 0 everywhere else.
//
//   train.csv            <- ImageSets/Main/train.txt, augmented (TRAIN_COPIES rows per photo)
//   validation.csv       <- ImageSets/Main/val.txt
//   test.csv             <- ImageSets/Main/test.txt
//   validation_boxes.txt <- one line per validation.csv row: the photo's
//   test_boxes.txt          name, size, and every labeled box, in its
//                           original pixel coordinates, for test.c to match
//                           detections against (a heatmap target alone
//                           can't say how big each object is, or that two
//                           objects landed in the same cell)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image_prep.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO_WARNINGS
#include "stb_image.h"

#define MAX_PATH_LEN 4096
#define MAX_NAME_LEN 256
#define MAX_OBJECTS 128
// Training rows per photo, and how far an augmented copy may be shifted
// (as a fraction of the frame size) and rescaled. See write_photo().
#define TRAIN_COPIES 32
#define JITTER 0.1f
// Heatmap target spread, in grid cells: 0 marks only the cell containing
// each object's center (1, every other cell 0); above 0, cells around it
// get exp(-d^2 / (2 * TARGET_SIGMA^2)) of the way to 1, d being their
// distance from the object's exact center, in cells.
#ifndef TARGET_SIGMA
#define TARGET_SIGMA 0.7f
#endif

typedef struct {
  int cls;                // cell_class_t
  float x0, y0, x1, y1;   // box, in the photo's pixel coordinates
} box_t;

// Reads the text between the first <tag> and </tag> at or after `from` and
// before `end`, into `out`. Returns a pointer just past </tag>, or NULL.
static const char *xml_field(const char *from, const char *end, const char *tag, char *out, size_t out_size)
{
  char open[64], close[64];
  snprintf(open, sizeof(open), "<%s>", tag);
  snprintf(close, sizeof(close), "</%s>", tag);
  const char *start = strstr(from, open);
  if (!start || start >= end)
    return NULL;
  start += strlen(open);
  const char *stop = strstr(start, close);
  if (!stop || stop > end)
    return NULL;
  size_t len = (size_t)(stop - start);
  if (len >= out_size)
    len = out_size - 1;
  memcpy(out, start, len);
  out[len] = '\0';
  return stop + strlen(close);
}

// Parses one BCCD annotation file into `boxes`. Returns the number of
// boxes, or -1 if the file can't be read. Skips (and counts in *skipped)
// the two degenerate zero-width/height boxes the dataset contains.
static int read_boxes(const char *path, box_t *boxes, int max_boxes, int *skipped)
{
  FILE *file = fopen(path, "rb");
  if (!file)
    return -1;
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *xml = malloc((size_t)size + 1);
  if (!xml || fread(xml, 1, (size_t)size, file) != (size_t)size) {
    fclose(file);
    free(xml);
    return -1;
  }
  fclose(file);
  xml[size] = '\0';
  int count = 0;
  const char *p = xml;
  const char *object;
  while ((object = strstr(p, "<object>")) != NULL) {
    const char *end = strstr(object, "</object>");
    if (!end)
      break;
    p = end + strlen("</object>");
    char name[MAX_NAME_LEN], value[4][32];
    static const char *coords[4] = { "xmin", "ymin", "xmax", "ymax" };
    if (!xml_field(object, end, "name", name, sizeof(name)))
      continue;
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++)
      ok = xml_field(object, end, coords[i], value[i], sizeof(value[i])) != NULL;
    if (!ok)
      continue;
    int cls;
    if (strcmp(name, "RBC") == 0)
      cls = CLASS_RBC;
    else if (strcmp(name, "WBC") == 0)
      cls = CLASS_WBC;
    else if (strcmp(name, "Platelets") == 0)
      cls = CLASS_PLATELET;
    else {
      fprintf(stderr, "Warning: %s: unknown class \"%s\" skipped\n", path, name);
      continue;
    }
    box_t b = { cls, strtof(value[0], NULL), strtof(value[1], NULL), strtof(value[2], NULL), strtof(value[3], NULL) };
    if (b.x1 <= b.x0 || b.y1 <= b.y0) {
      (*skipped)++;
      continue;
    }
    if (count < max_boxes)
      boxes[count++] = b;
  }
  free(xml);
  return count;
}

// Uniform random float in [lo, hi).
static float rand_range(float lo, float hi)
{
  return lo + (hi - lo) * ((float)rand() / ((float)RAND_MAX + 1.0f));
}

// Writes `copies` rows for one photo to `out` (and, if `boxes_out` is
// non-NULL, its boxes to that). Copies 0-3 are the photo as-is and its
// three mirror images (left-right, top-bottom, and both, which is a 180
// degree rotation) -- a blood smear has no "up", so all four are equally
// valid views of it. Every later copy is randomly mirrored, shifted by up
// to +/-JITTER of the frame size, and rescaled by up to +/-JITTER, as if
// the slide had been moved or the focus nudged. A 90 degree rotation would
// be just as valid, but would turn the 4:3 frame into a 3:4 one.
// Returns 1 on success, 0 if the photo or its annotation couldn't be read.
static int write_photo(const char *dataset_dir, const char *name, int copies, FILE *out, FILE *boxes_out,
                       int *object_counts, int *skipped)
{
  static float pixels[IMG_INPUTS];
  static float heatmap[HEATMAP_OUTPUTS];
  static box_t boxes[MAX_OBJECTS];
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "%s/Annotations/%s.xml", dataset_dir, name);
  int num_boxes = read_boxes(path, boxes, MAX_OBJECTS, skipped);
  if (num_boxes < 0) {
    fprintf(stderr, "Error: Could not read %s\n", path);
    return 0;
  }
  snprintf(path, sizeof(path), "%s/JPEGImages/%s.jpg", dataset_dir, name);
  int w, h, components;
  uint8_t *img = stbi_load(path, &w, &h, &components, 0);
  if (!img) {
    fprintf(stderr, "Error: Could not read %s: %s\n", path, stbi_failure_reason());
    return 0;
  }
  for (int b = 0; b < num_boxes; b++)
    object_counts[boxes[b].cls]++;
  if (boxes_out) {
    fprintf(boxes_out, "%s %d %d %d", name, w, h, num_boxes);
    for (int b = 0; b < num_boxes; b++)
      fprintf(boxes_out, " %d %g %g %g %g", boxes[b].cls, boxes[b].x0, boxes[b].y0, boxes[b].x1, boxes[b].y1);
    fprintf(boxes_out, "\n");
  }
  for (int k = 0; k < copies; k++) {
    float sx = 0.0f, sy = 0.0f, sw = (float)w, sh = (float)h;
    int flip_x = k & 1, flip_y = (k >> 1) & 1;
    if (k >= 4) {
      float scale = rand_range(1.0f - JITTER, 1.0f + JITTER);
      sw = w * scale;
      sh = h * scale;
      sx = (w - sw) * 0.5f + rand_range(-JITTER, JITTER) * w;
      sy = (h - sh) * 0.5f + rand_range(-JITTER, JITTER) * h;
      flip_x = rand() & 1;
      flip_y = rand() & 1;
    }
    image_prep_frame(img, w, h, components, sx, sy, sw, sh, flip_x, flip_y, pixels);
    memset(heatmap, 0, sizeof(heatmap));
    for (int b = 0; b < num_boxes; b++) {
      // Box center, as a fraction of the (shifted, rescaled, mirrored)
      // frame. An object whose center the shift pushed out of frame isn't
      // in this view at all.
      float u = ((boxes[b].x0 + boxes[b].x1) * 0.5f - sx) / sw;
      float v = ((boxes[b].y0 + boxes[b].y1) * 0.5f - sy) / sh;
      if (u < 0.0f || u >= 1.0f || v < 0.0f || v >= 1.0f)
        continue;
      if (flip_x)
        u = 1.0f - u;
      if (flip_y)
        v = 1.0f - v;
      int gx = (int)(u * GRID_W), gy = (int)(v * GRID_H);
      if (gx >= GRID_W)
        gx = GRID_W - 1;
      if (gy >= GRID_H)
        gy = GRID_H - 1;
      float *map = heatmap + boxes[b].cls * GRID_CELLS;
      map[gy * GRID_W + gx] = 1.0f;
      if (TARGET_SIGMA > 0.0f) {
        for (int ny = gy - 2; ny <= gy + 2; ny++) {
          for (int nx = gx - 2; nx <= gx + 2; nx++) {
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H || (nx == gx && ny == gy))
              continue;
            float dx = nx + 0.5f - u * GRID_W, dy = ny + 0.5f - v * GRID_H;
            float t = expf(-(dx * dx + dy * dy) / (2.0f * TARGET_SIGMA * TARGET_SIGMA));
            // Where two objects' spreads overlap, the higher one wins, so
            // a neighbor never ends up looking more like a center than
            // either object's own cell.
            if (t > map[ny * GRID_W + nx])
              map[ny * GRID_W + nx] = t;
          }
        }
      }
    }
    for (int i = 0; i < IMG_INPUTS; i++)
      fprintf(out, "%.4f,", pixels[i]);
    for (int i = 0; i < HEATMAP_OUTPUTS; i++)
      fprintf(out, i + 1 < HEATMAP_OUTPUTS ? "%g," : "%g\n", heatmap[i]);
  }
  stbi_image_free(img);
  return 1;
}

// Writes every photo listed in ImageSets/Main/<list_name> to `csv_path`
// (and, if `boxes_path` is non-NULL, their boxes to it). Returns 0, or -1
// on an error.
static int write_split(const char *dataset_dir, const char *list_name, int copies,
                       const char *csv_path, const char *boxes_path)
{
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "%s/ImageSets/Main/%s", dataset_dir, list_name);
  FILE *list = fopen(path, "r");
  if (!list) {
    fprintf(stderr, "Error: Could not read %s\n", path);
    return -1;
  }
  FILE *out = fopen(csv_path, "w");
  FILE *boxes_out = boxes_path ? fopen(boxes_path, "w") : NULL;
  if (!out || (boxes_path && !boxes_out)) {
    fprintf(stderr, "Error: Could not create %s\n", !out ? csv_path : boxes_path);
    fclose(list);
    if (out)
      fclose(out);
    return -1;
  }
  int photos = 0, skipped = 0, status = 0;
  int object_counts[CLASS_COUNT] = { 0 };
  char name[MAX_NAME_LEN];
  while (fscanf(list, "%255s", name) == 1) {
    if (!write_photo(dataset_dir, name, copies, out, boxes_out, object_counts, &skipped)) {
      status = -1;
      break;
    }
    photos++;
  }
  fclose(list);
  fclose(out);
  if (boxes_out)
    fclose(boxes_out);
  printf("%s: %d photos (x%d rows): %d %s, %d %s, %d %s", csv_path, photos, copies,
         object_counts[CLASS_RBC], class_names[CLASS_RBC], object_counts[CLASS_WBC], class_names[CLASS_WBC],
         object_counts[CLASS_PLATELET], class_names[CLASS_PLATELET]);
  if (skipped)
    printf(" (%d zero-size boxes skipped)", skipped);
  printf("\n");
  return status;
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <dataset-dir>\n", argv[0]);
    printf("  <dataset-dir> : The extracted BCCD/ directory (containing Annotations/, ImageSets/, JPEGImages/)\n");
    printf("Writes train.csv, validation.csv, test.csv, validation_boxes.txt, and test_boxes.txt\n");
    printf("to the current directory.\n");
    return 1;
  }
  const char *dataset_dir = argv[1];
  // Fixed seed, so rerunning this produces the same augmented train.csv.
  srand(1);
  if (write_split(dataset_dir, "train.txt", TRAIN_COPIES, "train.csv", NULL) < 0 ||
      write_split(dataset_dir, "val.txt", 1, "validation.csv", "validation_boxes.txt") < 0 ||
      write_split(dataset_dir, "test.txt", 1, "test.csv", "test_boxes.txt") < 0)
    return 1;
  return 0;
}
