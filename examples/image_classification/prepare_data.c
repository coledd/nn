/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// Builds this example's training/evaluation CSVs from extracted copies of
// two public cat/dog photo datasets:
//
// - Microsoft's "Kaggle Cats and Dogs" dataset (the Asirra CAPTCHA corpus,
//   photos from Petfinder.com shelters;
//   https://www.microsoft.com/en-us/download/details.aspx?id=54765;
//   Community Data License Agreement - Permissive 2.0): 12,500 cat and
//   12,500 dog photos, labeled by species only, with no published split.
// - The Oxford-IIIT Pet dataset (Parkhi, Vedaldi, Zisserman & Jawahar,
//   2012, "Cats and Dogs", https://www.robots.ox.ac.uk/~vgg/data/pets/; CC
//   BY-SA 4.0, image copyright with the original owners): ~7,400 photos of
//   37 breeds (12 cat, 25 dog), labeled with species and breed, with a
//   published train/test split and a per-pixel "trimap" outlining the
//   animal.
//
// Each image becomes one row: IMG_INPUTS normalized pixels (see
// image_prep.[ch] -- a square crop around the animal, resized and
// normalized exactly the way predict.c and a deployed camera would),
// followed by a CLASS_COUNT-wide one-hot target. For Oxford images the
// crop box is the bounding box of the trimap's animal pixels; Kaggle images
// have no such annotation, so they use the whole frame (image_prep_crop()
// then takes its center square), the same default predict.c uses.
//
//   train.csv              <- Kaggle images numbered 2-9 mod 10, plus Oxford
//                             trainval.txt minus every VALIDATION_EVERY'th
//   validation.csv         <- Kaggle images numbered 0 mod 10, plus every
//                             VALIDATION_EVERY'th Oxford trainval.txt image
//   test.csv               <- Kaggle images numbered 1 mod 10
//   test_oxford.csv        <- Oxford test.txt (the dataset's published test split)
//   test_oxford_labels.txt <- one breed name per test_oxford.csv row, same
//                             order, so test.c can break accuracy down by breed
//
// The Kaggle images are numbered 0-12499 per class with no meaningful order,
// so splitting by number is an arbitrary but fixed and reproducible split.
// Oxford's trainval.txt lists each breed's images together, so taking every
// VALIDATION_EVERY'th one spreads its validation share evenly across breeds.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "image_prep.h"
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
// Trimaps are PNGs, and a few hundred of the datasets' "*.jpg" photos are
// actually PNG, BMP or GIF data under a .jpg name.
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_STDIO_WARNINGS
#include "stb_image.h"

#define MAX_PATH_LEN 4096
#define MAX_NAME_LEN 256
#define KAGGLE_DIR "PetImages"
#define KAGGLE_PER_CLASS 12500
#define OXFORD_IMAGE_DIR "images"
#define OXFORD_ANNOTATION_DIR "annotations"
#define VALIDATION_EVERY 10
// Include Oxford's trainval images in train.csv/validation.csv (not just
// its test split in test_oxford.csv). See this example's README for the
// measured difference.
#ifndef OXFORD_TRAIN
#define OXFORD_TRAIN 1
#endif
// Training rows to generate per class (see write_split()), and how far an
// augmented crop may be shifted/rescaled, as a fraction of the box size.
#define TRAIN_ROWS_PER_CLASS 20000
#define JITTER 0.15f
// Trimap pixel values: 1 = animal, 2 = background, 3 = boundary.
#define TRIMAP_BACKGROUND 2

typedef struct {
  char image[MAX_PATH_LEN];   // image path, relative to the dataset directory
  char trimap[MAX_PATH_LEN];  // trimap path (Oxford only), or "" to use the whole frame
  char breed[MAX_NAME_LEN];   // e.g. "Abyssinian" (Oxford only), or ""
  int label;                  // CLASS_CAT or CLASS_DOG
} entry_t;

typedef struct {
  entry_t *items;
  int count;
  int capacity;
} entry_list_t;

static entry_t *list_add(entry_list_t *list)
{
  if (list->count == list->capacity) {
    int capacity = list->capacity ? list->capacity * 2 : 1024;
    entry_t *grown = realloc(list->items, (size_t)capacity * sizeof(entry_t));
    if (!grown)
      return NULL;
    list->items = grown;
    list->capacity = capacity;
  }
  entry_t *e = &list->items[list->count++];
  memset(e, 0, sizeof(*e));
  return e;
}

// Adds the Kaggle images whose number n satisfies (n % 10) in [lo, hi].
static int add_kaggle(entry_list_t *list, int lo, int hi)
{
  static const char *class_dirs[CLASS_COUNT] = { "Cat", "Dog" };
  for (int c = 0; c < CLASS_COUNT; c++) {
    for (int n = 0; n < KAGGLE_PER_CLASS; n++) {
      if (n % 10 < lo || n % 10 > hi)
        continue;
      entry_t *e = list_add(list);
      if (!e)
        return -1;
      snprintf(e->image, sizeof(e->image), "%s/%s/%d.jpg", KAGGLE_DIR, class_dirs[c], n);
      e->label = c;
    }
  }
  return 0;
}

// Adds entries from one of Oxford's split lists ("<name> <class-id>
// <species> <breed-id>" per line, species 1 = cat, 2 = dog; '#' lines are
// comments), keeping the i'th listed image when keep(i) is true.
static int add_oxford(entry_list_t *list, const char *dataset_dir, const char *list_name, int (*keep)(int))
{
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "%s/%s/%s", dataset_dir, OXFORD_ANNOTATION_DIR, list_name);
  FILE *file = fopen(path, "r");
  if (!file) {
    fprintf(stderr, "Error: Could not read %s\n", path);
    return -1;
  }
  char line[1024];
  int index = 0;
  while (fgets(line, sizeof(line), file)) {
    char name[MAX_NAME_LEN];
    int class_id, species, breed_id;
    if (line[0] == '#' || sscanf(line, "%255s %d %d %d", name, &class_id, &species, &breed_id) != 4)
      continue;
    if (!keep(index++))
      continue;
    entry_t *e = list_add(list);
    if (!e) {
      fclose(file);
      return -1;
    }
    snprintf(e->image, sizeof(e->image), "%s/%s.jpg", OXFORD_IMAGE_DIR, name);
    snprintf(e->trimap, sizeof(e->trimap), "%s/trimaps/%s.png", OXFORD_ANNOTATION_DIR, name);
    snprintf(e->breed, sizeof(e->breed), "%s", name);
    char *underscore = strrchr(e->breed, '_');
    if (underscore)
      *underscore = '\0';
    e->label = (species == 1) ? CLASS_CAT : CLASS_DOG;
  }
  fclose(file);
  return 0;
}

// Bounding box of the animal (every non-background trimap pixel), scaled to
// a `width` x `height` image. Uses the whole image if there's no trimap, or
// it's unreadable or empty.
static void animal_box(const char *dataset_dir, const entry_t *e, int width, int height,
                       float *bx, float *by, float *bw, float *bh)
{
  *bx = 0.0f;
  *by = 0.0f;
  *bw = (float)width;
  *bh = (float)height;
  if (!e->trimap[0])
    return;
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "%s/%s", dataset_dir, e->trimap);
  int tw, th, tc;
  uint8_t *trimap = stbi_load(path, &tw, &th, &tc, 1);
  if (!trimap)
    return;
  int x0 = tw, y0 = th, x1 = -1, y1 = -1;
  for (int y = 0; y < th; y++) {
    for (int x = 0; x < tw; x++) {
      if (trimap[y * tw + x] != TRIMAP_BACKGROUND) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  stbi_image_free(trimap);
  if (x1 < x0 || y1 < y0)
    return;
  float sx = (float)width / (float)tw, sy = (float)height / (float)th;
  *bx = x0 * sx;
  *by = y0 * sy;
  *bw = (x1 - x0 + 1) * sx;
  *bh = (y1 - y0 + 1) * sy;
}

static void write_row(FILE *out, const float *pixels, int label)
{
  for (int i = 0; i < IMG_INPUTS; i++)
    fprintf(out, "%.4f,", pixels[i]);
  for (int c = 0; c < CLASS_COUNT; c++)
    fprintf(out, c + 1 < CLASS_COUNT ? "%d," : "%d\n", c == label);
}

// Uniform random float in [lo, hi).
static float rand_range(float lo, float hi)
{
  return lo + (hi - lo) * ((float)rand() / ((float)RAND_MAX + 1.0f));
}

// Writes `copies[label]` rows for one image to `out` (and its breed to
// `breeds`, once per row, if non-NULL). The first row is the plain crop;
// any further copies are augmented. Returns 1 on success, 0 if the image
// couldn't be decoded.
static int write_image(const char *dataset_dir, const entry_t *e, const int *copies, FILE *out, FILE *breeds)
{
  static float pixels[IMG_INPUTS];
  char path[MAX_PATH_LEN];
  snprintf(path, sizeof(path), "%s/%s", dataset_dir, e->image);
  int w, h, components;
  uint8_t *img = stbi_load(path, &w, &h, &components, 0);
  if (!img)
    return 0;
  float bx, by, bw, bh;
  animal_box(dataset_dir, e, w, h, &bx, &by, &bw, &bh);
  for (int k = 0; k < copies[e->label]; k++) {
    // Copy 0 is the box exactly as derived; copy 1 its mirror image; every
    // later copy a randomly mirrored, shifted (up to +/-JITTER of the box
    // size) and rescaled (+/-JITTER) crop -- a different framing of the
    // same animal, the way a real detector's box around it would never
    // land in exactly the same place twice.
    float jx = 0.0f, jy = 0.0f, js = 1.0f;
    int flip = (k == 1);
    if (k >= 2) {
      jx = rand_range(-JITTER, JITTER) * bw;
      jy = rand_range(-JITTER, JITTER) * bh;
      js = rand_range(1.0f - JITTER, 1.0f + JITTER);
      flip = rand() & 1;
    }
    float cx = bx + bw * 0.5f + jx, cy = by + bh * 0.5f + jy;
    image_prep_crop(img, w, h, components, cx - bw * js * 0.5f, cy - bh * js * 0.5f,
                    bw * js, bh * js, flip, pixels);
    write_row(out, pixels, e->label);
    if (breeds)
      fprintf(breeds, "%s\n", e->breed);
  }
  stbi_image_free(img);
  return 1;
}

// Writes every entry in `list` to `csv_path` (and, if `breeds_path` is
// non-NULL, one breed name per row to it). Returns 0, or -1 on a file
// error.
static int write_split(const char *dataset_dir, const entry_list_t *list, int augment,
                       const char *csv_path, const char *breeds_path)
{
  int copies[CLASS_COUNT], counts[CLASS_COUNT] = { 0 }, skipped = 0;
  for (int c = 0; c < CLASS_COUNT; c++)
    copies[c] = 1;
  // Only the training split is augmented (the evaluation splits stay
  // exactly what a camera would see): each class gets however many copies
  // per image bring it to ~TRAIN_ROWS_PER_CLASS rows -- at least 2, the
  // image and its mirror -- so if one class has fewer images, it isn't also
  // the one the network sees least.
  if (augment) {
    int available[CLASS_COUNT] = { 0 };
    for (int i = 0; i < list->count; i++)
      available[list->items[i].label]++;
    for (int c = 0; c < CLASS_COUNT; c++) {
      copies[c] = available[c] > 0 ? (TRAIN_ROWS_PER_CLASS + available[c] - 1) / available[c] : 1;
      if (copies[c] < 2)
        copies[c] = 2;
    }
  }
  FILE *out = fopen(csv_path, "w");
  FILE *breeds = breeds_path ? fopen(breeds_path, "w") : NULL;
  if (!out || (breeds_path && !breeds)) {
    fprintf(stderr, "Error: Could not create %s\n", !out ? csv_path : breeds_path);
    if (out)
      fclose(out);
    return -1;
  }
  for (int i = 0; i < list->count; i++) {
    if (write_image(dataset_dir, &list->items[i], copies, out, breeds))
      counts[list->items[i].label]++;
    else
      skipped++;
  }
  fclose(out);
  if (breeds)
    fclose(breeds);
  printf("%s: %d %s images (x%d rows), %d %s images (x%d rows)", csv_path,
         counts[CLASS_CAT], class_names[CLASS_CAT], copies[CLASS_CAT],
         counts[CLASS_DOG], class_names[CLASS_DOG], copies[CLASS_DOG]);
  // A handful of the Kaggle files are empty or in a format stb_image
  // doesn't decode (e.g. Photoshop).
  if (skipped)
    printf(", %d unreadable skipped", skipped);
  printf("\n");
  return 0;
}

static int is_train(int index) { return index % VALIDATION_EVERY != 0; }
static int is_validation(int index) { return index % VALIDATION_EVERY == 0; }
static int is_any(int index) { (void)index; return 1; }

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <dataset-dir>\n", argv[0]);
    printf("  <dataset-dir> : Directory containing the extracted %s/ (Kaggle) and %s/ + %s/ (Oxford)\n",
           KAGGLE_DIR, OXFORD_IMAGE_DIR, OXFORD_ANNOTATION_DIR);
    printf("Writes train.csv, validation.csv, test.csv, test_oxford.csv, and test_oxford_labels.txt\n");
    printf("to the current directory.\n");
    return 1;
  }
  const char *dataset_dir = argv[1];
  // Fixed seed, so rerunning this produces the same augmented train.csv.
  srand(1);
  entry_list_t train = { 0 }, validation = { 0 }, test = { 0 }, test_oxford = { 0 };
  int status = 0;
  if (add_kaggle(&train, 2, 9) < 0 || add_kaggle(&validation, 0, 0) < 0 || add_kaggle(&test, 1, 1) < 0 ||
      (OXFORD_TRAIN && add_oxford(&train, dataset_dir, "trainval.txt", is_train) < 0) ||
      (OXFORD_TRAIN && add_oxford(&validation, dataset_dir, "trainval.txt", is_validation) < 0) ||
      add_oxford(&test_oxford, dataset_dir, "test.txt", is_any) < 0)
    status = 1;
  if (!status &&
      (write_split(dataset_dir, &train, 1, "train.csv", NULL) < 0 ||
       write_split(dataset_dir, &validation, 0, "validation.csv", NULL) < 0 ||
       write_split(dataset_dir, &test, 0, "test.csv", NULL) < 0 ||
       write_split(dataset_dir, &test_oxford, 0, "test_oxford.csv", "test_oxford_labels.txt") < 0))
    status = 1;
  free(train.items);
  free(validation.items);
  free(test.items);
  free(test_oxford.items);
  return status;
}
