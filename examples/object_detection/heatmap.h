/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HEATMAP_H
#define HEATMAP_H

#include "image_prep.h"

// Turns the network's output -- CLASS_COUNT heatmaps of GRID_W x GRID_H
// cells, each cell's value the network's confidence that an object of that
// class is centered there -- into a list of detected objects. Shared by
// test.c and predict.c, and, like image_prep.[ch], meant to ship into
// firmware: this is the step between nn_predict() and "14 red cells, 1
// white cell, 2 platelets". No heap allocation.

typedef struct {
  int cls;      // cell_class_t
  float x, y;   // object center, as a fraction of the frame's width/height (0..1)
  float score;  // the heatmap value at the peak cell
} detection_t;

// Default peak threshold. See this example's README for how it was chosen.
#define HEATMAP_THRESHOLD 0.4f

// Finds every cell whose value is at least `threshold` and is a local
// maximum among its 8 neighbors in the same class's heatmap, and reports
// each as one detection, centered at the value-weighted average position
// of that 3x3 neighborhood (so an object straddling two cells lands between
// them, not snapped to one). Writes up to `max_detections` to `out` and
// returns how many it found in total (which may be more than it wrote).
int heatmap_decode(const float *heatmap, float threshold, detection_t *out, int max_detections);

#endif /* HEATMAP_H */
