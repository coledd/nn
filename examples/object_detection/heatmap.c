/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

// See heatmap.h for what this is and why it's shared.

#include "heatmap.h"

int heatmap_decode(const float *heatmap, float threshold, detection_t *out, int max_detections)
{
  int found = 0;
  for (int c = 0; c < CLASS_COUNT; c++) {
    const float *map = heatmap + c * GRID_CELLS;
    for (int gy = 0; gy < GRID_H; gy++) {
      for (int gx = 0; gx < GRID_W; gx++) {
        const float v = map[gy * GRID_W + gx];
        if (v < threshold)
          continue;
        // Local maximum test. Ties go to the first cell in scan order (a
        // neighbor already passed counts as higher when equal), so two
        // adjacent cells with exactly the same value still yield one
        // detection, not two.
        int peak = 1;
        float sum = 0.0f, sx = 0.0f, sy = 0.0f;
        for (int dy = -1; dy <= 1 && peak; dy++) {
          for (int dx = -1; dx <= 1; dx++) {
            int nx = gx + dx, ny = gy + dy;
            if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H)
              continue;
            float n = map[ny * GRID_W + nx];
            int before = (dy < 0) || (dy == 0 && dx < 0);
            if ((dx || dy) && (n > v || (before && n == v))) {
              peak = 0;
              break;
            }
            sum += n;
            sx += n * (nx + 0.5f);
            sy += n * (ny + 0.5f);
          }
        }
        if (!peak)
          continue;
        if (found < max_detections) {
          out[found].cls = c;
          out[found].x = sx / sum / (float)GRID_W;
          out[found].y = sy / sum / (float)GRID_H;
          out[found].score = v;
        }
        found++;
      }
    }
  }
  return found;
}
