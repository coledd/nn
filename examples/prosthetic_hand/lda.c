/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lda.h"

// Solves A x = b for symmetric positive-definite A (d x d) by Cholesky
// decomposition, overwriting A with its factor. Returns -1 if A isn't
// positive definite.
static int cholesky_solve(double *a, int d, double *b)
{
  for (int j = 0; j < d; j++) {
    double s = a[j * d + j];
    for (int k = 0; k < j; k++)
      s -= a[j * d + k] * a[j * d + k];
    if (s <= 0.0)
      return -1;
    a[j * d + j] = sqrt(s);
    for (int i = j + 1; i < d; i++) {
      double t = a[i * d + j];
      for (int k = 0; k < j; k++)
        t -= a[i * d + k] * a[j * d + k];
      a[i * d + j] = t / a[j * d + j];
    }
  }
  for (int i = 0; i < d; i++) { // L y = b
    for (int k = 0; k < i; k++)
      b[i] -= a[i * d + k] * b[k];
    b[i] /= a[i * d + i];
  }
  for (int i = d - 1; i >= 0; i--) { // L^T x = y
    for (int k = i + 1; k < d; k++)
      b[i] -= a[k * d + i] * b[k];
    b[i] /= a[i * d + i];
  }
  return 0;
}

int lda_train(lda_t *lda, const float *x, const int *labels, int n, int dims, int classes, float shrinkage)
{
  int d = dims;
  double *mean = (double *)calloc((size_t)classes * d, sizeof(double));
  double *cov = (double *)calloc((size_t)d * d, sizeof(double));
  double *factor = (double *)malloc((size_t)d * d * sizeof(double));
  double *rhs = (double *)malloc((size_t)d * sizeof(double));
  int *count = (int *)calloc((size_t)classes, sizeof(int));
  lda->classes = classes;
  lda->dims = d;
  lda->w = (float *)malloc((size_t)classes * d * sizeof(float));
  lda->b = (float *)malloc((size_t)classes * sizeof(float));
  int result = -1;
  if (mean == NULL || cov == NULL || factor == NULL || rhs == NULL || count == NULL || lda->w == NULL || lda->b == NULL)
    goto done;

  for (int r = 0; r < n; r++) {
    count[labels[r]]++;
    for (int i = 0; i < d; i++)
      mean[labels[r] * d + i] += x[(size_t)r * d + i];
  }
  for (int k = 0; k < classes; k++) {
    if (count[k] == 0)
      goto done;
    for (int i = 0; i < d; i++)
      mean[k * d + i] /= count[k];
  }
  // Pooled within-class covariance.
  for (int r = 0; r < n; r++) {
    const float *row = x + (size_t)r * d;
    const double *m = mean + labels[r] * d;
    for (int i = 0; i < d; i++)
      for (int j = 0; j <= i; j++)
        cov[i * d + j] += (row[i] - m[i]) * (row[j] - m[j]);
  }
  double trace = 0.0;
  for (int i = 0; i < d; i++) {
    for (int j = 0; j <= i; j++) {
      cov[i * d + j] /= n - classes > 0 ? n - classes : 1;
      cov[j * d + i] = cov[i * d + j];
    }
    trace += cov[i * d + i];
  }
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++)
      cov[i * d + j] *= 1.0 - shrinkage;
    cov[i * d + i] += shrinkage * trace / d;
  }

  // w_k = cov^-1 mean_k, b_k = -w_k . mean_k / 2 (equal priors: the
  // decoder shouldn't favor an intent just because calibration happened
  // to hold it longer).
  for (int k = 0; k < classes; k++) {
    memcpy(factor, cov, (size_t)d * d * sizeof(double));
    memcpy(rhs, mean + k * d, (size_t)d * sizeof(double));
    if (cholesky_solve(factor, d, rhs) != 0)
      goto done;
    double bias = 0.0;
    for (int i = 0; i < d; i++) {
      lda->w[k * d + i] = (float)rhs[i];
      bias += rhs[i] * mean[k * d + i];
    }
    lda->b[k] = (float)(-0.5 * bias);
  }
  result = 0;
done:
  free(mean);
  free(cov);
  free(factor);
  free(rhs);
  free(count);
  return result;
}

int lda_predict(const lda_t *lda, const float *x)
{
  int best = 0;
  float best_score = -INFINITY;
  for (int k = 0; k < lda->classes; k++) {
    float s = lda->b[k];
    for (int i = 0; i < lda->dims; i++)
      s += lda->w[k * lda->dims + i] * x[i];
    if (s > best_score) {
      best_score = s;
      best = k;
    }
  }
  return best;
}

void lda_free(lda_t *lda)
{
  free(lda->w);
  free(lda->b);
  lda->w = NULL;
  lda->b = NULL;
}
