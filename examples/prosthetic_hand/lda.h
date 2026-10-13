/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LDA_H
#define LDA_H

// Linear discriminant analysis: the classifier behind the Hudgins
// features in most of the myoelectric-control literature and in the
// commercial pattern-recognition controllers, here as the baseline the
// network is measured against. It models each intent as a Gaussian with
// its own mean and a covariance shared by all of them, which makes the
// decision rule linear: score_k(x) = w_k . x + b_k, highest score wins.
// Training is closed-form (no epochs), so it fits from a minute of
// calibration data in milliseconds -- the reason it's the incumbent.

typedef struct {
  int classes, dims;
  float *w; // classes x dims
  float *b; // classes
} lda_t;

// Fits from `n` rows of `dims` features with labels in [0, classes).
// The pooled covariance is shrunk toward a scaled identity by `shrinkage`
// (0..1) so it stays invertible with little data. Returns 0, or -1 if a
// class has no rows or the covariance can't be inverted.
int lda_train(lda_t *lda, const float *x, const int *labels, int n, int dims, int classes, float shrinkage);

int lda_predict(const lda_t *lda, const float *x);

void lda_free(lda_t *lda);

#endif /* LDA_H */
