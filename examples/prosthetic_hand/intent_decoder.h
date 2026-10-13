/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef INTENT_DECODER_H
#define INTENT_DECODER_H

// The firmware-side application around the model: everything between "a
// sample arrived from the electrodes" and "drive the hand this way, this
// fast", built on emg_features.[ch] and nn.[ch] and nothing else, so it
// ships to the target unchanged. train.c, test.c and replay.c all drive
// this same code on the desktop.
//
// Lifecycle on the target:
//
//   1. Calibration, each time the prosthesis is put on. The wearer is
//      prompted through each movement in turn (a few seconds each, plus
//      rest), and the feature vectors are kept with the prompted movement
//      as their label. Electrode contact, skin, sweat and the socket's exact
//      position change from one donning to the next, and from one person to
//      the next far more, so the decoder never sees raw features: every
//      feature is standardized by the mean and spread it had during this
//      wearer's latest calibration (intent_decoder_calibrate()).
//      Optionally the model's weights are then fine-tuned on the same
//      calibration data, on the device (intent_decoder_fine_tune()) -- test.c
//      measures what each step is worth.
//
//   2. Control. Every EF_HOP samples (50ms) the model classifies the last
//      200ms. A classification the model is less than ID_REJECT_BELOW sure
//      of is turned into rest -- for a prosthesis, doing nothing is a much
//      smaller mistake than doing the wrong thing -- and the result is
//      majority-voted over the last ID_VOTES decisions, so one stray
//      window doesn't twitch the hand. The movement is paired with a
//      proportional speed from how hard the wearer is contracting relative
//      to calibration. replay.c measures what rejection and voting each do.

#include <stdbool.h>
#include "emg_features.h"
#include "nn.h"

// The model's output classes: one per direction of the three degrees of
// freedom a powered hand and wrist rotator have, plus rest.
typedef enum {
  INTENT_REST = 0,
  INTENT_HAND_OPEN,
  INTENT_HAND_CLOSE,
  INTENT_WRIST_PRONATION,  // rotate palm down
  INTENT_WRIST_SUPINATION, // rotate palm up
  INTENT_WRIST_FLEXION,
  INTENT_WRIST_EXTENSION,
  INTENT_COUNT
} intent_t;

extern const char *intent_names[INTENT_COUNT];

#define ID_VOTES 5 // 250ms of decisions
// The confidence (softmax probability) below which a classification is
// treated as rest. A per-wearer tuning knob in practice: higher means
// fewer unintended movements and a slower, more hesitant hand (see
// replay.c's sweep).
#define ID_REJECT_BELOW 0.8f
// intent_decoder_fine_tune()'s schedule, chosen on validation subjects
// (see train.c): short and gentle, so a minute of calibration data adapts
// the shared model to the wearer without overwriting what it learned from
// everyone else.
#define ID_FINE_TUNE_EPOCHS 10
#define ID_FINE_TUNE_RATE 0.0005f

// What intent_decoder_calibrate() learns about the wearer; small enough to
// keep in flash or backup RAM so an unchanged fit needn't be recalibrated
// after a reset.
typedef struct {
  float mean[EF_FEATURES];
  float inv_std[EF_FEATURES];
  float rest_level;   // mean log MAV across channels while at rest
  float active_level; // the same, averaged over the calibration movements
} id_calibration_t;

// A majority vote over the last ID_VOTES decisions, kept separate so
// replay.c can vote over alternative decision streams too.
typedef struct {
  int votes[ID_VOTES];
  int count, head;
} id_vote_t;

typedef struct {
  emg_features_t ef;
  id_calibration_t cal;
  float features[EF_FEATURES];    // the latest window's, raw
  float model_input[EF_FEATURES]; // the same, standardized
  id_vote_t vote;
} intent_decoder_t;

typedef struct {
  intent_t raw;               // this window's classification
  intent_t intent;            // after rejection and voting -- what drives the motors
  float confidence;           // softmax probability of `raw`
  float speed;                // 0 at rest-level effort to 1 at calibration-level effort (clamped)
  const float *features;      // EF_FEATURES raw features of this window
  const float *probabilities; // all INTENT_COUNT probabilities, owned by the model
} id_report_t;

void intent_decoder_init(intent_decoder_t *id);

// Computes the wearer's calibration from `n` raw feature vectors
// (n x EF_FEATURES, as emg_features produces them) and their prompted
// intents. Calibration must include rest and at least one movement.
void intent_decoder_calibrate(intent_decoder_t *id, const float *features, const int *labels, int n);

// Trains `nn` on the same calibration data, standardized with the
// calibration already computed: `epochs` passes in a shuffled order, with
// Adam at learning rate `rate` (fresh optimizer state each call). With
// ID_FINE_TUNE_EPOCHS/ID_FINE_TUNE_RATE this fine-tunes the shared model
// train.c produces; test.c also uses it to train a freshly initialized
// model from calibration data alone, for comparison. The model must be
// writable (in RAM -- not loaded with nn_load_model_inplace()), and Adam
// needs two more floats per weight while it runs (freed again on return,
// by switching the model back to SGD). `scratch` is
// EF_FEATURES + INTENT_COUNT floats of working space.
void intent_decoder_fine_tune(const intent_decoder_t *id, nn_t *nn, const float *features, const int *labels, int n,
                              int epochs, float rate, float *scratch);

// Adds `decision` to the vote and returns the winner: the most common
// intent among the last ID_VOTES decisions, ties going to the most recent.
intent_t id_vote_add(id_vote_t *vote, intent_t decision);

// Standardizes one raw feature vector with the current calibration.
void intent_decoder_standardize(const intent_decoder_t *id, const float *features, float *out);

// Adds one sample from every electrode (EF_CHANNELS values). Returns true,
// having filled `report`, when a new decision is ready (every EF_HOP samples).
bool intent_decoder_add_sample(intent_decoder_t *id, nn_t *nn, const float *sample, id_report_t *report);

#endif /* INTENT_DECODER_H */
