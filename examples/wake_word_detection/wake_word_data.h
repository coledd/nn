/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef WAKE_WORD_DATA_H
#define WAKE_WORD_DATA_H

#include <stdint.h>
#include "nn.h"

// Shared feature-extraction shape for this example's whole pipeline
// (prepare_data.c, train_gru.c, train_cnn.c, test.c, predict.c). Unlike gesture_data.h/fall_data.h in the other RNN examples,
// there's no on-the-fly synthetic generator here -- training data comes
// from the real Speech Commands dataset (see prepare_data.c) -- so this
// header holds only the dimensions every stage has to agree on: every clip
// is decoded, padded/truncated to CLIP_SAMPLES, and split into SEQUENCE_LEN
// frames of NUM_MEL_BINS log-mel energies each via
// audio_features_extract_sequence() (see ../../audio_features.h) -- the
// exact same call a deployed firmware would make against its own live mic
// buffer.
#define SAMPLE_RATE  16000  // Hz, matches Speech Commands' native sample rate
#define FRAME_LEN    512    // samples/frame (32ms @ 16kHz); must be a power of 2 for audio_features_frame()'s FFT
#define HOP_LEN      320    // samples between frame starts (20ms @ 16kHz)
#define NUM_MEL_BINS 20
#define CLIP_SAMPLES SAMPLE_RATE // every clip is padded/truncated to exactly 1.0s

// Frame count for one CLIP_SAMPLES-long clip at FRAME_LEN/HOP_LEN -- must
// match audio_features_num_frames(af, CLIP_SAMPLES, HOP_LEN) exactly, since
// it sizes every fixed-width array/CSV row in this example (49 frames for
// the defaults above).
#define SEQUENCE_LEN ((CLIP_SAMPLES - FRAME_LEN) / HOP_LEN + 1)

// Runs one clip's SEQUENCE_LEN * NUM_MEL_BINS features (frame-major, as
// prepare_data.c writes them) through either kind of model this example
// trains, and returns the wake-word probability, or a negative value if
// the model's shape is neither:
//  - train_gru.c's: one NUM_MEL_BINS-wide frame per call, a single sigmoid
//    output read after the last frame.
//  - train_cnn.c's: the whole clip in one call, a two-way softmax output
//    (other, wake word).
static inline float wake_word_probability(nn_t *model, float *features)
{
  uint32_t inputs = model->width[0];
  uint32_t outputs = model->width[model->depth - 1];
  if (inputs == NUM_MEL_BINS && outputs == 1) {
    nn_reset_state(model);
    float *prediction = NULL;
    for (int t = 0; t < SEQUENCE_LEN; t++)
      prediction = nn_predict(model, features + t * NUM_MEL_BINS);
    return prediction[0];
  }
  if (inputs == SEQUENCE_LEN * NUM_MEL_BINS && outputs == 2)
    return nn_predict(model, features)[1];
  return -1.0f;
}

// The decision rule a listener applies to those probabilities as its
// one-second window slides along the microphone stream (stream.c measures
// exactly this). Feed it one probability per window, in order; it returns
// 1 when the device should wake. It wakes only after `consecutive` windows
// in a row reach `threshold` -- a spoken word keeps scoring high as the
// window slides across it, a chance match often doesn't -- and then ignores
// the next `hold_off` windows, so one utterance wakes it once. Combining
// two models is the caller's job: pass the smaller of their probabilities,
// so that both have to agree (the second model only needs to run once the
// first scores at or above the threshold).
typedef struct {
  float threshold;
  int consecutive; // windows in a row at or above threshold before waking (1 = wake on one)
  int hold_off;    // windows ignored after waking
  int run;         // internal: windows in a row at or above threshold so far
  int ignoring;    // internal: hold-off windows still to ignore
} wake_word_detector_t;

static inline void wake_word_detector_init(wake_word_detector_t *d, float threshold, int consecutive, int hold_off)
{
  d->threshold = threshold;
  d->consecutive = consecutive;
  d->hold_off = hold_off;
  d->run = 0;
  d->ignoring = 0;
}

static inline int wake_word_detector_step(wake_word_detector_t *d, float probability)
{
  if (d->ignoring > 0) {
    d->ignoring--;
    return 0;
  }
  d->run = probability >= d->threshold ? d->run + 1 : 0;
  if (d->run < d->consecutive)
    return 0;
  d->run = 0;
  d->ignoring = d->hold_off;
  return 1;
}

#endif /* WAKE_WORD_DATA_H */
