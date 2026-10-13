/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NINAPRO_H
#define NINAPRO_H

// Desktop-only: loads one Ninapro subject's exercise-1 recording (as
// fetched by fetch_ninapro.py) and relabels it with this example's intents.
//
// Ninapro (Atzori et al., 2014, "Electromyography data for non-invasive
// naturally-controlled robotic hand prostheses", Scientific Data 1:140053)
// recorded 12 Delsys Trigno electrodes at 2kHz: eight spaced evenly around
// the forearm just below the elbow, two over the finger flexor and
// extensor muscles, and two on the biceps and triceps. Only the ring of
// eight is used by default (the first EF_CHANNELS columns of `emg`): it's
// the arrangement a transradial prosthesis socket can hold, and it fits one
// 8-channel analog front end. Building with `make CHANNELS=10` adds the
// two targeted forearm electrodes, and CHANNELS=12 the upper-arm pair too.
//
// Exercise 1 ("B" in the paper) is 17 basic finger and wrist movements,
// each held for 5 seconds and repeated 6 times with 3 seconds of rest
// between. Six of them are the movements a powered hand and wrist are
// driven with; the rest periods give the seventh class.

#include <stddef.h>
#include "emg_features.h"
#include "intent_decoder.h"

#define NINAPRO_REPETITIONS 6
#define NINAPRO_DB2_SUBJECTS 40 // intact
#define NINAPRO_DB3_SUBJECTS 11 // transradial amputees
// Of the rest windows ninapro_window_intent() accepts, prepare_data.c and
// replay.c's calibration keep one in this many: the refined labels leave
// rest about five times as long as each movement, and this makes it about
// as common.
#define NINAPRO_REST_THINNING 5

typedef struct {
  int samples;
  float *emg;              // samples x EF_CHANNELS, row-major, in volts
  signed char *intent;     // per sample: an intent_t, or -1 for a movement this example doesn't use
  signed char *repetition; // per sample: 1..NINAPRO_REPETITIONS (rest periods count toward the repetition that follows them)
  signed char *next_movement; // per sample: the intent of the first non-rest sample at or after it (-1 if not one of ours)
  int *run;                // per sample: how many samples up to and including it share its intent and repetition
} ninapro_recording_t;

// The Ninapro exercise-1 movement number behind each intent (0 = rest).
extern const int ninapro_movement[INTENT_COUNT];

// Path of a subject's recording under `dataset_dir`: db is 2 (intact
// subjects) or 3 (amputees).
void ninapro_path(char *buf, size_t len, const char *dataset_dir, int db, int subject);

// Returns 0 on success, -1 (with a message on stderr) on failure.
int ninapro_load(const char *path, ninapro_recording_t *rec);

void ninapro_free(ninapro_recording_t *rec);

// The intent a window of EF_WINDOW samples ending at sample `end` is
// labeled with, or -1 if it shouldn't be used for training or window-level
// testing: it straddles a change of intent or repetition, it belongs to a
// movement this example doesn't use, or it's rest before one of those.
// (Continuous replay, replay.c, uses every window instead.)
int ninapro_window_intent(const ninapro_recording_t *rec, int end);

#endif /* NINAPRO_H */
