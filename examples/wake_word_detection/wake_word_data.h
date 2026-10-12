/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef WAKE_WORD_DATA_H
#define WAKE_WORD_DATA_H

// Shared feature-extraction shape for this example's whole pipeline
// (prepare_data.c, train.c, test.c,
// predict.c). Unlike gesture_data.h/fall_data.h in the other RNN examples,
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

#endif /* WAKE_WORD_DATA_H */
