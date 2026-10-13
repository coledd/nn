/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

// Turns raw PCM audio into the log-mel feature frames a sequence model
// (LAYER_TYPE_RNN/GRU/LSTM, see nn.h) can consume one timestep at a time --
// the piece nn.[ch] alone doesn't provide. Unlike everything else added
// alongside this file (data_prep.[ch], wav_reader.[ch], and every
// examples/*/ file), this one is meant to ship into firmware next to
// nn.c/nn.h: the exact same audio_features_frame()/
// audio_features_extract_sequence() calls that turn a training corpus into
// feature rows also have to run, unchanged, against a deployed target's
// live mic buffer -- otherwise the model sees different features at
// inference than it was trained on. It has no file-I/O dependency of its
// own (see wav_reader.[ch] for that, which is desktop-only) and does no
// heap allocation outside of audio_features_init()/audio_features_free(),
// so a single audio_features_t can be created once at startup and reused,
// allocation-free, for every subsequent frame.
//
// Not part of libnn.a (it isn't neural-net code) -- see the top-level
// Makefile's audio_features.o rule.

typedef struct {
  int sample_rate;   // Hz
  int frame_len;     // samples per frame (FFT size); must be a power of 2
  int num_fft_bins;  // frame_len/2 + 1 -- the non-redundant half of a real-input FFT
  int num_mel_bins;
  float *window;       // frame_len Hann window coefficients, precomputed once
  float *mel_weights;  // num_mel_bins * num_fft_bins triangular filterbank weights, row-major, precomputed once
  // Scratch buffers, sized once at init and reused by every
  // audio_features_frame() call so it never allocates -- important for an
  // embedded caller running this in a real-time mic-input path.
  float *scratch_real; // frame_len
  float *scratch_imag; // frame_len
  float *power;        // num_fft_bins
} audio_features_t;

// Precomputes the window and mel filterbank for the given `sample_rate`,
// `frame_len` (must be a power of 2 -- audio_features_frame() uses an
// in-place radix-2 FFT, see fft.h), and `num_mel_bins`. Returns NULL on invalid
// arguments or allocation failure.
audio_features_t *audio_features_init(int sample_rate, int frame_len, int num_mel_bins);
void audio_features_free(audio_features_t *af);

// Computes log-mel filterbank energies for one windowed frame of
// `af->frame_len` raw PCM samples (nominally in [-1, 1], as wav_reader.h
// produces). Writes `af->num_mel_bins` values to `out`. Allocation-free --
// safe to call from a real-time mic-input interrupt/callback path once
// `af` has been created.
void audio_features_frame(audio_features_t *af, const float *frame, float *out);

// Number of frames audio_features_extract_sequence() produces for
// `num_samples` samples at `af->frame_len` and hop `hop`: frames covering
// fewer than a full frame_len of trailing samples are not produced (the
// same way a live streaming caller wouldn't emit a frame until frame_len
// fresh samples have actually arrived). Returns 0 if num_samples < frame_len.
int audio_features_num_frames(const audio_features_t *af, int num_samples, int hop);

// Slides a frame_len-sample window across `pcm` (`num_samples` samples) in
// `hop`-sample steps, writing one row of `af->num_mel_bins` log-mel
// energies per frame to `out` (row-major: frame t's energies start at
// `out[t * af->num_mel_bins]`), then normalizes the whole sequence to zero
// mean and unit variance in place -- see this function's definition in
// audio_features.c for why that's done here (self-contained, per clip)
// rather than left to the caller. Because normalization needs the whole
// sequence's statistics, this is a batch operation over a fixed window,
// not an incremental/streaming one -- audio_features_frame() remains the
// building block for a caller that wants raw, unnormalized per-frame
// energies one at a time. `out` must have room for
// audio_features_num_frames(af, num_samples, hop) * af->num_mel_bins
// floats.
void audio_features_extract_sequence(audio_features_t *af, const float *pcm, int num_samples, int hop, float *out);

#endif /* AUDIO_FEATURES_H */
