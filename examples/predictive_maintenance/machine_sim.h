/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MACHINE_SIM_H
#define MACHINE_SIM_H

// Desktop-only vibration simulator, shared by train.c, test.c and
// monitor.c, standing in for an accelerometer bolted to a real machine:
// a 4-pole induction motor on 50Hz mains driving a gearbox (pinion on the
// motor shaft) through 6206 deep-groove ball bearings. It synthesizes raw
// acceleration samples, in g, at VF_SAMPLE_RATE -- exactly what the
// firmware would read from the IIS3DWB after decimation -- so everything
// downstream (vibration_features.[ch], health_monitor.[ch], the model) is
// exercised on the same kind of input it would see on the target.
//
// See train.c's top-of-file comment for why this example simulates its
// data rather than downloading a dataset.

#include "health_monitor.h"

// One physical machine, as built and installed: everything about it that
// differs from its siblings off the same production line but stays fixed
// for its whole life. machine_create() randomizes these within realistic
// ranges, so no two simulated machines look alike -- the reason a
// per-machine baseline (see health_monitor.h) is worth having.
typedef struct {
  float slip;               // nominal induction-motor slip at typical load (shaft runs at 25Hz * (1 - slip))
  float amp_1x;             // residual imbalance: 1x running-speed amplitude, g
  float amp_2x;             // residual misalignment: 2x amplitude, g
  float phase_1x, phase_2x;
  float amp_line;           // 2x line-frequency (100Hz) electromagnetic vibration, g
  float amp_mesh;           // gear-mesh tone amplitude, g
  float resonance_hz[2];    // structural resonances excited by impacts and friction
  float resonance_damping[2];
  float resonance_mix;      // how strongly resonance 2 responds relative to resonance 1
  float resonance_noise;    // healthy broadband RMS level in the resonance region, g
  float noise;              // white sensor/process noise RMS, g
  float gain;               // sensor mounting gain (stud vs. magnet vs. adhesive, location)
  float defect_phase;       // shaft angle at which a damaged gear tooth meshes, in revolutions
} machine_t;

// Builds a new, randomly-varied machine using rand().
void machine_create(machine_t *m);

// Synthesizes VF_SNAPSHOT_SAMPLES acceleration samples (in g) into `out`
// for machine `m` in `condition`, at `severity` from 0 (no fault; ignored
// for CONDITION_HEALTHY) to 1 (on the point of failure). The operating
// point -- load (and so running speed), signal phases, noise -- is
// re-randomized every call, the way two snapshots of a real machine taken
// on different days never match exactly.
void machine_snapshot(const machine_t *m, machine_condition_t condition, float severity, float *out);

// Takes one snapshot of `m` (machine_snapshot()) and runs it through `hm`
// one frame at a time, exactly as firmware would feed the accelerometer's
// frames in, then ends the snapshot with health_monitor_end_snapshot(hm,
// nn, report). Not reentrant (uses one static sample buffer).
void machine_observe(health_monitor_t *hm, nn_t *nn, const machine_t *m, machine_condition_t condition, float severity, health_report_t *report);

// Runs `hm`'s whole commissioning phase against `m` while it's healthy.
void machine_commission(health_monitor_t *hm, const machine_t *m);

// Fault severity on `day` of a simulated run to failure: 0 up to
// `onset_day`, then growing quadratically -- slowly at first, then faster,
// the way wear accelerates once damage starts -- to 1 at `failure_day`.
float fault_severity_on_day(int day, int onset_day, int failure_day);

// Uniform random float in [lo, hi), from rand().
float sim_uniform(float lo, float hi);

#endif /* MACHINE_SIM_H */
