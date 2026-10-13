/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "machine_sim.h"

// Local, not M_PI -- see ../../audio_features.c's identical constant.
static const float PI = 3.14159265358979323846f;

#define SYNC_SPEED_HZ 25.0f // 4-pole motor on 50Hz mains: 1500rpm synchronous
#define LINE_HZ 50.0f
#define GEAR_TEETH 23       // pinion tooth count: gear mesh at 23x running speed (~565Hz)
// 6206 ball bearing defect frequencies, as multiples of shaft speed (from
// its geometry: 9 balls, 9.5mm diameter, 46mm pitch diameter).
#define BPFO_ORDER 3.58f    // ball pass frequency, outer race
#define BPFI_ORDER 5.42f    // ball pass frequency, inner race
#define IMPULSE_LEN 160     // samples of resonance ring-down kept per impact (~24ms)

float sim_uniform(float lo, float hi)
{
  return lo + (hi - lo) * ((float)rand() / ((float)RAND_MAX + 1.0f));
}

// Approximately standard-normal noise: the sum of four uniforms, rescaled
// to unit variance. Plenty Gaussian for sensor noise, and far cheaper than
// Box-Muller for the ~100K draws a snapshot needs.
static float gauss(void)
{
  float sum = 0.0f;
  for (int i = 0; i < 4; i++)
    sum += (float)rand() / (float)RAND_MAX;
  return (sum - 2.0f) * 1.7320508f;
}

void machine_create(machine_t *m)
{
  m->slip = sim_uniform(0.015f, 0.035f);
  m->amp_1x = sim_uniform(0.010f, 0.025f);
  m->amp_2x = sim_uniform(0.002f, 0.007f);
  m->phase_1x = sim_uniform(0.0f, 2.0f * PI);
  m->phase_2x = sim_uniform(0.0f, 2.0f * PI);
  m->amp_line = sim_uniform(0.002f, 0.008f);
  m->amp_mesh = sim_uniform(0.015f, 0.050f);
  m->resonance_hz[0] = sim_uniform(1500.0f, 2300.0f);
  m->resonance_hz[1] = sim_uniform(2400.0f, 3100.0f);
  m->resonance_damping[0] = sim_uniform(0.03f, 0.06f);
  m->resonance_damping[1] = sim_uniform(0.03f, 0.06f);
  m->resonance_mix = sim_uniform(0.3f, 1.0f);
  m->resonance_noise = sim_uniform(0.002f, 0.005f);
  m->noise = sim_uniform(0.002f, 0.004f);
  m->gain = sim_uniform(0.7f, 1.4f);
  m->defect_phase = sim_uniform(0.0f, 1.0f);
}

// A sum of sinusoids at multiples of the running speed.
typedef struct {
  float hz, amp, phase;
} tone_t;

#define MAX_TONES 40

static void add_tone(tone_t *tones, int *count, float hz, float amp, float phase)
{
  if (*count < MAX_TONES && amp > 0.0f && hz < VF_SAMPLE_RATE / 2.0f) {
    tones[*count].hz = hz;
    tones[*count].amp = amp;
    tones[*count].phase = phase;
    (*count)++;
  }
}

// One machine resonance, as a two-pole resonator driven by white noise:
// y[n] = x[n] + a1*y[n-1] - a2*y[n-2]. `norm` scales unit-variance input to
// unit-variance output (the resonator's closed-form noise gain), so the
// caller can ask for an RMS level directly.
typedef struct {
  float a1, a2, norm, y1, y2;
} resonator_t;

static void resonator_init(resonator_t *r, float hz, float damping)
{
  float omega = 2.0f * PI * hz / VF_SAMPLE_RATE;
  float radius = expf(-damping * omega);
  float c = cosf(omega * sqrtf(1.0f - damping * damping));
  r->a1 = 2.0f * radius * c;
  r->a2 = radius * radius;
  float rr = radius * radius;
  float variance = (1.0f + rr) / ((1.0f - rr) * ((1.0f + rr) * (1.0f + rr) - 4.0f * rr * c * c));
  r->norm = 1.0f / sqrtf(variance);
  r->y1 = r->y2 = 0.0f;
}

static float resonator_step(resonator_t *r, float x)
{
  float y = x + r->a1 * r->y1 - r->a2 * r->y2;
  r->y2 = r->y1;
  r->y1 = y;
  return y * r->norm;
}

// The response of the machine's structure to one sharp impact (a ball
// hitting a spall, a broken tooth meshing): a decaying ring at each
// resonance, peak-normalized to 1.
static void build_impulse(const machine_t *m, float *impulse)
{
  float peak = 0.0f;
  for (int i = 0; i < IMPULSE_LEN; i++) {
    float t = (float)i / VF_SAMPLE_RATE;
    float v = 0.0f;
    for (int r = 0; r < 2; r++) {
      float omega = 2.0f * PI * m->resonance_hz[r];
      float mix = r == 0 ? 1.0f : m->resonance_mix;
      v += mix * expf(-m->resonance_damping[r] * omega * t) * sinf(omega * t);
    }
    impulse[i] = v;
    if (fabsf(v) > peak)
      peak = fabsf(v);
  }
  for (int i = 0; i < IMPULSE_LEN; i++)
    impulse[i] /= peak;
}

// Adds a train of impacts to `out`: one every 1/rate_hz seconds (with the
// ~1% timing jitter real rolling elements show, since they slip a little),
// each scaled by `amp` and, if `modulation_hz` is nonzero, by how deep the
// defect currently is in the bearing's load zone -- an inner-race defect
// rotates with the shaft, so its impacts swell and fade once per revolution.
static void add_impacts(float *out, const float *impulse, float rate_hz, float amp, float modulation_hz, float phase)
{
  float period = VF_SAMPLE_RATE / rate_hz;
  float t = sim_uniform(0.0f, period);
  while (t < (float)VF_SNAPSHOT_SAMPLES) {
    float a = amp * sim_uniform(0.85f, 1.15f);
    if (modulation_hz > 0.0f)
      a *= 0.2f + 0.8f * (0.5f + 0.5f * cosf(2.0f * PI * modulation_hz * t / VF_SAMPLE_RATE + phase));
    int start = (int)t;
    for (int i = 0; i < IMPULSE_LEN && start + i < VF_SNAPSHOT_SAMPLES; i++)
      out[start + i] += a * impulse[i];
    t += period * sim_uniform(0.99f, 1.01f);
  }
}

void machine_snapshot(const machine_t *m, machine_condition_t condition, float severity, float *out)
{
  float s = condition == CONDITION_HEALTHY ? 0.0f : severity;
  // Load varies day to day, and slip (so running speed) with it.
  float shaft_hz = SYNC_SPEED_HZ * (1.0f - m->slip * sim_uniform(0.8f, 1.2f));
  float mesh_hz = shaft_hz * GEAR_TEETH;
  float wobble = sim_uniform(0.9f, 1.1f); // day-to-day amplitude variation

  float amp_1x = m->amp_1x * wobble;
  float amp_2x = m->amp_2x * wobble;
  float amp_3x = m->amp_2x * 0.3f * wobble;
  float resonance_noise = m->resonance_noise;
  float noise = m->noise;
  float mesh_modulation = 0.0f;  // depth of once-per-revolution gear-mesh amplitude modulation
  float impact_amp = 0.0f, impact_rate = 0.0f, impact_modulation = 0.0f;
  float friction_rate = 0.0f;     // random (aperiodic) friction bursts per second
  tone_t tones[MAX_TONES];
  int num_tones = 0;

  switch (condition) {
    case CONDITION_IMBALANCE:
      // Centrifugal force from an off-center mass: a pure 1x increase.
      amp_1x *= 1.0f + 5.0f * s;
      break;
    case CONDITION_MISALIGNMENT:
      // Two flexing coupling halves: dominant 2x, some 3x, a little 1x.
      amp_1x *= 1.0f + 1.0f * s;
      amp_2x *= 1.0f + 30.0f * s;
      amp_3x *= 1.0f + 30.0f * s;
      break;
    case CONDITION_LOOSENESS:
      // A loose foot or bearing fit makes the response nonlinear (it
      // clips): a long comb of harmonics, plus half-orders as the part
      // rattles every other revolution.
      for (int k = 1; k <= 10; k++) {
        add_tone(tones, &num_tones, k * shaft_hz, m->amp_1x * 4.0f * s * powf(0.75f, (float)(k - 1)), sim_uniform(0.0f, 2.0f * PI));
        add_tone(tones, &num_tones, (k - 0.5f) * shaft_hz, m->amp_1x * 2.0f * s * powf(0.7f, (float)(k - 1)), sim_uniform(0.0f, 2.0f * PI));
      }
      break;
    case CONDITION_BEARING_OUTER_RACE:
      impact_rate = BPFO_ORDER * shaft_hz;
      impact_amp = 0.25f * s;
      add_tone(tones, &num_tones, impact_rate, 0.004f * s, sim_uniform(0.0f, 2.0f * PI));
      break;
    case CONDITION_BEARING_INNER_RACE:
      impact_rate = BPFI_ORDER * shaft_hz;
      impact_amp = 0.25f * s;
      impact_modulation = shaft_hz;
      add_tone(tones, &num_tones, impact_rate, 0.003f * s, sim_uniform(0.0f, 2.0f * PI));
      add_tone(tones, &num_tones, impact_rate - shaft_hz, 0.0015f * s, sim_uniform(0.0f, 2.0f * PI));
      add_tone(tones, &num_tones, impact_rate + shaft_hz, 0.0015f * s, sim_uniform(0.0f, 2.0f * PI));
      break;
    case CONDITION_GEAR_TOOTH:
      // One cracked/chipped tooth: a jolt once per pinion revolution, which
      // also amplitude-modulates the mesh tone (sidebands at mesh +/- 1x).
      impact_rate = shaft_hz;
      impact_amp = 0.3f * s;
      mesh_modulation = 3.0f * s;
      break;
    case CONDITION_LUBRICATION:
      // Metal-to-metal contact through a thinning oil film: friction noise
      // excites the resonances broadband, in random bursts with no
      // repetition rate to find in the envelope spectrum.
      resonance_noise *= 1.0f + 6.0f * s;
      friction_rate = 400.0f * s;
      break;
    default:
      break;
  }
  // Every fault ends the same way: once the damage is advanced, the
  // machine is rattling and everything gets worse -- broadband noise and
  // looseness-like 1x growth. This is roughly when a traditional overall
  // vibration alarm finally notices a bearing, gear or lubrication fault.
  if (condition == CONDITION_BEARING_OUTER_RACE || condition == CONDITION_BEARING_INNER_RACE ||
      condition == CONDITION_GEAR_TOOTH || condition == CONDITION_LUBRICATION) {
    float late = s > 0.6f ? (s - 0.6f) / 0.4f : 0.0f;
    amp_1x *= 1.0f + 3.0f * late * late;
    noise *= 1.0f + 4.0f * late * late;
  }

  add_tone(tones, &num_tones, shaft_hz, amp_1x, m->phase_1x);
  add_tone(tones, &num_tones, 2.0f * shaft_hz, amp_2x, m->phase_2x);
  add_tone(tones, &num_tones, 3.0f * shaft_hz, amp_3x, sim_uniform(0.0f, 2.0f * PI));
  add_tone(tones, &num_tones, 2.0f * LINE_HZ, m->amp_line * wobble, sim_uniform(0.0f, 2.0f * PI));

  float impulse[IMPULSE_LEN];
  build_impulse(m, impulse);
  memset(out, 0, VF_SNAPSHOT_SAMPLES * sizeof(float));
  if (impact_amp > 0.0f)
    add_impacts(out, impulse, impact_rate, impact_amp, impact_modulation, sim_uniform(0.0f, 2.0f * PI));
  if (friction_rate > 0.0f) {
    // Poisson-timed bursts of random strength.
    for (float t = 0.0f; t < (float)VF_SNAPSHOT_SAMPLES;) {
      t += -logf(sim_uniform(1e-6f, 1.0f)) * VF_SAMPLE_RATE / friction_rate;
      int start = (int)t;
      float a = 0.06f * s * sim_uniform(0.2f, 1.0f) * (rand() & 1 ? 1.0f : -1.0f);
      for (int i = 0; i < IMPULSE_LEN && start + i < VF_SNAPSHOT_SAMPLES; i++)
        out[start + i] += a * impulse[i];
    }
  }

  resonator_t res[2];
  resonator_init(&res[0], m->resonance_hz[0], m->resonance_damping[0]);
  resonator_init(&res[1], m->resonance_hz[1], m->resonance_damping[1]);
  float mesh_phase = sim_uniform(0.0f, 2.0f * PI);
  float mesh2_phase = sim_uniform(0.0f, 2.0f * PI);
  float shaft_phase = sim_uniform(0.0f, 1.0f); // in revolutions
  for (int n = 0; n < VF_SNAPSHOT_SAMPLES; n++) {
    float t = (float)n / VF_SAMPLE_RATE;
    float v = out[n];
    for (int i = 0; i < num_tones; i++)
      v += tones[i].amp * sinf(2.0f * PI * tones[i].hz * t + tones[i].phase);
    // Gear mesh: the damaged tooth's pass (a narrow window of the
    // revolution, one tooth wide) briefly swells the mesh amplitude.
    float mesh_amp = m->amp_mesh * wobble;
    if (mesh_modulation > 0.0f) {
      float rev = shaft_hz * t + shaft_phase - m->defect_phase;
      float d = (rev - floorf(rev + 0.5f)) * (float)GEAR_TEETH; // distance from the bad tooth, in teeth
      mesh_amp *= 1.0f + mesh_modulation * expf(-d * d);
    }
    v += mesh_amp * sinf(2.0f * PI * mesh_hz * t + mesh_phase);
    v += 0.4f * m->amp_mesh * wobble * sinf(4.0f * PI * mesh_hz * t + mesh2_phase);
    v += resonance_noise * (resonator_step(&res[0], gauss()) + m->resonance_mix * resonator_step(&res[1], gauss())) / sqrtf(1.0f + m->resonance_mix * m->resonance_mix);
    v += noise * gauss();
    out[n] = v * m->gain;
  }
}

void machine_observe(health_monitor_t *hm, nn_t *nn, const machine_t *m, machine_condition_t condition, float severity, health_report_t *report)
{
  static float samples[VF_SNAPSHOT_SAMPLES];
  machine_snapshot(m, condition, severity, samples);
  for (int f = 0; f < VF_FRAMES_PER_SNAPSHOT; f++)
    health_monitor_add_frame(hm, samples + f * VF_FRAME_LEN);
  health_monitor_end_snapshot(hm, nn, report);
}

void machine_commission(health_monitor_t *hm, const machine_t *m)
{
  health_report_t report;
  for (int i = 0; i < HM_COMMISSIONING_SNAPSHOTS; i++)
    machine_observe(hm, NULL, m, CONDITION_HEALTHY, 0.0f, &report);
}

float fault_severity_on_day(int day, int onset_day, int failure_day)
{
  if (day <= onset_day)
    return 0.0f;
  if (day >= failure_day)
    return 1.0f;
  float x = (float)(day - onset_day) / (float)(failure_day - onset_day);
  return x * x;
}
