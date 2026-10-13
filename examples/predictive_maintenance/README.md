# Predictive Maintenance

Trains a small fully-connected network to tell, from a few seconds of vibration off an accelerometer bolted to an industrial machine, whether the machine is healthy or which of seven failure modes is developing in it -- months before it fails, while its overall vibration level still reads normal. Every snapshot becomes a frequency "fingerprint" computed with FFTs, and the network compares the machine's fingerprint now with the one it had when the sensor was installed. Built around an STM32 microcontroller; nothing to download -- training data comes from a physics-based simulator.

## In an embedded system

Motors, pumps, fans, compressors and gearboxes rarely fail without warning. A bearing defect, a loosening mount or a drying lubricant changes how a machine vibrates weeks or months before anything breaks -- but in the *pattern* of frequencies first, not the overall level. The traditional vibration monitor watches one number, the RMS vibration velocity of ISO 10816, against a threshold. That number barely moves until damage is advanced, and for bearing, gear and lubrication faults it often moves only in the last few days.

Predictive maintenance closes that gap by analyzing the whole spectrum, and doing it on the sensor node itself pays off in practice. A vibration sensor sampling at kilohertz rates produces megabytes per hour per machine. A plant has hundreds of machines, many of them running on battery-powered or low-bandwidth wireless nodes. A node that reduces each snapshot to "healthy" or "inner-race bearing defect, 100%" sends a few bytes instead, and it keeps working when the network is down.

This example targets that node:

| | |
|---|---|
| MCU | **STM32U585** -- Cortex-M33 at 160MHz with single-precision FPU and DSP instructions, 2MB flash, 786KB SRAM, low-power modes for battery operation |
| Sensor | **IIS3DWB** -- ST's wideband vibration accelerometer: 3-axis, flat frequency response to 6kHz, fixed 26.667kHz output data rate. One axis is used here; the firmware decimates by 4 to 6.67kHz. |
| Reference hardware | ST's **STWIN.box** (STEVAL-STWBXCORE) industrial condition-monitoring kit, which pairs exactly these two parts |

Any Cortex-M4F/M7/M33 with ~100KB of free RAM would do; nothing here is specific to the U5 beyond the compiler flags in the Makefile's `stm32` target.

The simulated machine is a common one: a 4-pole induction motor on 50Hz mains (running at about 1,460rpm, or 24.4Hz, depending on load) driving a gearbox, with 6206 ball bearings. The seven failure modes are the ones the [article](https://embeddedcomputing.com/technology/ai-machine-learning/predictive-maintenence/industrial-predictive-maintenance-maximizes-ai-at-the-edge) that motivated this example lists -- each with a well-known vibration signature:

| Condition | What's physically happening | Where it shows up in the fingerprint |
|---|---|---|
| `IMBALANCE` | an off-center mass on the rotor | 1x running speed (24Hz) grows |
| `MISALIGNMENT` | motor and gearbox shafts not collinear; the coupling flexes twice a revolution | 2x and 3x running speed grow |
| `LOOSENESS` | a loose foot or bearing fit; the response clips and rattles | a long comb of harmonics, plus half-orders (12Hz, 37Hz, ...) |
| `BEARING_OUTER` | a spall on the outer race; each passing ball strikes it | envelope spectrum: lines at the outer-race defect frequency (3.58x, ~87Hz) |
| `BEARING_INNER` | a spall on the inner race, rotating in and out of the load zone | envelope spectrum: the inner-race defect frequency (5.42x, ~132Hz), plus 1x from the modulation |
| `GEAR_TOOTH` | one cracked or chipped tooth on the pinion | sidebands around the gear-mesh frequency (23 teeth x 24Hz = ~560Hz); envelope line at 1x |
| `LUBRICATION` | a thinning oil film; metal-to-metal friction | the broadband high-frequency floor rises, with *no* periodicity in the envelope |

## How the fingerprint is computed

`vibration_features.[ch]` runs on the device. Each snapshot is 8 frames of 4,096 samples (about 4.9 seconds), and each frame passes through three FFTs (`../../fft.[ch]`):

1. **Vibration spectrum.** Hann window, 4,096-point FFT: 1.63Hz per bin, 0 to 3.3kHz. The shaft-rate faults (imbalance, misalignment, looseness), the gear-mesh tone and the friction floor all live here.
2. **Envelope.** A bearing defect makes a tiny impact every time a ball passes over it. Each impact rings the machine's structural resonances at 1-3kHz, and in the ordinary spectrum that's just a rise in the high-frequency floor, which looks the same as friction from poor lubrication. The information that tells them apart is *how often* the impacts happen. So the same FFT output is band-passed to 1-3.2kHz with the negative frequencies removed, and an inverse FFT turns it into the analytic signal, whose magnitude is the envelope of the resonance ringing.
3. **Envelope spectrum.** The FFT of that envelope has a line at the impact repetition rate: the outer-race (~87Hz) or inner-race (~132Hz) defect frequency, or 1x for a damaged gear tooth that strikes once per revolution. Friction stays flat. This is the classic *envelope analysis* technique of vibration-based condition monitoring.

The two power spectra are averaged across the 8 frames (Welch's method) and reduced to log energy in logarithmically spaced bands. The vibration spectrum gets 48 bands from 8Hz to 3.2kHz: single-bin bands around the shaft harmonics, ~13% wide up by the resonances. The envelope spectrum gets 16 bands from 8Hz to 500Hz. Each set is shifted to zero mean, so the sensor's mounting gain cancels out. The result is **64 numbers per snapshot**: the machine's fingerprint. The same pass also produces the ISO 10816 RMS velocity (10Hz-1kHz, integrated from the spectrum), which is used only for comparison.

`train.c` starts by printing which fingerprint bands each condition moves most on one simulated machine, at severity 0.5. It's the "look at your data" step, and the bands line up with the table above:

```
Largest fingerprint changes from one machine's baseline, at severity 0.5
(spectrum = vibration spectrum band, envelope = envelope spectrum band; change in log10 energy):
  HEALTHY        (0.68 mm/s):  spectrum 10-11Hz +0.32  spectrum 15-16Hz +0.31  spectrum 11-13Hz +0.29
  IMBALANCE      (2.33 mm/s):  spectrum 24-28Hz +0.96  spectrum 23-24Hz +0.95  spectrum 20-23Hz +0.93
  MISALIGNMENT   (1.37 mm/s):  spectrum 47-52Hz +2.24  spectrum 41-47Hz +2.24  spectrum 67-76Hz +2.18
  LOOSENESS      (2.31 mm/s):  spectrum 11-13Hz +2.98  spectrum 13-15Hz +2.87  spectrum 36-41Hz +2.48
  BEARING_OUTER  (0.70 mm/s):  envelope 83-107Hz +1.72  envelope 138-179Hz +1.43  spectrum 86-98Hz +1.33
  BEARING_INNER  (0.75 mm/s):  envelope 23-29Hz +1.46  envelope 107-138Hz +1.44  spectrum 1517-1717Hz +1.08
  GEAR_TOOTH     (0.68 mm/s):  spectrum 436-495Hz +1.41  spectrum 560-635Hz +1.38  envelope 23-29Hz +1.24
  LUBRICATION    (0.74 mm/s):  spectrum 1717-1945Hz +0.80  spectrum 1517-1717Hz +0.75  spectrum 2826-3200Hz +0.71
```

At this severity the bearing, gear and lubrication faults have each moved their fingerprint bands by a factor of 5-50 in energy, while the overall vibration level (the mm/s figure) hasn't changed at all.

## Past and present: the baseline

No two machines vibrate alike, even two of the same model off the same production line. The simulator varies every one: running speed (motor slip), residual imbalance (a 2.5x range), residual misalignment, gear-mesh level, the frequencies and damping of its structural resonances, noise floor, and sensor mounting gain. So a machine whose 1x peak is 50% higher than average may be developing an imbalance, or it may simply have been built that way.

`health_monitor.[ch]` (also on-device) resolves that the way a vibration analyst would: by comparing the machine with itself. The first 4 snapshots after installation are averaged into a **baseline fingerprint**, 64 floats that the firmware keeps in flash. After that, the model gets 128 inputs: the present fingerprint, and the present fingerprint minus the baseline -- how far this machine has moved from its own past. An alert is raised once snapshots have been classified as faulty 3 times in a row, so a single noisy snapshot never sends anyone out to the machine.

`train --present-only` trains the same network on the present fingerprint alone (64 inputs), so what the baseline is worth can be measured rather than assumed -- see [Results](#results).

## Model architecture

![Pipeline: IIS3DWB accelerometer at 26.7kHz decimated to 6.67kHz, into a Hann-windowed 4096-point FFT, splitting into a 48-band vibration spectrum and a 16-band envelope spectrum (1-3.2kHz band, inverse FFT, magnitude of analytic signal, FFT), combined into a 64-value fingerprint. The first 4 fingerprints are averaged into a baseline kept in flash. The input layer takes 128 values: the present fingerprint and present minus baseline. Then FC 32 GELU, FC 16 GELU, 8-way softmax output, and an alert after 3 faulty snapshots in a row.](architecture.svg)

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 128 | 64-band present fingerprint + 64-band change from the machine's baseline (64 with `--present-only`) |
| 1 | FC | 32 | GELU |
| 2 | FC | 16 | GELU |
| 3 | Output | 8 | Softmax over {HEALTHY, IMBALANCE, MISALIGNMENT, LOOSENESS, BEARING_OUTER, BEARING_INNER, GEAR_TOOTH, LUBRICATION} |

4,792 trainable parameters. This is the simplest network in this repository's examples, and the reason is worth noting: the hard part -- turning 33,000 raw samples into a few dozen numbers that capture what matters -- is done by signal processing that's been understood for decades, and the history the problem needs is supplied by the baseline rather than learned by a recurrent layer. What's left for the network is drawing boundaries in a 128-dimensional space that's already well organized.

## Build and run

```
cd examples/predictive_maintenance
make
```

Train (simulates 400 training machines and 80 validation machines, 20 snapshots each -- under a minute):
```
./train model.txt
```

Evaluate on 100 more unseen machines, then run 160 more to failure:
```
./test model.txt
```

Follow one machine from installation to failure, day by day (`<condition>` is any name from the table above; the optional seed replays the same machine):
```
./monitor model.txt BEARING_INNER 42
```

Train the no-baseline comparison model:
```
./train --present-only model_present.txt
./test model_present.txt
```

Cross-compile the on-device code for the STM32U585 and report its size (needs only `arm-none-eabi-gcc` -- STM32CubeIDE bundles one; point `ARM_CC` at it if it isn't on your `PATH`):
```
make stm32 ARM_CC=/path/to/arm-none-eabi-gcc
```

## Sample output

Training log:
```
Simulating 400 training machines and 80 validation machines (20 snapshots each)...
Creating new model.
train error, validation error, learning rate
0.22005, 0.10190, 0.01000
0.09095, 0.06413, 0.01000
0.07137, 0.07678, 0.01000
...
0.00298, 0.01277, 0.01000
No validation improvement for 10 epochs (best: 0.00829) -- stopping early.
Final (last epoch) train error: 0.002981, validation error: 0.012770
Best validation error (the model saved to disk): 0.008295
Validation accuracy (last epoch): 1276/1280 = 99.7%
Training epochs: 54
```

`test model.txt`:
```
Model inputs: present fingerprint + change from baseline

Snapshot accuracy: 100 unseen machines, 16 snapshots each, fault severity 0.05-1.00
Confusion matrix (rows = actual, columns = predicted):
              HEALTH IMBALA MISALI LOOSEN BEARIN BEARIN GEAR_T LUBRIC 
HEALTHY          189      0      0      0      0      0      0      0  (100.0%)
IMBALANCE          0    195      0      0      0      0      0      0  (100.0%)
MISALIGNMENT       0      0    204      0      0      0      0      0  (100.0%)
LOOSENESS          0      0      0    191      0      0      0      0  (100.0%)
BEARING_OUTER      1      0      0      0    210      0      0      0  ( 99.5%)
BEARING_INNER      1      0      0      0      0    217      0      1  ( 99.1%)
GEAR_TOOTH         0      0      0      0      0      0    198      0  (100.0%)
LUBRICATION        1      0      0      0      0      0      0    192  ( 99.5%)
Overall accuracy: 1596/1600 = 99.75%

Fault accuracy by severity (0 = no fault, 1 = failing):
  0.05-0.10:   77/81   =  95.1%
  0.10-0.20:  127/127  = 100.0%
  0.20-0.40:  287/287  = 100.0%
  0.40-0.70:  469/469  = 100.0%
  0.70-1.00:  447/447  = 100.0%

Run to failure: 20 unseen machines per condition, failing on day 120 (onset day 10-40), one snapshot a day.
Warning = days before failure that each alarm first fired (held 3 days, at least a day before failure); ISO alarm = RMS velocity >= 2.8 mm/s.
               NN warned diagnosis  NN median  NN severity   ISO warned   ISO median
                 in time   correct    warning   at warning      in time      warning
IMBALANCE          20/20     20/20    70 days         0.07        17/20      32 days
MISALIGNMENT       20/20     20/20    78 days         0.04        15/20      12 days
LOOSENESS          20/20     20/20    81 days         0.03        18/20      25 days
BEARING_OUTER      20/20     19/20    78 days         0.06         6/20       4 days  (1 first called LUBRICATION)
BEARING_INNER      20/20     19/20    68 days         0.07         2/20       1 days  (1 first called GEAR_TOOTH)
GEAR_TOOTH         20/20     20/20    73 days         0.07         7/20       5 days
LUBRICATION        20/20     19/20    68 days         0.05         8/20       3 days  (1 first called BEARING_INNER)
False alarms (alert raised before any fault existed): 0 of 160 machines, 0 alert-days in 5845 machine-days of healthy running.
```

`monitor model.txt BEARING_INNER 42` -- printed weekly, plus every day the model's output or the ISO alarm state changes:
```
Machine (seed 42): BEARING_INNER develops from day 25, fails on day 120.

Commissioned: baseline fingerprint averaged over 4 snapshots.

day  severity  ISO 10816 velocity       model says            model alert
  1     0.000   0.52 mm/s zone A         HEALTHY        100%  
  7     0.000   0.57 mm/s zone A         HEALTHY        100%  
...
 42     0.032   0.53 mm/s zone A         HEALTHY         99%  
 47     0.054   0.50 mm/s zone A         BEARING_INNER   61%  
 49     0.064   0.50 mm/s zone A         BEARING_INNER  100%  ALERT raised
 56     0.106   0.49 mm/s zone A         BEARING_INNER  100%  ALERT
...
105     0.709   0.65 mm/s zone A         BEARING_INNER  100%  ALERT
112     0.839   1.06 mm/s zone A         BEARING_INNER  100%  ALERT
119     0.979   1.79 mm/s zone B         BEARING_INNER  100%  ALERT
120     1.000   2.01 mm/s zone B         BEARING_INNER  100%  ALERT
```

On this machine the inner-race defect was named on day 49, 71 days before failure. The ISO 10816 velocity never left the "acceptable" zones A and B, so a threshold monitor would have stayed silent until the bearing seized.

## Results

Each `test` run simulates machines that weren't in training, so numbers shift by a few percent from run to run, and a separately trained model shifts them a little more; the pattern below has held across every run made while writing this example.

**Snapshot accuracy** is 99.75% overall. At the earliest severities in the test (0.05-0.10, where the fault's energy is a small fraction of what it will be at failure) it's 95%, and 100% above that.

**Run to failure** is the result that matters. Over 140 simulated machines run to failure, the model raised an alert on every one, with a median of **68 to 81 days' warning** depending on the fault. It warned at severity 0.03-0.07, the stage where the bearing defect is a pit, not a crater. Its diagnosis at the first alert was right 137 times out of 140. The three misses were adjacent conditions at the very first alert, such as an outer-race bearing defect first called lubrication: at the earliest stage both are mostly a rise in high-frequency energy, and the defect's repetition rate only stands out in the envelope a few days later. The model then corrected itself. Across 5,845 days of healthy running (160 machines), it raised **no false alarms**.

The ISO 10816 alarm on the same machines and days:

| Fault | Model: median warning | ISO 10816 alarm: warned at all | ISO: median warning |
|---|---|---|---|
| Imbalance | 70 days | 17 / 20 | 32 days |
| Misalignment | 78 days | 15 / 20 | 12 days |
| Looseness | 81 days | 18 / 20 | 25 days |
| Bearing, outer race | 78 days | 6 / 20 | 4 days |
| Bearing, inner race | 68 days | 2 / 20 | 1 day |
| Gear tooth | 73 days | 7 / 20 | 5 days |
| Lubrication | 68 days | 8 / 20 | 3 days |

The ISO alarm does reasonably on the faults that raise low-frequency vibration (imbalance, misalignment, looseness), but still weeks later than the model. On the four faults whose early signature is high-frequency or periodic-impact energy, it misses most failures outright and gives a few days' notice on the rest -- which matches what condition-monitoring practice says about overall-level alarms and rolling-element bearings.

### What the baseline is worth

The same network trained and tested without the baseline (`train --present-only`, 64 inputs):

| | With baseline (128 inputs) | Present fingerprint only (64 inputs) |
|---|---|---|
| Snapshot accuracy, overall | 99.75% | 96.00% |
| Snapshot accuracy, severity 0.05-0.10 | 95.1% | 67.2% |
| Snapshot accuracy, severity 0.10-0.20 | 100.0% | 83.2% |
| Diagnosis right at first alert (run to failure) | 137 / 140 | 124 / 140 |
| Median warning, imbalance | 70 days (severity 0.07) | 55 days (severity 0.19) |
| Median warning, lubrication | 68 days (severity 0.05) | 58 days (severity 0.16) |
| False alarms, healthy running | **0** machines | **9 of 160** machines (92 alert-days in 6,051) |

Without the baseline, the model has to decide whether a machine's 1x level or high-frequency floor is abnormal by comparing it with every other machine it trained on. Machines legitimately differ by more than an early fault changes them. So it waits until imbalance and lubrication faults are nearly three times as advanced, confuses looseness with imbalance more often, and occasionally decides a healthy machine with an unusually strong 1x or a lively resonance is faulty -- 9 false alarms where the baseline model had none. A false alarm sends a technician to a healthy machine, and after a few of them people start ignoring the alerts, so this is the cost that matters most in practice. Faults that add something new rather than changing the size of something already there -- envelope-spectrum lines, half-orders, gear-mesh sidebands -- are caught about as early either way.

Comparing a machine's frequency fingerprint now with the one it had when healthy is cheap: 64 floats of flash per machine and one subtraction per input.

## Running it on the STM32

`vibration_features.[ch]`, `health_monitor.[ch]`, `../../fft.[ch]` and `../../nn.[ch]` are the firmware. `train.c`, `test.c` and `monitor.c` call the same functions on the desktop; `machine_sim.[ch]` stands in for the machine and the sensor. `make stm32` compiles the firmware half for the Cortex-M33 (`-mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -Os`):

```
   text	   data	    bss	    dec	    hex	filename
    394	      0	      0	    394	    18a	stm32/fft.o
    432	     32	      0	    464	    1d0	stm32/health_monitor.o
  24227	      0	      0	  24227	   5ea3	stm32/nn.o
   1070	      0	      0	   1070	    42e	stm32/vibration_features.o
  26123	     32	      0	  26155	   662b	(TOTALS)
```

Those are object sizes; linking with `--gc-sections` drops the parts of `nn.c` an inference-only device doesn't use (training, model file I/O). The example-specific code is under 2KB.

**Model in flash.** Quantize to int8 and export in the zero-copy inplace format (see the top-level README), then embed it with `xxd -i`:
```
../../quantize model.txt model_q.txt
../../export model_q.txt model_q.inplace --inplace
xxd -i model_q.inplace > model_q_inplace.h
```
That's **5,092 bytes** (19,232 as float32), and `nn_load_model_inplace()` runs it straight out of flash with no RAM copy of the weights. Quantization cost nothing measurable here: `test model_q.inplace` gave 99.81% snapshot accuracy (vs. 99.75% for the float model on a different random draw of test machines), median warnings of 69-85 days, and still no false alarms.

**RAM.** One `health_monitor_t` is about 67KB, almost all of it `vibration_features_t`'s FFT buffers and the two accumulated spectra (4,096 x 3 + 2,049 x 2 floats). Add one frame of samples per DMA buffer (16KB; double-buffered, 32KB) and the network's activations (under 1KB). That's around 100KB of the U585's 786KB. A smaller part would need a smaller FFT (2,048 points halves it, at 3.3Hz resolution) or fixed-point buffers.

**Firmware loop.** `monitor.c`'s main loop is shaped like the firmware's. On the device it becomes:

```c
static health_monitor_t hm;                 // ~67KB, in .bss
nn_t *nn = nn_load_model_inplace(model_q_inplace, model_q_inplace_len);
health_monitor_init(&hm, nn->width[0]);
if (baseline_saved_in_flash())              // commissioned on an earlier boot
  health_monitor_set_baseline(&hm, saved_baseline);
for (;;) {
  sleep_until_next_snapshot();              // e.g. hourly; the MCU and sensor sleep in between
  for (int f = 0; f < VF_FRAMES_PER_SNAPSHOT; f++)
    health_monitor_add_frame(&hm, wait_for_accelerometer_frame()); // IIS3DWB FIFO -> SPI DMA, decimated by 4
  health_report_t report;
  health_monitor_end_snapshot(&hm, nn, &report);
  if (report.commissioning && hm.baseline_snapshots == HM_COMMISSIONING_SNAPSHOTS)
    save_baseline_to_flash(hm.baseline);
  else if (report.alert)
    transmit(report.condition, report.confidence, report.velocity_rms);  // a few bytes
}
```

The three functions not defined in this repository (`wait_for_accelerometer_frame()`, the flash helpers and `transmit()`) are the board-specific parts: the IIS3DWB driver (ST publishes one in its `STMems_Standard_C_drivers` repository), a decimating low-pass filter ahead of `health_monitor_add_frame()` (CMSIS-DSP's `arm_fir_decimate_f32()`), and whatever radio or bus the node reports over.

**Faster FFTs.** `../../fft.c` is portable C. With CMSIS-DSP available, `arm_rfft_fast_f32()` computes the same real-input transform several times faster, and it's a drop-in replacement at `vibration_features.c`'s three call sites. Either way the work runs once per snapshot (24 transforms of 4,096 points every few seconds at most), so for a node that takes one snapshot an hour, speed matters for battery life rather than for keeping up.

This example hasn't been run on hardware; the sizes above come from the cross-compiler.

## Limitations

The numbers above measure a simulator, not a factory floor, and should be read that way:

- **The physics is textbook, and the levels are guesses.** Each fault's signature (which frequencies, which modulation, which resonances) follows the vibration-analysis literature. How strongly a fault shows at a given severity, how noisy real machines are, and the 0-to-1 severity scale itself are this simulator's choices. The size of the gap between the model and the ISO alarm depends on those choices. That a gap exists, and that it's largest for bearing, gear and lubrication faults, is well established in practice.
- **One fault at a time, on one machine type.** Real machines can develop two faults at once (misalignment often causes bearing wear), and other machine types have other speeds, bearings and gear ratios. A model for a different machine needs training on its own fault frequencies, or order-normalized fingerprints (bands at multiples of a measured running speed rather than fixed Hz) to generalize across speeds.
- **The baseline assumes a healthy start.** If a machine is commissioned with a fault already present, the baseline absorbs it. The present-fingerprint half of the input still sees the fault, but only as well as the `--present-only` model does.
- **Real data is the next step.** Public bearing datasets -- Case Western Reserve University's seeded-fault data, and the IMS (NASA prognostics repository) and FEMTO/PRONOSTIA run-to-failure sets -- cover a subset of these faults on real rigs. Feeding their raw signals (resampled to 6.67kHz) through `vibration_features.[ch]` is the honest test of whether this front end and model carry over.
