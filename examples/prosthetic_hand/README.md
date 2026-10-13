# Prosthetic Hand

Trains a small fully-connected network to read what the wearer of a powered prosthetic hand and wrist is trying to do -- open the hand, close it, rotate the wrist either way, flex or extend it, or nothing -- from the electrical activity of the muscles left in their forearm, picked up by a ring of 8 electrodes inside the prosthesis socket. A new decision every 50ms. The data is real: [Ninapro](https://ninapro.hevs.ch) recordings of 40 intact subjects and 11 people with a transradial (below-elbow) amputation. This README reports the numbers as measured, including how far they are from good enough for the amputees.

## In an embedded system

Most powered prosthetic hands are still driven by *direct control*: two electrodes, one over a muscle that opens the hand and one over a muscle that closes it. Each muscle drives one motor direction, and to rotate the wrist instead the wearer switches modes, typically with a quick co-contraction of both. It is reliable and slow, and it gets slower with every degree of freedom a modern hand adds.

*Pattern-recognition control* reads the whole forearm instead. Every movement the wearer attempts -- even with the hand missing, the muscles that used to move it still contract -- produces its own pattern of activity around the ring of electrodes, and a classifier maps that pattern to a motor command. Commercial systems (Coapt's, Ottobock's Myo Plus) have offered this for transradial prostheses since the 2010s. The classifier in the myoelectric-control literature, and the baseline here, is linear discriminant analysis (LDA) on four simple features per electrode (Englehart & Hudgins, 2003, "A robust, real-time control scheme for multifunction myoelectric control", *IEEE Transactions on Biomedical Engineering*).

It has to run inside the prosthesis: on a battery, within about 100-150ms of the muscle contracting (users notice longer delays -- Farrell & Weir, 2007, "The optimal controller delay for myoelectric prostheses", *IEEE TNSRE*), with no network. And it has to be **calibrated to its wearer**, every time the prosthesis is put on: skin contact, sweat and the socket's exact position change from one donning to the next, and one person's muscle patterns barely resemble another's.

| | |
|---|---|
| MCU | **STM32U585** -- Cortex-M33 at 160MHz with single-precision FPU, 2MB flash, 786KB SRAM (the same part as [`predictive_maintenance`](../predictive_maintenance/README.md)) |
| Analog front end | **TI ADS1298** -- 8 channels of 24-bit biopotential ADC, here at 2kHz each |
| Electrodes | 8 dry surface electrodes spaced evenly around the forearm, in the socket |

Any Cortex-M4F/M7/M33 with ~250KB of free RAM would do; see [Running it on the device](#running-it-on-the-device).

## The data

Ninapro ([Atzori et al., 2014](https://www.nature.com/articles/sdata201453)) recorded 12 Delsys Trigno electrodes at 2kHz while subjects copied movements shown on a screen. Eight electrodes form a ring around the forearm just below the elbow; two more sit over the finger flexor and extensor muscles, and two on the biceps and triceps. This example uses:

- **DB2**: 40 intact subjects. 28 train the shared model, 4 are its validation set, and 8 are held out for testing.
- **DB3**: 11 transradial amputees, all held out for testing, who performed the movements with their phantom hand.

Of Ninapro's 17 exercise-1 movements, the six that drive a hand and wrist rotator are used, plus rest between movements: 5 "abduction of the fingers" (`HAND_OPEN`), 6 "fingers flexed together" (`HAND_CLOSE`), 10 and 9 wrist pronation and supination, 13 and 14 wrist flexion and extension. Each movement is held for 5 seconds and repeated 6 times, with 3 seconds of rest between. Labels are Ninapro's *refined* ones (`restimulus`), which mark when each subject actually moved rather than when the cue appeared -- worth about 10 percentage points of accuracy over the cue labels.

Only the **ring of 8** electrodes is used by default, because that's what a transradial socket can hold and what fits one ADS1298. `make CHANNELS=10` adds the two targeted forearm electrodes, and `CHANNELS=12` the upper-arm pair; see [Results](#results) for what they buy.

The Makefile downloads only exercise 1 from each subject's archive: `fetch_ninapro.py` uses HTTP range requests to pull one ~100-170MB file out of each 250-470MB zip, about 7GB in all instead of 22GB. Ninapro asks that work using the data cite the papers above; its site gives no other license terms for DB2 and DB3. The data isn't redistributed here.

## How the features are computed

`emg_features.[ch]` runs on the device. Every 50ms it looks at the last 200ms (400 samples) on each electrode and computes the four features of Hudgins, Parker & Scott (1993), "A new strategy for multifunction myoelectric control", *IEEE Transactions on Biomedical Engineering*:

| Feature | What it measures |
|---|---|
| MAV, mean absolute value (log) | how hard the muscle under the electrode is contracting |
| WL, waveform length (log) | intensity and frequency together |
| ZC, zero crossings | the signal's dominant frequency, cheaply |
| SSC, slope sign changes | the same, weighted toward higher frequencies |

That makes 32 numbers per decision. `intent_decoder.[ch]` then standardizes each one by its mean and spread in the wearer's latest calibration, so the model never sees electrode gain, skin impedance or how strong the wearer is.

`train.c` starts by printing how active each electrode is, on average, for each movement across the 28 training subjects (in standard deviations from that subject's calibration mean). Each movement uses a different part of the ring:

```
                     ch1    ch2    ch3    ch4    ch5    ch6    ch7    ch8
  REST               -0.91  -0.95  -1.01  -0.96  -0.93  -0.96  -1.05  -1.14
  HAND_OPEN          -0.01  +0.08  +0.17  +0.08  -0.12  -0.07  +0.12  +0.20
  HAND_CLOSE         +0.49  +0.89  +0.51  +0.13  +0.13  +0.35  +0.43  +0.18
  WRIST_PRONATION    +0.09  -0.20  -0.17  +0.01  +0.10  +0.12  +0.05  +0.11
  WRIST_SUPINATION   -0.10  -0.08  -0.04  -0.19  -0.00  +0.13  +0.07  +0.11
  WRIST_FLEXION      +0.37  +0.54  +0.84  +1.27  +0.92  +0.22  +0.14  +0.26
  WRIST_EXTENSION    +0.57  +0.28  +0.30  +0.32  +0.47  +0.70  +0.77  +0.83
```

Wrist flexion and extension light up opposite sides of the forearm, and a fist the flexor side. An open hand and the two wrist rotations are much fainter: spreading the fingers takes little force, the muscles that rotate the forearm lie deep, and the strongest supinator, the biceps, is above the elbow. Those three turn out to be the hardest to tell apart.

Other front ends were measured on the same data (per-subject, calibrated on four repetitions) before settling on this one. Autoregressive coefficients added 2 points for LDA (not tried with the network). Spectral band powers added 1.6 points for LDA and nothing for the network. Appending the features of the previous 1-3 windows helped both, LDA a little more, and the longer histories make the hand slower to respond to a change. A 300ms window added 2 points for LDA, at the cost of 100ms more delay.

## Calibration and the shared model

A prosthesis is calibrated each time it's put on: the wearer is prompted through each movement in turn while the controller records. Here, **calibration with 1 repetition** means one 5-second hold of each movement plus the rest between them -- under a minute.

`intent_decoder.[ch]` (on-device) uses calibration two ways:

1. **Standardization.** The mean and spread of each feature during calibration (64 floats, kept in flash until the next calibration).
2. **Fine-tuning.** The calibration data trains the network itself, on the device: 10 passes with Adam (`intent_decoder_fine_tune()`), starting from a **shared model** that `train.c` trained on 28 other people.

`test.c` compares four decoders on the same calibration data:

- **LDA**: trained on the calibration data alone, as commercial controllers do.
- **NN, scratch**: the same network, trained from random weights on the calibration data alone.
- **NN, shared**: the shared model, unchanged. Only the standardization is calibrated.
- **NN, tuned**: the shared model, fine-tuned on the calibration data. This is what the firmware does.

## Model architecture

![Pipeline: 8 surface electrodes around the forearm sampled at 2kHz by an ADS1298, into Hudgins features (log MAV, log WL, zero crossings, slope sign changes) per electrode every 50ms over the last 200ms, standardized with the wearer's calibration mean and spread. Calibration at each donning also fine-tunes the network. The input layer takes 32 values, then FC 64 GELU, FC 32 GELU, 7-way softmax output, then the motor command: rest if confidence is below 0.8, majority of the last 5 decisions, speed from contraction effort.](architecture.svg)

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 32 | 4 standardized features x 8 electrodes |
| 1 | FC | 64 | GELU |
| 2 | FC | 32 | GELU |
| 3 | Output | 7 | Softmax over {REST, HAND_OPEN, HAND_CLOSE, WRIST_PRONATION, WRIST_SUPINATION, WRIST_FLEXION, WRIST_EXTENSION} |

4,423 trainable parameters. Like `predictive_maintenance`'s, it's a plain fully-connected network behind a signal-processing front end, and for the same reason: the features already summarize the signal, and each window is classified on its own.

What turns the network's output into a motor command matters as much as the network. The decoder treats any classification with a softmax probability below **0.8 as rest**, because a hand that pauses is a much smaller problem than a hand that does the wrong thing (Scheme, Hudgins & Englehart, 2013, "Confidence-based rejection for improved pattern recognition myoelectric control", *IEEE Transactions on Biomedical Engineering*). It then takes the **majority of the last 5 decisions** (250ms). The speed comes from how hard the wearer is contracting relative to their calibration (proportional control).

## Build and run

```
cd examples/prosthetic_hand
make
```

The first `make` downloads the 51 recordings (about 7GB; a few minutes on a fast connection) and turns them into `features.csv` (184,000 feature rows, about 2 minutes).

Train the shared model (seconds):
```
./train model.bin
```

Calibrate, fine-tune and test it on the 19 held-out people, alongside LDA and a network trained from scratch:
```
./test model.bin
```

Stream the held-out sessions through the firmware decoder continuously, as the wearer would experience it:
```
./replay model.bin          # all 19, summarized
./replay model.bin 2 33     # one person (DB2 subject 33), with a timeline of each movement
```

Use 10 or 12 electrodes instead of 8 (rebuilds `features.csv`):
```
make CHANNELS=12
./train model12.bin && ./test model12.bin
```

Cross-compile the on-device code for the STM32U585 and report its size:
```
make stm32 ARM_CC=/path/to/arm-none-eabi-gcc
```

## Sample output

Training log (`train.c` early-stops on what the model is for: validation accuracy after each validation subject has been calibrated and the model fine-tuned on their first repetition):
```
Training on 98091 windows from 28 subjects, validating on 14112 windows from 4 others.
...
train error, validation error, validation accuracy (shared), validation accuracy (fine-tuned on 1 rep)
1.11787, 1.63859, 45.4%, 61.2%
0.99536, 1.59095, 47.0%, 62.1%
...
0.85947, 1.81656, 45.5%, 62.2%
...
No validation improvement for 10 epochs (best: 62.2%) -- stopping early.
Best fitted validation accuracy (the model saved to disk): 62.2%
Training epochs: 15
```

`test model.bin`:
```
Intact (DB2 subjects 33-40): mean window accuracy on repetitions 2 and 5, by calibration data
                  1 rep   2 reps   4 reps
  LDA             63.8%    68.7%    71.2%
  NN, scratch     64.6%    73.1%    76.6%
  NN, shared      51.7%    51.9%    52.0%
  NN, tuned       66.8%    74.0%    77.9%
  NN, tuned, per subject (1 / 2 / 4 reps):
    s33 66/72/79   s34 73/79/81   s35 65/69/72   s36 64/66/70
    s37 70/79/82   s38 59/71/73   s39 52/68/74   s40 86/88/93

Intact (DB2 subjects 33-40), NN tuned on 1 rep: confusion matrix (rows: actual, columns: decoded, % of row):
                      REST HAND_O HAND_C PRONAT SUPINA FLEXIO EXTENS
  REST                84.7    0.4    4.0    4.4    2.9    1.8    1.8
  HAND_OPEN           21.0   44.5   11.6    4.6    6.7    5.7    6.0
  HAND_CLOSE           9.0    5.5   71.0    3.1    6.9    3.0    1.6
  WRIST_PRONATION     14.4    3.8    0.8   63.8   12.2    3.9    1.1
  WRIST_SUPINATION     6.8    4.9    1.1   21.3   61.8    2.2    1.8
  WRIST_FLEXION        1.4    3.5    0.5    3.8    3.0   74.0   13.7
  WRIST_EXTENSION     15.8    3.5    0.9    6.0    4.5   10.2   59.2

Amputee (DB3 subjects 1-11): mean window accuracy on repetitions 2 and 5, by calibration data
                  1 rep   2 reps   4 reps
  LDA             44.1%    50.2%    55.2%
  NN, scratch     44.7%    51.9%    55.9%
  NN, shared      24.6%    26.6%    26.6%
  NN, tuned       44.6%    52.1%    56.9%
  NN, tuned, per subject (1 / 2 / 4 reps):
    s1 48/56/61   s2 50/61/68   s3 56/64/65   s4 36/44/45
    s5 36/40/48   s6 48/55/55   s7 18/19/20   s8 49/64/68
    s9 64/66/71   s10 28/36/46   s11 59/68/80
```

`replay model.bin`, the firmware decoder run continuously over each held-out session, transitions included (calibrated on repetition 1; scored on repetitions 2 and 5):
```
Intact (DB2 subjects 33-40), calibrated on repetition 1, continuous decoding of repetitions 2 and 5:
                                    response   named  correct  wrong  drift
  raw                                  467ms   96/96    59.7%  26.5%  17.5%
  voted                                621ms   95/96    60.7%  24.5%  18.3%
  rejected < 0.7                       583ms   95/96    53.3%  17.2%  11.0%
  rejected < 0.8                       655ms   95/96    50.0%  13.8%   8.5%
  rejected < 0.9                       729ms   95/96    44.7%   9.4%   5.7%
  decoder (rejected < 0.8, voted)      801ms   95/96    51.2%  11.9%   8.7%

Amputee (DB3 subjects 1-11), calibrated on repetition 1, continuous decoding of repetitions 2 and 5:
                                    response   named  correct  wrong  drift
  raw                                  514ms 127/132    40.0%  48.4%  46.6%
  voted                                698ms 126/132    41.4%  46.8%  46.3%
  rejected < 0.7                       833ms 127/132    29.8%  26.8%  24.1%
  rejected < 0.8                       967ms 126/132    26.2%  20.9%  17.9%
  rejected < 0.9                      1212ms 123/132    21.4%  14.0%  11.5%
  decoder (rejected < 0.8, voted)     1260ms 119/132    25.7%  17.5%  16.0%
```

*Response* is the time from a movement's labeled onset to the first decision naming it. *Correct* and *wrong* are the shares of decisions during a movement that name it, or name a different movement (a decision of rest is neither). *Drift* is the share of decisions during rest that name any movement.

`replay model.bin 2 33` -- one intact subject, one character per 50ms decision of the firmware decoder, from 1 second before each movement's labeled onset to 1 second after its end:
```
  . rest  O open  C close  P pronation  S supination  F flexion  E extension

  HAND_OPEN         actual   ....................OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO....................
                    decoded  ...............................................OOOO...OOOOSSSSOOOOOSSSS....OOOOOOO...OOOO...OOOOCOO...............PPPPPPP.......................................

  HAND_CLOSE        actual   ....................CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC....................
                    decoded  .......................CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC...................

  WRIST_PRONATION   actual   ....................PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP....................
                    decoded  ........................................SSSSSPPPPPPPPPPPPPPP...PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP...........................................

  WRIST_SUPINATION  actual   ....................SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS....................
                    decoded  P..........PPPPPPPPPPPPPPP...SSSS.....SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSOOOOOOOOOOOO.................PP................PPP

  WRIST_FLEXION     actual   ....................FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF....................
                    decoded  ..........................EEEE...EEEEEEEEEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF....EEEEEEEEEEEEEEEEEEP.PP..........

  WRIST_EXTENSION   actual   ....................EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE....................
                    decoded  ............................EEEEE.....EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE......EEEE............
```

A fist is decoded within 150ms and held without a break. Wrist extension takes about 400ms and one short gap, then holds. The two wrist rotations each spend a moment as the *other* rotation, and flexion starts and ends as extension. The end is explainable: Ninapro's labels span the whole movement, the return to neutral included, and returning from a flexed wrist *is* an extension, made by the extensor muscles. The pronation decoded before supination is the same effect from the previous repetition's return. Some of the "wrong" and "drift" decisions in the summary are the decoder reporting a muscle pattern the labels don't describe. The open hand is this subject's weakest movement: late, intermittent, and mixed with supination.

## Results

`test.c` uses fixed random seeds, so the LDA and from-scratch rows are identical on every run. The shared model differs between `train` runs. Across seven of them, the tuned network's accuracy varied by up to about 2 points (intact, 1 rep: 66.6-68.4%; amputee, 1 rep: 42.9-45.2%), and the comparisons below held in every one -- except, as noted, the amputees at one repetition, where the network and LDA trade places from run to run. The continuous-control numbers move a little more between models: replaying two of them, the decoder's wrong-movement rate for intact subjects was 11.9% and 10.9%, and for amputees 17.5% and 18.1%.

**Window accuracy.** For the intact subjects, the fine-tuned network beats LDA by 3-5 points with one repetition of calibration and by 6-8 points with four. That's the difference between the 64% and 68% rows, or 71% and 78% -- real, but not transformative. For the amputees, the network is within about a point of LDA at one repetition (ahead in most training runs, behind in one) and leads by about 2 points with more; all four decoders are within 2-3 points of each other.

**What the shared model is worth.** Without fine-tuning, a model trained on 28 other people gets 52% on a new intact person and 25-27% on an amputee, even with the features standardized to them. People's muscle patterns differ too much to share a decoder. As a *starting point* for fine-tuning it helps a little over training the same network from scratch on the same calibration data: 2-4 points for intact subjects with one repetition, 1-2 points with more, and 0-2 for amputees. And it gets there in 10 passes over the data instead of 40. That's less work for a battery-powered device. Its validation curve during `train` shows why the shared model can't do more. Fine-tuned accuracy peaks within the first few epochs and stays flat while the model keeps learning the training subjects. What transfers between people is the general shape of the problem, and the network learns that fast.

**Calibration time** is the biggest lever available to the wearer. Going from one repetition of the movements to two (about 50 seconds more) is worth 5-7 points to every decoder, two more add another 2.5-5, and the curve hasn't flattened at four.

**Electrodes** are the biggest lever available to the designer. Mean window accuracy, 1 / 4 repetitions of calibration:

| Electrodes | Intact, LDA | Intact, NN tuned | Amputee, LDA | Amputee, NN tuned |
|---|---|---|---|---|
| 8 (forearm ring) | 63.8 / 71.2% | 66.8 / 77.9% | 44.1 / 55.2% | 44.6 / 56.9% |
| 10 (+ finger flexor, extensor) | 65.0 / 73.2% | 68.6 / 80.7% | 45.0 / 57.3% | 45.6 / 58.6% |
| 12 (+ biceps, triceps) | 69.2 / 78.1% | 69.1 / 81.4% | 48.3 / 62.6% | 47.8 / 62.5% |

The two upper-arm electrodes help most, presumably through the biceps, the forearm's strongest supinator. With all 12, LDA catches up with the network. The network's advantage is largest when its input is poorest, which is the realistic case for a socket.

**Continuous control** (`replay`) is the measure that matters to a wearer. Here the gap between intact and amputee subjects shows plainly. With every window decoded as is, an intact subject's hand would do the wrong thing in about a quarter of the decisions during a movement, and an amputee's in almost half. Majority voting barely changes that: the errors last longer than 250ms, so they aren't the flicker voting is meant to remove. Confidence rejection does change it. With the decoder's settings (rejection at 0.8, then voting), wrong movements fall by more than half for intact subjects (26.5% → 11.9%) and by almost two-thirds for amputees (48.4% → 17.5%), and unintended movement during rest falls by half and by two-thirds (17.5% → 8.7%, 46.6% → 16.0%). The cost is a more hesitant hand: fewer correct decisions, and a slower start -- about 330ms more for intact subjects and 750ms for amputees (800ms and 1.3s from the labeled onset). The rejection threshold is a per-wearer setting, and the `replay` table is the trade-off curve a clinician would tune it on.

**The amputees.** Per-subject accuracy runs from 80% (subject 11, after four repetitions) down to 18-20% (subject 7) -- not far above chance for seven classes. Ninapro's records explain subject 7: no forearm is left below the elbow (0% remaining), so the muscles that moved the hand and wrist aren't under the electrodes at all. For the others, how much forearm remains doesn't predict much: subject 5, with 90% remaining, is among the weakest, and subject 3, with 30%, among the strongest. A real fitting would also train the wearer, with feedback, to make each phantom movement more consistent. Ninapro's subjects had none of that.

## Running it on the device

`emg_features.[ch]`, `intent_decoder.[ch]` and `../../nn.[ch]` are the firmware. `prepare_data.c`, `train.c`, `test.c` and `replay.c` call the same functions on the desktop; `ninapro.[ch]` and `mat_reader.[ch]` stand in for the electrodes. `make stm32` compiles the firmware half for the Cortex-M33 (`-mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard -Os`):

```
   text	   data	    bss	    dec	    hex	filename
    488	     16	  12800	  13304	   33f8	stm32/emg_features.o
   1339	     28	      0	   1367	    557	stm32/intent_decoder.o
  24227	      0	      0	  24227	   5ea3	stm32/nn.o
  26054	     44	  12800	  38898	   97f2	(TOTALS)
```

Unlike the other examples, this firmware trains: fine-tuning needs `nn_train()`, so linking with `--gc-sections` drops less of `nn.c` (mostly model file I/O).

**Model.** The shared model goes to flash in float (`../../export model.bin model.inplace --inplace`, 17.8KB). At calibration, `nn_load_model_inplace_copy()` copies it into RAM, where fine-tuning can change it. The fine-tuned model can stay in float -- inference is 4,423 multiply-adds every 50ms -- or be quantized in place with `nn_quantize()`. Quantization cost nothing measurable here: 68.40% → 68.32% (intact) and 44.37% → 44.38% (amputee), 1 rep.

**RAM.** The calibration data is the largest buffer: one repetition is at most 810 feature vectors in this data (104KB). The model in RAM is 18KB, plus an 18KB gradient buffer and, while fine-tuning runs, 35KB of Adam state (freed when it returns). `emg_features_t` is 12.8KB (the 200ms ring of samples), plus 12.8KB of static scratch. That's about 200KB of the U585's 786KB during calibration, and about 60KB once the calibration buffer is released.

**Calibration time.** Fine-tuning on one repetition is about 8,000 training steps, around half a billion floating-point operations. By an operation count, that's 10-20 seconds at 160MHz, during which the hand stays still -- not measured on hardware.

**Firmware loop.** On the device, `replay.c`'s calibration and control loop becomes:

```c
static intent_decoder_t id;
static float cal_features[MAX_CALIBRATION][EF_FEATURES]; // ~100KB
static int cal_labels[MAX_CALIBRATION];
static float scratch[EF_FEATURES + INTENT_COUNT];
nn_t *nn = nn_load_model_inplace_copy(shared_model, shared_model_len); // RAM copy, so it can be fine-tuned
intent_decoder_init(&id);

// Calibration: prompt each movement in turn, keep the windows of the steady hold.
int n = 0;
for (int k = 0; k < INTENT_COUNT; k++) {
  prompt_wearer(k);                            // "Now open your hand and hold it..."
  while (prompt_active()) {
    float sample[EF_CHANNELS], features[EF_FEATURES];
    read_ads1298(sample);                      // DRDY interrupt, 2kHz, SPI DMA
    if (emg_features_add_sample(&id.ef, sample, features) && in_steady_hold() && n < MAX_CALIBRATION) {
      memcpy(cal_features[n], features, sizeof(features));
      cal_labels[n++] = k;
    }
  }
}
intent_decoder_calibrate(&id, cal_features[0], cal_labels, n);
intent_decoder_fine_tune(&id, nn, cal_features[0], cal_labels, n, ID_FINE_TUNE_EPOCHS, ID_FINE_TUNE_RATE, scratch);
save_calibration_to_flash(&id.cal);            // so an unchanged fit survives a reset

// Control.
for (;;) {
  float sample[EF_CHANNELS];
  read_ads1298(sample);
  id_report_t report;
  if (intent_decoder_add_sample(&id, nn, sample, &report))
    drive_motors(report.intent, report.speed); // every 50ms
}
```

The functions not defined in this repository are the board-specific parts: the ADS1298 driver, the prompting user interface (an app, or tones and vibration in the socket), and the motor drivers. `in_steady_hold()` drops the first and last half-second of each prompt, so calibration doesn't learn the transitions as the movement.

This example hasn't been run on hardware; the sizes above come from the cross-compiler.

## Limitations

- **Not a medical device.** This is a demonstration of the signal processing and machine learning, measured offline on recorded data. Controlling a real prosthesis takes clinical validation, safety interlocks (a hand that can crush must not close on a decoding error), and a regulatory pathway.
- **Offline accuracy isn't usability.** The numbers above score decisions against labels. How well someone can actually use a hand is measured with the wearer in the loop -- task completion, time and effort -- and users adapt to a decoder as much as it adapts to them. Offline accuracy and usability are known to correlate only loosely (Scheme & Englehart, 2011, "Electromyogram pattern recognition for control of powered upper-limb prostheses: state of the art and challenges for clinical use", *Journal of Rehabilitation Research & Development*).
- **One session per person.** Calibration and testing come from the same sitting. Over a day of wear, electrodes shift, skin sweats and muscles tire, and accuracy falls. Recalibrating at each donning, as assumed here, covers some of that; the rest is why research systems add adaptation during use.
- **Ninapro's subjects are not prosthesis users.** They copied on-screen movements without practice or feedback, and the amputees had no myoelectric training. Experienced users of pattern-recognition prostheses produce more consistent patterns.
- **Six movements and rest.** Real hands add grips (key grip, pinch, point), and a classifier only offers one movement at a time. Ninapro's other exercises include grasps, which would be a natural next step, as would simultaneous control (regressing each degree of freedom separately rather than classifying one at a time).
