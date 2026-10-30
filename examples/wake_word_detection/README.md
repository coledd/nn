<p align="center">
  <img src="figures/banner.svg" alt="Wake Word Detection: listening for one spoken word in a second of audio, with log-mel features and a small CNN." width="100%">
</p>

> **Abstract.** A voice-controlled device has to notice when it's being addressed without streaming its microphone anywhere or keeping a power-hungry speech recognizer awake. This example builds that always-on listener: it turns each second of 16 kHz audio into 49 frames of 20 log-mel energies, normalized per clip, and a 14,818-parameter convolutional network (CNN) reads the whole second as a 49 × 20 image and decides whether it was the wake word, "marvin". It trains and tests on real recordings from Google's Speech Commands dataset, split by speaker with the dataset's own lists, so every test clip is a voice the network never heard. Over five training runs, with the decision threshold set on validation data for at most 1% false accepts, it missed 4.1–6.2% of 195 test "marvin" clips and was falsely triggered by 0.6–1.0% of 10,853 other clips. On a continuous 5.7-hour stream of the same test speech, scanned every 100 ms the way a listener runs, it caught 86–91% of the wake words at 6–8 false wake-ups per hour (four of five runs; the fifth never got down to 10 per hour on the validation stream). That took training on one-second windows cut from runs of words laid end to end; trained on whole clips alone, the same network woke falsely 17 to 55 times an hour even at its strictest threshold. A 5,121-parameter GRU that reads the clip one frame at a time, trained with backpropagation through all 49 frames, misses 6–10% at the 1% false-accept point. Quantized to int8 the CNN takes 21 KB of flash and about 78 KiB of RAM, against the GRU's 2 KiB.

| | |
|---|---|
| **Task** | Binary classification of a 1-second audio clip: the wake word "marvin", or anything else |
| **Data** | Google Speech Commands v0.02 (CC BY 4.0), downloaded by the Makefile, split by speaker: training on 1,710 "marvin" clips (plus time-shifted copies), 9,274 other-word and background-noise clips, and 20,520 one-second windows cut from runs of training words; testing on 195 "marvin" and 10,853 others from unseen speakers, as clips and as a continuous stream |
| **Model** | 49 × 20 log-mel image → 3 × (3 × 3 conv + 2 × 2 max pool) → softmax 2; 14,818 parameters. Baseline: GRU 32 → sigmoid, 5,121 parameters, backpropagated through all 49 frames |
| **Hardware** | An always-on microcontroller with a MEMS microphone, in front of a larger system it wakes; 21 KB of flash and ~78 KiB of RAM for the int8 CNN |
| **Result** | Per clip, at a threshold set on validation: 4.1–6.2% false rejects at 0.6–1.0% false accepts (5 runs). On a 5.7-hour test stream at 20 dB SNR: 86–91% of wake words caught at 6–8 false wake-ups per hour (4 of 5 runs) |

## 1. The problem

A doorbell, a smart speaker or a hearing aid that responds to its name needs something listening all the time. Streaming raw audio to a server to find out whether anyone spoke to it costs bandwidth, power and privacy; keeping a full speech recognizer awake costs power. The usual answer is a *wake-word detector*: a small model on a low-power processor whose only job is deciding, from a rolling one-second window of its own microphone, whether the one word it cares about was just spoken. Only then does it wake everything else.

Two pieces have to work together, and this example keeps them visibly separate:

1. **Feature extraction** (`audio_features.[ch]`, at the repository root) turns a window of raw samples into a short sequence of log-mel energy frames — the same transform full speech recognizers start with. It runs identically on the device, against the live microphone buffer.
2. **The classifier** (this directory's `train_cnn.c`) is a small CNN that reads that whole sequence at once and outputs a wake-word probability. `train_gru.c` trains a recurrent network on the same features as a baseline.

## 2. The device

<p align="center">
  <img src="figures/device.svg" alt="A smart speaker shown cut away: a MEMS microphone at 16 kHz feeds an always-on microcontroller running log-mel features and a CNN, which wakes a sleeping main processor only for the wake word. Beside it, what the device hears: 'Marvin', other words such as 'house', 'go' and 'stop', and background noise like a running tap." width="900">
</p>
<p align="center"><sub><b>Figure 1.</b> Where a wake-word detector sits: between the microphone and everything that costs power. Schematic, not to scale.</sub></p>

`predict.c` shows the split in code: except for loading a WAV file (`wav_reader.[ch]`, which a device doesn't need — it already has samples from its microphone driver), everything it does is what the firmware does.

## 3. The data

The [Speech Commands dataset](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) [1] is tens of thousands of one-second clips of single spoken words from thousands of speakers, recorded on their own devices, plus several minutes of background noise. The positive class is "marvin", one of two words the dataset's authors included to stand in for a wake word. The negative class is every other word, plus one-second stretches of the background-noise recordings, so the model learns to stay quiet through both other speech and ambient noise.

**Split by speaker.** `prepare_data.c` writes `train.csv`, `validation.csv` and `test.csv` directly, following the dataset's own `validation_list.txt` and `testing_list.txt`. Those lists put all of one speaker's clips in the same split, so the test set measures the model on voices it never trained on, which is what a product shipped to new users faces. The background-noise recordings aren't on the lists; each one is split by position, its first 80% to training and the next two 10% stretches to validation and test.

**Training: balanced, and five times larger than it needs to be.** A wake-word detector's hardest job is staying quiet through everything else, so training uses five times as many distinct negatives as a balanced set would, and evens the classes by writing each "marvin" clip once as recorded and four more times shifted by up to ±100 ms. The words in Speech Commands sit wherever the speaker said them within the second, and a live microphone window won't be aligned either. The shift is seeded, so `train.csv` comes out the same on every run.

**Training: windows from a stream.** Every clip above is one word alone in its second. A listener's window, sliding along real audio, mostly isn't: it catches the end of one word and the start of the next, or half a word over background. `prepare_data.c` also writes 20,520 windows that look like that, all from training speakers. Each comes from three training clips laid end to end with gaps of 250–1500 ms, over the training stretch of the background recordings at a random 5–60 dB signal-to-noise ratio, from noisy to nearly silent. A positive window has a "marvin" clip in the middle and is placed anywhere that still holds all of the spoken word (found from the clip's energy); there are two per training "marvin". A negative window has three other words and is placed anywhere at all, five times as many. These windows are what cut false wake-ups on a stream tenfold (section 5.3).

**Validation and test: everything.** Both get every other-word clip their speakers recorded, about 50 negatives per wake-word clip. A false-accept rate measured on a couple of hundred negatives moves by half a point per clip; measured on 10,000, it can be trusted to a tenth of a point.

| Split | "marvin" | Other words | Background noise | Rows |
|---|---|---|---|---|
| Training | 1,710 clips × 5 = 8,550, plus 3,420 stream windows | 7,718, plus 17,100 stream windows | 1,556 overlapping windows | 38,344 |
| Validation | 195 | 9,786 | 39 | 10,020 |
| Test | 195 | 10,810 | 43 | 11,048 |

<p align="center">
  <img src="figures/audio.svg" alt="Waveforms and log-mel feature images for three one-second clips: the word 'marvin' with two voiced syllables, the word 'house' from the same speaker with one burst and a hissing ending, and background noise from a running tap that fills all bands evenly." width="900">
</p>
<p align="center"><sub><b>Figure 2.</b> Three clips as the network sees them: waveform (top) and the 49 × 20 normalized log-mel features (bottom, darker = more energy). "marvin" and "house" are the same speaker. Normalizing each clip to zero mean and unit variance removes loudness, so even quiet noise fills the scale.</sub></p>

## 4. Method

### 4.1 Features

`audio_features_extract_sequence()` computes, for each one-second clip:

| Step | Setting |
|---|---|
| Frames | 512 samples (32 ms), every 320 samples (20 ms): 49 frames per clip |
| Spectrum | Hann window, 512-point FFT, power |
| Mel filterbank | 20 triangular bands evenly spaced on the mel scale, 0-8 kHz |
| Compression | natural log of each band's energy |
| Normalization | all 980 values of the clip shifted and scaled to mean 0, variance 1 |

Per-clip normalization matters: real recordings vary enormously in loudness and microphone response, and normalizing removes that before the network sees anything.

### 4.2 `train_cnn.c`: the clip as an image

<p align="center">
  <img src="figures/architecture.svg" alt="Pipeline: microphone at 16 kHz, frames of 512 samples every 320, Hann window, FFT and 20 log-mel bands, per-clip normalization; then the whole clip as a 49 by 20 image into three stages of 3 by 3 convolution with GELU and 2 by 2 max pooling, with 16, 32 and 32 filters, down to 6 by 2 by 32; a two-way softmax output; and a decision: wake word if its probability is at least the threshold." width="900">
</p>
<p align="center"><sub><b>Figure 3.</b> The pipeline. The CNN reads all 49 frames at once, time down and mel bands across, and decides in one call.</sub></p>

The CSV row `prepare_data.c` writes is frame after frame, which is exactly a 49 × 20 image in row-major order. Convolution learns small time-frequency shapes — the onset of a formant, the band pattern of a voiced syllable — wherever in the second they occur, the way most small keyword-spotting models work [3].

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 49 × 20 × 1 | the whole clip: 49 frames (time) × 20 mel bands |
| 1 | CNN | 49 × 20 × 16 | 3 × 3 kernel, 16 filters, "same" padding, GELU |
| 2 | Pool | 24 × 10 × 16 | 2 × 2 max pool, stride 2 |
| 3 | CNN | 24 × 10 × 32 | 3 × 3 kernel, 32 filters, "same" padding, GELU |
| 4 | Pool | 12 × 5 × 32 | 2 × 2 max pool, stride 2 |
| 5 | CNN | 12 × 5 × 32 | 3 × 3 kernel, 32 filters, "same" padding, GELU |
| 6 | Pool | 6 × 2 × 32 | 2 × 2 max pool, stride 2: each value now spans about a quarter of a second |
| 7 | Dropout | 384 | 20%, training only |
| 8 | Output | 2 | Softmax (cross-entropy loss): other, wake word |

14,818 trainable parameters, 14,048 of them in the three convolutions. About 1.8 million multiply-adds per clip, 61% of them in the second convolution.

### 4.3 `train_gru.c`: the recurrent baseline

The baseline reads the same features one 20-value frame per call into a 32-unit GRU [2], whose hidden state carries what it has heard across the clip, and a sigmoid output gives the wake-word probability after the last frame. It has 5,121 parameters and needs far less memory and arithmetic than the CNN (section 6). It is trained the way `examples/gesture_recognition` trains its RNN, every frame of a clip against the clip's label, with plain SGD at 0.005, multiplied by 0.97 each epoch. Each frame's error is backpropagated through every earlier frame of its clip (`nn_set_bptt_depth(nn, 49)`), so the GRU learns directly to carry evidence from the first syllable to the end of the word. With the library's default of one frame, five runs missed 8–17% of test wake words at threshold 0.5 while triggering on 4–13% of other clips. Through all 49 frames, on the same whole-clip training data, that fell to 5.1% and 2.8% (medians), for training that takes minutes instead of seconds.

### 4.4 Training and testing

`train_cnn` uses plain SGD at a learning rate of 0.002, multiplied by 0.95 each epoch, and stops once validation error hasn't improved for 5 epochs, keeping the best model. Because validation holds about 50 negatives per wake-word clip, both trainers average its error *per class* — the mean over "marvin" clips and the mean over everything else, weighted equally. A plain average would be almost entirely about false accepts, and early stopping would favor a model that rarely fires at all.

`test` scores a model of either kind: the false-reject rate (wake-word clips it missed) and false-accept rate (other clips it fired on) at a chosen threshold, then both rates across thresholds from 0.05 to 0.95. `stream` scores it the way a listener runs instead (section 5.3). Weight initialization and shuffling aren't seeded, so every training run differs.

## 5. Results

Five runs of each trainer, each tested on the same 11,048 held-out clips from speakers none of the models trained on, and then on a continuous stream made from them. Overall accuracy isn't a useful number here — 98% of test clips aren't the wake word, so accuracy is roughly 100% minus the false-accept rate — so the two error rates are reported separately.

### 5.1 At a threshold of 0.5

<p align="center">
  <img src="figures/runs.svg" alt="Dot plot of five CNN and five GRU runs at threshold 0.5 on 11,048 test clips. False-reject rate: CNN 2.6 to 5.6 percent, median 3.6; GRU 4.1 to 8.2 percent, median 4.6. False-accept rate: CNN 1.0 to 3.1 percent, median 1.1; GRU 1.1 to 2.4 percent, median 1.9." width="760">
</p>
<p align="center"><sub><b>Figure 4.</b> Five training runs of each network at the default threshold. Each dot is one run; the bar is the median.</sub></p>

| | CNN (14,818 parameters) | GRU (5,121 parameters) |
|---|---|---|
| False-reject rate (missed "marvin"), median (range) | **3.6%** (2.6–5.6) | 4.6% (4.1–8.2) |
| False-accept rate (other audio taken for it), median (range) | **1.1%** (1.0–3.1) | 1.9% (1.1–2.4) |

The CNN is ahead on both, though not by much; the GRU gets close with a third of the parameters and 2 KiB of RAM against the CNN's 78 KiB (section 6). Two training changes got both networks here. The GRU needs backpropagation through the whole clip: trained one frame at a time, the library's default, it missed 11.8% and false-triggered on 9.4% (medians of five runs). And the stream windows of section 3, added for the sake of section 5.3, also helped on clips: trained on whole clips alone, the CNN missed 4.6% at 2.8% false accepts.

### 5.2 Choosing the threshold

<p align="center">
  <img src="figures/tradeoff.svg" alt="False-reject rate against false-accept rate, on a log scale, as the threshold moves from 0.05 to 0.95, for one CNN run and one GRU run. The CNN curve runs from 0.5 percent missed at 5.8 percent false accepts to 8.7 percent missed at 0.15 percent; the GRU curve lies just above it, from 1 percent missed at 7.6 percent false accepts to 11 percent missed at 0.16 percent. Five black diamonds mark the five CNN runs at their validation-chosen thresholds, at 4 to 6 percent missed and 0.6 to 1.0 percent false accepts." width="760">
</p>
<p align="center"><sub><b>Figure 5.</b> The trade-off every wake-word detector makes. Raising the threshold moves each curve up and to the left: fewer false accepts, more missed wake words. The open circles mark 0.5; the diamonds are the five CNN runs at thresholds chosen on validation data.</sub></p>

The threshold decides where on that curve a product sits, and the right point depends on the product: a doorbell can afford a few false wake-ups, a device that starts recording when it wakes can't. With 10,000 validation clips, the threshold can be chosen on validation data and trusted to carry over. For each CNN run, the lowest threshold giving at most 1% false accepts on validation was applied unchanged to the test set:

| Run | Threshold (chosen on validation) | False-reject rate | False-accept rate |
|---|---|---|---|
| 1 | 0.75 | 5.1% | 0.92% |
| 2 | 0.50 | 5.6% | 0.96% |
| 3 | 0.90 | 6.2% | 0.57% |
| 4 | 0.60 | 4.1% | 0.81% |
| 5 | 0.55 | 5.1% | 0.84% |

All five land within about two points of each other: **about 5% of wake words missed at under 1% false accepts**, on voices the network never heard. Trained on whole clips alone, the same network missed 7.2–8.2% at this operating point. The thresholds themselves vary from run to run (0.50–0.90), which is why each model needs its own, set on validation rather than fixed in code. The GRU's five runs all reach 1% on validation too, at thresholds of 0.70–0.90, and miss 6.2–9.7% of the test wake words, median 8.7%.

`predict` shows the same decision on single files. With run 4's model, a "marvin" clip scores 1.000 and a "house" clip from the same speaker 0.00003.

### 5.3 On a continuous stream

A clip test can't say how often a device wakes up per hour, because a device never sees clips. `stream.c` builds what it does see. It lays all 11,005 test clips (195 of them "marvin") end to end in a shuffled order, with 250–1500 ms gaps, over the test stretch of the background recordings looped underneath, 5.7 hours in all. Then it slides a one-second window along it every 100 ms, computing each window's features exactly as for a clip, normalization included. A window at or above the threshold wakes the detector, which then ignores the next second, so one utterance wakes it once. A wake-up whose window overlaps a "marvin" detects it; any other wake-up is false.

<p align="center">
  <img src="figures/stream.svg" alt="Missed wake words against false wake-ups per hour, on a log scale, on the 5.7-hour test stream at 20 dB signal-to-noise ratio, as the threshold moves from 0.05 to 0.995. The CNN trained on whole clips only, dashed, never gets below 25 false wake-ups per hour, where it misses 16 percent. The CNN trained with streaming windows reaches 3.7 per hour at 11 percent missed and 6.4 per hour at 9 percent. The GRU with streaming windows reaches lower false-wake-up rates only by missing far more, 26 percent at 1 per hour. Four black diamonds mark four CNN runs at thresholds chosen on a validation stream, at 6 to 8 false wake-ups per hour and 9 to 14 percent missed." width="760">
</p>
<p align="center"><sub><b>Figure 6.</b> The same trade-off on a stream: missed wake words against false wake-ups per hour. Each curve is one run; the diamonds are four CNN runs at thresholds chosen on a validation stream.</sub></p>

The threshold is chosen the same way as for clips, but on a stream built from the validation speakers and the validation background: the lowest threshold at which that stream gives at most 10 false wake-ups per hour. Applied unchanged to the test stream, at 20 dB signal-to-noise ratio:

| | CNN, 5 runs | GRU, 5 runs | CNN trained on whole clips only, 5 runs |
|---|---|---|---|
| Runs reaching ≤10 false wake-ups/h on validation | 4 | 5 | 0 (none at any threshold up to 0.995) |
| Thresholds chosen | 0.97–0.99 | 0.97–0.995 | — |
| Wake words caught, median (range) | **89.8%** (86.2–90.8) | 79.5% (74.9–84.1) | at 0.995: 81–88% |
| False wake-ups per hour, median (range) | **7.2** (6.4–8.3) | 4.8 (4.2–7.6) | at 0.995: 17–55 |

With the background at 60 dB, nearly silent between words, the CNN's four runs catch 82–91% at 6–12 false wake-ups per hour; the clip-trained CNN wakes falsely 73–203 times an hour there even at 0.995. Near-silence is the harder case for a clip-trained model, probably because per-clip normalization stretches whatever is in a quiet window, the tail of a word or a breath, to the same scale as a word.

So a stream test finds what a clip test can't. The clip-trained CNN looked almost as good per clip (7.7% missed at 1% false accepts), and on a stream it was unusable: run 4, at the threshold chosen for it per clip (0.70), caught 99.5% of the wake words but woke falsely 513 times an hour. Training on stream-like windows fixed most of that, but 6–8 false wake-ups an hour is still more than a product wants (section 7).

**An earlier version of this example** used the GRU alone, split the data randomly by clip, and tested on 441 clips, about half of them "marvin". It reported 85% accuracy, with false-reject and false-accept rates of 16% and 14%. A random split puts the same speakers in training and test, so that figure was optimistic; on the speaker split, that same GRU and data scored 82–84%.

## 6. Running on the device

On the device, `audio_features.[ch]` and `../../fft.[ch]` run against the microphone buffer and `nn.[ch]` runs the network; `predict.c` is the same sequence of calls on a WAV file, and `wake_word_probability()` in `wake_word_data.h` runs either model on one window in a single call. A real listener slides the one-second window along the stream rather than waiting for separate clips, compares the probability against its own threshold, and holds off for a moment after waking, as `stream.c` does (section 5.3). At one window every 100 ms, the CNN costs about 18 million multiply-adds per second.

| Resource | CNN | GRU |
|---|---|---|
| **Flash**, int8, `export --inplace` | **21,300 bytes** | 5,572 bytes |
| **RAM**, model loaded with `nn_load_model_inplace()` | **~78 KiB** | ~1.7 KiB |
| **Compute** per one-second window | ~1.8 million multiply-adds | ~250,000 multiply-adds, over 49 calls |
| Accuracy cost of int8 (CNN run 4, GRU run 3; threshold 0.5) | none: 3.6% / 1.08% → 3.6% / 1.08% | 4.6% / 1.52% → 5.1% / 1.47% |

Flash and RAM were measured on the quantized, exported models. To build one:

```
../../quantize wake_word_cnn.txt wake_word_cnn_q.txt
../../export wake_word_cnn_q.txt wake_word_cnn.inplace --inplace
./test wake_word_cnn.inplace
```

**RAM is the CNN's larger cost.** A model loaded with `nn_load_model_inplace()` keeps its weights in flash, and its activations in two buffers that every layer shares, each as wide as the widest layer it holds. The first convolution's 49 × 20 × 16 output is four-fifths of the CNN's 78 KiB. That fits a microcontroller with 128 KiB of RAM alongside the audio buffers, though not the smallest always-on parts. Two smaller layouts were measured, five runs each, at the same 1% false-accept operating point. These were trained on the earlier whole-clip data, before the stream windows of section 3, so this example's CNN is shown on that data too:

| Layout | RAM | Flash | False-reject rate at ~1% false accepts, median (range) |
|---|---|---|---|
| This example's CNN, whole-clip data | 78 KiB | 21 KB | **7.7%** (7.2–8.2) |
| Stride-2 first convolution, 8 → 16 → 32 filters | 25 KiB | 9.5 KB | 13.3% (8.7–14.9) |
| Stride-2 first convolution, then pooling, 16 → 32 → 32 filters | 21 KiB | 21 KB | 13.3% (7.7–16.4) |

Both cut RAM by two-thirds or more, and both nearly double the missed wake words. Before the library shared activation buffers across layers, the same three layouts needed 251, 77 and 65 KiB.

## 7. Limitations

- **A stream of single words, not a room.** The stream in section 5.3 is isolated words with gaps, over 43 seconds of test background looped for 5.7 hours. Real room audio is continuous conversation, television and music, with words running together, and products typically aim for well under one false wake-up per hour of it; this test can't measure that, and 6–8 per hour on single words is still far above it. More stream-like training data (real conversation as negatives) and a second check before waking the main processor are the usual next steps.
- **Triggering on part of the word.** The detection rule credits any wake-up whose window overlaps a "marvin", including one that fires on "mar-" before the word ends. A product that has to hear the whole word needs a stricter rule.
- **The threshold isn't stored in the model.** `test` and `predict` default to 0.5; the operating point in section 5.2 is chosen per model on `validation.csv` and has to be carried to the firmware separately.
- **Simple augmentation.** Training copies of the wake word are shifted in time, and the stream windows mix in background noise; volume and pitch aren't varied, and the background comes from just six recordings.
- **Small model, small data.** 1,710 training examples of the wake word; production wake-word models train on far more, from far more speakers.

## Appendix A. Build and run

```
cd examples/wake_word_detection
make
```

The first `make` downloads and extracts the ~2.4 GB Speech Commands archive, builds `prepare_data` and runs it (about a minute and a half) to write `train.csv`, `validation.csv` and `test.csv` (~345, ~90 and ~100 MB), and builds `train_cnn`, `train_gru`, `test`, `predict` and `stream`. `make clean` keeps the dataset; `make distclean` removes it.

| Command | What it does |
|---|---|
| `./train_cnn wake_word_cnn.txt` | trains the CNN (27–36 minutes for 8–12 epochs, single-threaded, measured with ten trainings sharing one machine; resumes if the file exists) |
| `./train_gru wake_word_gru.txt` | trains the GRU baseline (12–15 minutes for 9–14 epochs, measured the same way); use a different model file than the CNN's |
| `./test wake_word_cnn.txt validation.csv` | false-reject and false-accept rates across thresholds on the validation set: choose the threshold here |
| `./test wake_word_cnn.txt test.csv 0.7` | both rates on the test set at the chosen threshold (either model; the defaults are `test.csv` and 0.5) |
| `./predict wake_word_cnn.txt dataset/marvin/<file>.wav` | the wake-word probability of one WAV file, the way an application would get it |
| `./stream dataset validation 20 wake_word_cnn.txt` | detection rate and false wake-ups per hour on the validation stream at 20 dB SNR, across thresholds: choose a streaming threshold here (about 5 minutes per CNN; several models can share one run) |
| `./stream dataset test 20 wake_word_cnn.txt` | the same on the test stream (`SEED=n` in the environment shuffles it differently) |
| `./prepare_data dataset sheila` | rebuilds all three CSVs for a different wake word; then train a new model |

The figures in this document are generated by `tools/docs/wake_word_detection.py`.

## Appendix B. Program output

`./train_cnn wake_word_cnn.txt` (run 4 of the five in section 5; the validation error is the per-class average of section 4.4):

```
Creating new model.
train error, validation error, learning rate
0.22159, 0.21565, 0.00200
0.08983, 0.13623, 0.00190
0.06312, 0.11839, 0.00181
0.05204, 0.16175, 0.00171
0.04620, 0.14698, 0.00163
0.03989, 0.13160, 0.00155
0.03414, 0.09342, 0.00147
0.03122, 0.13752, 0.00140
0.02764, 0.11830, 0.00133
0.02490, 0.13150, 0.00126
0.02383, 0.13795, 0.00120
0.02071, 0.12920, 0.00114
No validation improvement for 5 epochs (best: 0.09342) -- stopping early.
Final (last epoch) train error: 0.020711, validation error: 0.129195
Best validation error (the model saved to disk): 0.093424
Training epochs: 12
```

`./test wake_word_cnn.txt validation.csv`, the rows around 1% false accepts (the lowest threshold at or under it is 0.60):

```
threshold, false-reject %, false-accept %
...
0.50, 5.6, 1.26
0.55, 5.6, 1.14
0.60, 5.6, 0.96
...
```

`./test wake_word_cnn.txt test.csv 0.6`:

```
test.csv: 11048 clips (195 wake word, 10853 other), threshold 0.60
Accuracy: 10952/11048 = 99.13%
False-reject rate (missed wake word): 8/195 = 4.1%
False-accept rate (wrongly triggered): 88/10853 = 0.81%

threshold, false-reject %, false-accept %
0.05, 0.5, 5.84
0.10, 1.0, 4.13
0.15, 1.0, 3.34
0.20, 2.6, 2.79
0.25, 2.6, 2.33
0.30, 2.6, 1.95
0.35, 3.6, 1.72
0.40, 3.6, 1.47
0.45, 3.6, 1.29
0.50, 3.6, 1.08
0.55, 4.1, 0.96
0.60, 4.1, 0.81
0.65, 4.1, 0.66
0.70, 4.6, 0.63
0.75, 4.6, 0.53
0.80, 4.6, 0.45
0.85, 5.6, 0.33
0.90, 6.7, 0.21
0.95, 8.7, 0.15
```

`./stream dataset validation 20 wake_word_cnn.txt`, the top rows (the lowest threshold at or under 10 false wake-ups per hour is 0.99):

```
threshold, detected %, missed %, false wake-ups, per hour
...
0.970, 93.3, 6.7, 113, 22.01
0.980, 92.3, 7.7, 80, 15.58
0.990, 91.3, 8.7, 50, 9.74
0.995, 90.8, 9.2, 34, 6.62
```

`./stream dataset test 20 wake_word_cnn.txt`, selected rows:

```
Stream: 5.66 hours, 11005 test clips (195 "marvin"), gaps 250-1500 ms, background at 20 dB SNR (43.0 s of test background, looped), seed 1
Window: 1 s every 100 ms, 1000 ms hold-off after each trigger

wake_word_cnn.txt
threshold, detected %, missed %, false wake-ups, per hour
...
0.500, 97.4, 2.6, 1552, 274.15
...
0.950, 94.4, 5.6, 124, 21.90
0.970, 93.3, 6.7, 84, 14.84
0.980, 93.3, 6.7, 64, 11.31
0.990, 90.8, 9.2, 36, 6.36
0.995, 89.2, 10.8, 21, 3.71
```

`./predict wake_word_cnn.txt` on a "marvin" clip and a "house" clip from the same speaker:

```
$ ./predict wake_word_cnn.txt dataset/marvin/00b01445_nohash_0.wav
Wake-word probability: 1.00000
$ ./predict wake_word_cnn.txt dataset/house/00b01445_nohash_0.wav
Wake-word probability: 0.00003
```

## References

1. P. Warden, "[Speech Commands: a dataset for limited-vocabulary speech recognition](https://arxiv.org/abs/1804.03209)," arXiv:1804.03209, 2018.
2. K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," *EMNLP*, 2014 (the GRU).
3. T. N. Sainath and C. Parada, "Convolutional neural networks for small-footprint keyword spotting," *Interspeech*, 2015.
