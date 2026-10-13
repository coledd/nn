<p align="center">
  <img src="figures/banner.svg" alt="Wake Word Detection: listening for one spoken word in a second of audio, with log-mel features and a 32-unit GRU." width="100%">
</p>

> **Abstract.** A voice-controlled device has to notice when it's being addressed without streaming its microphone anywhere or keeping a power-hungry speech recognizer awake. This example builds that always-on listener: it turns each second of 16 kHz audio into 49 frames of 20 log-mel energies, normalized per clip, and feeds them one frame at a time to a 32-unit GRU that decides whether the clip is the wake word, "marvin". It trains and tests on real recordings from Google's Speech Commands dataset. Over seven training runs, the network classified 78-89% of 441 held-out clips correctly (median 85%), missing the wake word in a median 16% of "marvin" clips and taking other audio for it in 14% of the rest. The split is by clip, not by speaker, so these numbers are likely optimistic for new voices.

| | |
|---|---|
| **Task** | Binary classification of a 1-second audio clip: the wake word "marvin", or anything else |
| **Data** | Google Speech Commands v0.02 (CC BY 4.0): 2,100 "marvin" clips and 2,302 others (other words and background noise), downloaded by the Makefile |
| **Model** | 49 × 20 log-mel frames → GRU 32 → sigmoid; 5,121 parameters |
| **Hardware** | An always-on microcontroller with a MEMS microphone, in front of a larger system it wakes |
| **Result** | Test accuracy 78-89% over 7 runs (median 85.3%); false-reject median 15.7%, false-accept median 14.3% |

## 1. The problem

A doorbell, a smart speaker or a hearing aid that responds to its name needs something listening all the time. Streaming raw audio to a server to find out whether anyone spoke to it costs bandwidth, power and privacy; keeping a full speech recognizer awake costs power. The usual answer is a *wake-word detector*: a small model on a low-power processor whose only job is deciding, from a rolling one-second window of its own microphone, whether the one word it cares about was just spoken. Only then does it wake everything else.

Two pieces have to work together, and this example keeps them visibly separate:

1. **Feature extraction** (`audio_features.[ch]`, at the repository root) turns a window of raw samples into a short sequence of log-mel energy frames — the same transform full speech recognizers start with. It runs identically on the device, against the live microphone buffer.
2. **The classifier** (this directory's `train.c`) is a small GRU network that reads that sequence one frame at a time and outputs a wake-word probability.

## 2. The device

<p align="center">
  <img src="figures/device.svg" alt="A smart speaker shown cut away: a MEMS microphone at 16 kHz feeds an always-on microcontroller running log-mel features and a GRU, which wakes a sleeping main processor only for the wake word. Beside it, what the device hears: 'Marvin', other words such as 'house', 'go' and 'stop', and background noise like a running tap." width="900">
</p>
<p align="center"><sub><b>Figure 1.</b> Where a wake-word detector sits: between the microphone and everything that costs power. Schematic, not to scale.</sub></p>

`predict.c` shows the split in code: except for loading a WAV file (`wav_reader.[ch]`, which a device doesn't need — it already has samples from its microphone driver), everything it does is what the firmware does.

## 3. The data

The [Speech Commands dataset](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) [1] is tens of thousands of one-second clips of single spoken words from thousands of speakers, recorded on their own devices, plus several minutes of background noise. The positive class is "marvin", one of two words the dataset's authors included to stand in for a wake word. The negative class mixes a balanced sample of the other spoken words with one-second chunks of the background-noise recordings, so the model learns to stay quiet through both other speech and ambient noise.

`prepare_data.c` turns every clip into one CSV row of features, through the same `audio_features.[ch]` code the device runs, and `../../split.py` splits the 4,402 rows 80/10/10: 3,521 for training, 440 for validation and 441 for testing. The wake word is a command-line argument, so training on a different word is a rerun, not a code change.

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

### 4.2 Network

<p align="center">
  <img src="figures/architecture.svg" alt="Pipeline: microphone at 16 kHz, frames of 512 samples every 320, Hann window, FFT and 20 log-mel bands, per-clip normalization; then each 20-value frame into a 32-unit GRU with a recurrent hidden state across the clip's 49 frames, a sigmoid output for wake-word probability, and a decision after the last frame: wake word if at least 0.5." width="900">
</p>
<p align="center"><sub><b>Figure 3.</b> The pipeline. The GRU reads one frame per call and carries its hidden state across the clip; the decision is made on its output after the last frame.</sub></p>

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 20 | one log-mel frame per call |
| 1 | GRU | 32 | reset and update gates; hidden state carried across the clip's 49 frames |
| 2 | Output | 1 | Sigmoid: wake-word probability |

5,121 trainable parameters, nearly all in the GRU. `nn_reset_state()` clears the hidden state before each clip. During training, every frame of a clip is trained against the clip's label; at test time the output after the 49th frame is compared with 0.5. Training uses plain SGD at a learning rate of 0.03, multiplied by 0.97 each epoch, and stops once validation error hasn't improved for 5 epochs.

## 5. Results

Weight initialization and shuffling aren't seeded, so every training run differs. Seven runs of `train` and `test`, each on the same 441 held-out clips (217 "marvin", 224 others):

<p align="center">
  <img src="figures/runs.svg" alt="Dot plot of seven runs: accuracy between 78 and 89 percent, median 85.3; false-reject rate between 6 and 26 percent, median 15.7; false-accept rate between 4 and 23 percent, median 14.3." width="760">
</p>
<p align="center"><sub><b>Figure 4.</b> Seven training runs, each tested on the 441 held-out clips. Each dot is one run; the bar is the median.</sub></p>

| | Median | Range over 7 runs |
|---|---|---|
| Accuracy | 85.3% | 78.2-89.1% |
| False-reject rate (missed "marvin") | 15.7% | 6.0-26.3% |
| False-accept rate (other audio taken for "marvin") | 14.3% | 3.6-22.8% |

Accuracy is fairly stable, but **where the errors fall varies a lot between runs**: one run missed 26% of the wake words while almost never triggering falsely (3.6%), another missed only 6% while triggering on 16% of other clips. Each run's 0.5 threshold lands at a different point on the trade-off between the two errors. A product would choose that point deliberately — a doorbell can afford a few false wake-ups, a device that records after waking can't — by setting the threshold on validation data rather than leaving it at 0.5.

`predict` shows the same decision on single files. With one of the trained models above, a "marvin" clip scores 0.989 and a "house" clip from the same speaker 0.008.

## 6. Running on the device

On the device, `audio_features.[ch]` and `../../fft.[ch]` run against the microphone buffer and `nn.[ch]` runs the GRU; `predict.c` is the same sequence of calls on a WAV file. Per second of audio, that is 49 512-point FFTs and 49 GRU steps of 32 units. The 5,121 weights are about 20 KB as 32-bit floats, and the repository's `quantize` and `export` tools convert a trained model to int8 and to the zero-copy inplace format for flash (see the top-level README). A real listener slides the one-second window along the stream rather than waiting for separate clips.

## 7. Limitations

- **The split is by clip, not by speaker.** `split.py` shuffles clips randomly, so test clips can come from speakers who also contributed training clips. Speech Commands publishes speaker-separated validation and test lists; using them would measure how the model does on new voices, which is what a product needs.
- **One-second, pre-cut clips.** Every clip is one word within a second of audio. A live microphone stream has words at arbitrary offsets, overlapping speech and long silences; streaming detection needs the window to slide and a rule for not triggering twice on one utterance.
- **One fixed threshold.** The 0.5 decision point isn't tuned; see section 5.
- **Small model, small data.** About 2,100 examples of the wake word; production wake-word models train on far more, from far more speakers.

## Appendix A. Build and run

```
cd examples/wake_word_detection
make
```

The first `make` downloads and extracts the ~2.4 GB Speech Commands archive, builds `prepare_data` and runs it to produce `samples.csv`, splits that into `train.csv`, `validation.csv` and `test.csv`, and builds `train`, `test` and `predict`. `make clean` keeps the dataset; `make distclean` removes it.

| Command | What it does |
|---|---|
| `./train wake_word_model.txt` | trains the GRU (seconds) |
| `./test wake_word_model.txt` | accuracy, false-reject and false-accept rates on `test.csv` |
| `./predict wake_word_model.txt dataset/marvin/<file>.wav` | the wake-word probability of one WAV file, the way an application would get it |
| `./prepare_data dataset samples.csv sheila` | rebuilds the features for a different wake word (then re-split with `python ../../split.py samples.csv --train 0.8 --validation 0.1` and train a new model) |

The figures in this document are generated by `tools/docs/wake_word_detection.py`.

## Appendix B. Program output

`./train wake_word_model.txt` (one of the seven runs in figure 4):

```
Creating new model.
train error, validation error, learning rate
0.05437, 0.09123, 0.03000
0.04301, 0.10277, 0.02910
0.04194, 0.16661, 0.02823
0.04199, 0.08007, 0.02738
...
0.03697, 0.11497, 0.02351
No validation improvement for 5 epochs (best: 0.08007) -- stopping early.
Final (last epoch) train error: 0.036972, validation error: 0.114969
Best validation error (the model saved to disk): 0.080068
Training epochs: 9
Validation accuracy: 291/440 = 66.14%
False rejects (missed wake word): 146/206 = 70.9%
False accepts (wrongly triggered): 3/234 = 1.3%
```

The closing validation figures are for the *last* epoch's weights, still in memory; the model saved to disk is the epoch with the best validation error, which `test` evaluates:

```
Test clips: 441 (217 wake word, 224 other), 49 frames each
Accuracy: 382/441 = 86.62%
False-reject rate (missed wake word): 34/217 = 15.7%
False-accept rate (wrongly triggered): 25/224 = 11.2%
```

`./predict wake_word_model.txt` on a "marvin" clip and a "house" clip from the same speaker:

```
$ ./predict wake_word_model.txt dataset/marvin/00b01445_nohash_0.wav
Wake-word probability: 0.98928
$ ./predict wake_word_model.txt dataset/house/00b01445_nohash_0.wav
Wake-word probability: 0.00754
```

## References

1. P. Warden, "[Speech Commands: a dataset for limited-vocabulary speech recognition](https://arxiv.org/abs/1804.03209)," arXiv:1804.03209, 2018.
2. K. Cho et al., "Learning phrase representations using RNN encoder-decoder for statistical machine translation," *EMNLP*, 2014 (the GRU).
