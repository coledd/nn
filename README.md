Neural Network library for embedded systems

Copyright (c) 2019-2026 Cole Design and Development, LLC

https://coledd.com

SPDX-License-Identifier: Apache-2.0

## Overview

This is a lightweight neural network library for use in microcontrollers and embedded systems.

The repository root holds the library itself and a set of general-purpose model tools (each works on a saved model regardless of which example produced it); everything under `examples/` is a self-contained example of constructing, training, and evaluating one particular kind of network, each with its own Makefile:

1. `nn.[ch]` - The neural net library, which can be pulled directly into your embedded project. Nothing else in this repository needs to accompany it into an embedded target.

2. `data_prep.[ch]` / `split.py` - General-purpose CSV data preparation: `data_prep.[ch]` reads/parses/shuffles flat input/target rows, and `split.py <input.csv> [--train FRACTION] [--validation FRACTION]` (default 0.8/0.1, remainder to test) shuffles and splits any line-oriented CSV file into train.csv/validation.csv/test.csv, with no idea what's in any given line. Unlike `gesture_data.[ch]`/`fall_data.[ch]` below (each specific to one example's own synthetic generator), neither has any domain-specific logic of its own, so they live here rather than inside `examples/character_recognition/` -- reusable by any example that wants CSV-backed training data. `data_prep.[ch]` isn't part of `libnn.a` (see the comment on its Makefile rule) -- both are desktop-side training-data utilities, not something an embedded target would want.

3. `audio_features.[ch]` - Turns raw PCM audio into the log-mel feature frames a sequence model (`LAYER_TYPE_RNN`/`GRU`/`LSTM`) can consume one timestep at a time: framing/windowing, an FFT, a mel filterbank, and log-energy output. Unlike everything else in this list besides `nn.[ch]` itself (and `fft.[ch]` below, which it's built on), this one is meant to ship into firmware too -- a keyword-spotting target has to run the exact same feature extraction against its live mic buffer that prepared its training data. Allocation-free after `audio_features_init()`, so it's safe to call from a real-time mic-input path. Not part of `libnn.a` (it isn't neural-net code); used by `examples/wake_word_detection/`.

4. `fft.[ch]` - The in-place radix-2 FFT (and inverse) shared by the on-device feature extractors: `audio_features.[ch]` above and `examples/predictive_maintenance/`'s `vibration_features.[ch]`, so the transform both depend on lives in one place. Ships into firmware with whichever of them a target uses; allocation-free, no dependency beyond `<math.h>`. Not part of `libnn.a` (it isn't neural-net code). On a Cortex-M with CMSIS-DSP, `arm_rfft_fast_f32()` is a faster drop-in replacement at the call sites.

5. `wav_reader.[ch]` - Decodes a 16-bit PCM mono WAV file into a float sample buffer. Same category as `data_prep.[ch]`: general enough to live here rather than inside one example, but desktop-only (not part of `libnn.a`) since firmware gets PCM straight from its own mic/ADC driver, never a `.wav` file. Used by `examples/wake_word_detection/`.

6. `stb_image.h` - Sean Barrett's single-header image decoder ([nothings/stb](https://github.com/nothings/stb), v2.30; public domain / MIT). Same category as `wav_reader.[ch]`: desktop-only (a camera target gets raw pixels from its own sensor driver, never a JPEG file), shared by `examples/image_classification/` and `examples/object_detection/` to decode their datasets' photos, and by `examples/prosthetic_hand/` for its zlib decoder (Ninapro's MATLAB files are compressed).

7. `prune.c` - Removes least contributing neurons from a network to reduce model size and improve performance.

8. `quantize.c` - Converts a floating-point model to a 8-bit integer model.

9. `dequantize.c` - Converts an 8-bit integer model into a floating point model.

10. `export.c` / `import.c` - Convert a model between the ASCII, binary, and inplace file formats (see Model File Format below).

11. `summary.c` - Describes a model file, including which of the three formats (ASCII, binary, or inplace) it's saved in.

The eight examples below share a progression worth reading in order -- see [`examples/README.md`](examples/README.md) for a one-page summary and comparison table before diving into any one of them.

12. [`examples/character_recognition/`](examples/character_recognition/README.md) - Recognizes handwritten digits (MNIST), with two training programs sharing the same data and evaluation: `train_cnn.c` (convolution + pooling ahead of the fully-connected layers) and `train_fc.c` (a plain fully-connected network on the same task -- the baseline a CNN is normally justified against). Both use `data_prep.[ch]`/`split.py` above (this example's `samples.csv` download rule stays local to its own Makefile, though, since -- unlike the splitting logic -- fetching *this* dataset specifically isn't reusable), and share a single architecture-agnostic `test.c`. `predict.c` demonstrates how to use either's trained model in a target application to make predictions on new data (hardcoded to this example's 28x28 MNIST-style input, unlike the format-agnostic tools above). See its [README](examples/character_recognition/README.md) for the embedded-system framing, both architectures' diagrams, and sample output.

13. [`examples/gesture_recognition/`](examples/gesture_recognition/README.md) - Trains a recurrent (RNN) neural network to classify synthetic 3-axis accelerometer "gesture" time-series (the kind of sensor stream a wearable or remote control's IMU would produce) instead of a fixed image dataset -- no data file to download; every training window is generated on the fly by `gesture_data.[ch]`, shared by `train.c` and `test.c` so neither duplicates the other's data-generation logic. `test.c` independently re-evaluates a saved model against a freshly-synthesized batch of windows (unseen by construction, since nothing about `train.c`'s data is ever persisted to disk) and reports a confusion matrix plus per-class/overall accuracy. See its [README](examples/gesture_recognition/README.md) for the embedded-system framing, architecture diagram, and sample output.

14. [`examples/fall_detection/`](examples/fall_detection/README.md) - Trains three recurrent neural network architectures -- `train_rnn.c`, `train_gru.c`, `train_lstm.c` -- on the identical task: continuously monitoring a synthetic 3-axis accelerometer stream (generated by `fall_data.[ch]`) for a fall event, a multi-phase pattern (free-fall dip, impact spike, then a long stretch of post-fall stillness) spread across a much longer sequence than the gesture example's fixed windows, with a per-timestep (not per-window) label. All three share a single architecture-agnostic `test.c`, which independently re-evaluates a saved model against a freshly-synthesized, larger batch of monitoring sequences and reports per-timestep accuracy, recall (falls ever detected), false alarms, and average detection latency. See its [README](examples/fall_detection/README.md) for the embedded-system framing, all three architectures' diagrams, and a measured (not assumed) comparison of how quickly each converges.

15. [`examples/wake_word_detection/`](examples/wake_word_detection/README.md) - Trains a GRU-based network (the winner of a real RNN/GRU/LSTM comparison on this task -- see its README) to classify a whole ~1-second spoken utterance as a wake word or not, using `audio_features.[ch]`/`wav_reader.[ch]` above to turn the real Google Speech Commands dataset into training data via this example's own `prepare_data.c` (unlike MNIST's pre-flattened download, turning audio into features is itself the thing this example demonstrates). See its [README](examples/wake_word_detection/README.md) for the embedded-system framing, the library-vs-application split `predict.c` illustrates, and build/run instructions.

16. [`examples/image_classification/`](examples/image_classification/README.md) - Trains a CNN to classify a photo as a cat or a dog, using two real public datasets together, Microsoft's Kaggle Cats and Dogs and Oxford-IIIT Pet (both downloaded by this example's Makefile). This example's `prepare_data.c` decodes the images with a vendored `stb_image.h`, crops Oxford's animals using that dataset's per-pixel outlines, and builds its CSVs from fixed, reproducible splits instead of `split.py`. Its `test.c` reports accuracy on a held-out set from each dataset, and per breed on Oxford's (all 37 of them). `image_prep.[ch]` (crop, resize, normalize) is shared with `predict.c` and, like `audio_features.[ch]`, is meant to run on the target too. See its [README](examples/image_classification/README.md) for the embedded-system framing, architecture diagram, and measured results.

17. [`examples/object_detection/`](examples/object_detection/README.md) - Trains a fully convolutional network to find and count the red blood cells, white blood cells, and platelets in a microscope photo of a blood smear, using the BCCD dataset (downloaded by this example's Makefile; MIT license). Instead of one label per image, the network outputs one coarse heatmap per class -- its last layer is a 1x1 `LAYER_TYPE_CNN` with a sigmoid rather than a `LAYER_TYPE_OUTPUT` -- and `heatmap.[ch]` turns each heatmap's peaks into detections, so detection needs no changes to the library. `prepare_data.c` keeps BCCD's published split and decodes the photos with `stb_image.h` (vendored at the repository root and shared with `image_classification`). Its `test.c` scores detections against the labeled boxes and reports precision, recall, and counting error per class. `image_prep.[ch]` and `heatmap.[ch]` are shared with `predict.c` and are meant to run on the target too. See its [README](examples/object_detection/README.md) for the embedded-system framing, architecture diagram, and measured results, including how far the dataset's incomplete labels limit them.

18. [`examples/predictive_maintenance/`](examples/predictive_maintenance/README.md) - Trains a fully-connected network to recognize, from a few seconds of vibration off an accelerometer on an industrial motor and gearbox, whether the machine is healthy or which of seven failure modes (imbalance, misalignment, looseness, outer- or inner-race bearing defect, a damaged gear tooth, lubrication breakdown) is developing -- months before it fails, while its overall vibration level still looks normal. Built around an STM32U585 and an IIS3DWB vibration sensor (ST's STWIN.box kit): `vibration_features.[ch]` turns each snapshot into a 64-band spectral fingerprint with FFTs (`fft.[ch]` above) -- the vibration spectrum plus the envelope spectrum that reveals bearing and gear impacts -- and `health_monitor.[ch]` compares it with the fingerprint the same machine had at commissioning before asking the model, then debounces alerts. Both are firmware code, and `make stm32` cross-compiles them for the Cortex-M33. Training data comes from `machine_sim.[ch]`, a physics-based simulator of the seven faults on machines that each differ slightly, so nothing is downloaded. Its `test.c` runs simulated machines to failure and compares the model's warning time with a standard ISO 10816 vibration alarm's; its README measures how much the comparison with the machine's own past is worth.

19. [`examples/prosthetic_hand/`](examples/prosthetic_hand/README.md) - Trains a fully-connected network to decide, every 50ms, from a ring of 8 surface EMG electrodes around the forearm, whether the wearer of a powered prosthetic hand and wrist is trying to open or close the hand, pronate or supinate, flex or extend the wrist, or rest. Built around an STM32U585 and a TI ADS1298 analog front end, and trained and tested on real recordings from Ninapro DB2 (40 intact subjects) and DB3 (11 transradial amputees): `fetch_ninapro.py` downloads just the one exercise needed out of each subject's archive with HTTP range requests, and `mat_reader.[ch]` reads their MATLAB files using `stb_image.h`'s zlib decoder. `emg_features.[ch]` computes the classic Hudgins features, and `intent_decoder.[ch]` standardizes them with the wearer's calibration, fine-tunes the shared model to the wearer on the device (`nn_train()` running in firmware), and applies confidence rejection and a majority vote before driving the motors; both are firmware code, and `make stm32` cross-compiles them. Its `test.c` compares the network with linear discriminant analysis, the classifier commercial controllers use, on people neither has seen, and `replay.c` streams whole sessions through the firmware decoder to measure response time and how often the hand would do the wrong thing.

## Features

With this library, neural networks of any width and depth may be constructed and trained. The following activation functions are supported:

* Identity
* Linear
* ReLU
* Leaky ReLU
* ELU
* Threshold
* Sigmoid
* TanH
* GELU
* SiLU
* Softmax (output layer only -- paired automatically with cross-entropy loss; see `ACTIVATION_FUNCTION_TYPE_SOFTMAX` in nn.h)

Different activation functions may be assigned to each layer in the network.

A bias can be added to each layer independently.

`nn_train()` defaults to plain SGD (`weight += gradient * rate`, no extra memory beyond the model itself), but the optimizer is selectable via `nn_set_optimizer(nn, optimizer, momentum, beta2, epsilon)`: `NN_OPTIMIZER_MOMENTUM` (classic momentum, +1 float of persistent state per weight/bias) or `NN_OPTIMIZER_ADAM` (+2 floats per weight/bias) typically converge faster/more reliably than plain SGD, at that extra RAM cost. This is a genuine tradeoff worth thinking about for a model that needs to keep training in the field (as new samples become available on a deployed device) rather than just once, offline, before deployment: pick whichever optimizer fits the RAM you can spare on that specific target. None of this state is ever written to a saved model file, though (see `nn_save_model_ascii()`/`nn_save_model_binary()`) -- like the existing `weight_adj` gradient scratch, it's purely a RAM-side training aid, so a model saved mid-training and reloaded later always resumes with a fresh optimizer warmup, regardless of which optimizer produced it. See `nn_optimizer_t` and `nn_set_optimizer()`'s comments in `nn.h` for the full details.

The following layer types are supported (added one at a time, in order, via `nn_add_layer()`):

* **Input** - holds the network's input values; has no weights, bias, or activation of its own.
* **Fully Connected (FC)** - every neuron connects to every neuron in the previous layer; the standard building block for dense layers.
* **Convolutional (CNN)** - 2D convolution over the previous layer's feature maps, with configurable kernel size, stride, padding (same/zero-padding), and channel counts (see `cnn_t` in nn.h).
* **Pooling** - downsamples a CNN layer's feature maps; Min, Max, or Average (see `pooling_type_t` in nn.h).
* **Dropout** - training-only regularization that randomly zeroes a configurable fraction of a layer's outputs; a no-op pass-through at inference (see `dropout_t` in nn.h). In a network with no recurrent (RNN/GRU/LSTM) layer, it draws a fresh random mask every training call, same as dropout everywhere else. In a network that does have one -- where every `nn_train()`/`nn_predict()`/`nn_error()` call is one timestep of an ongoing sequence, not an independent sample -- it instead holds the same mask for the whole sequence, only resampling right after `nn_reset_state()` marks a new one starting (see `dropout_mask_valid`'s comment in nn.h); without that, dropout would inject a different random mask into the recurrent layer's hidden state on every single frame instead of cleanly regularizing it.
* **Recurrent (RNN)** - a single-timestep Elman-style recurrent layer: its hidden state (`neuron[layer]`) persists across calls, so a sequence is processed by calling `nn_train()`/`nn_predict()`/`nn_error()` once per timestep. Call `nn_reset_state()` before starting a new, independent sequence. Trained with the recurrent connection treated as a constant for gradient purposes (truncated BPTT with a truncation depth of 1), so memory use stays flat regardless of sequence length -- consistent with this library's one-sample-at-a-time training model. Width (hidden units) is set directly via `nn_add_layer()`'s `width` argument, same as Fully Connected; it takes no config struct.
* **Gated Recurrent Unit (GRU)** - like Recurrent (RNN) above (one timestep per call, `nn_reset_state()` between sequences, truncated BPTT depth 1, width set directly, no config struct), but with three internal gates (reset, update, candidate) with fixed nonlinearities that let it hold onto a signal over more timesteps than a plain RNN, at roughly 3x the parameter count and no second persistent state (unlike LSTM below). Should not be used as the network's final layer -- follow it with a normal FC/OUTPUT layer, the same way `examples/fall_detection/train_gru.c`'s GRU layer is followed by an OUTPUT layer.
* **Long Short-Term Memory (LSTM)** - like Recurrent (RNN) above (one timestep per call, `nn_reset_state()` between sequences, truncated BPTT depth 1, width set directly, no config struct), but with a second persistent state vector (the cell state, alongside the hidden state) and four internal gates (input, forget, cell-candidate, output) with fixed nonlinearities, more resistant than a plain RNN to vanishing gradients over longer sequences. Should not be used as the network's final layer -- follow it with a normal FC/OUTPUT layer, the same way `examples/fall_detection/train_lstm.c`'s LSTM layer is followed by an OUTPUT layer.
* **Output** - the network's final layer; computed the same way as Fully Connected, with support for Softmax (see Features above) in addition to the other activation functions.

`layer_type_t` also declares `LAYER_TYPE_ATTENTION`/`LAYER_TYPE_TRANSFORMER` placeholders, but they're not implemented and not currently planned: attention needs to weight multiple past states at once (and full self-attention is O(n²) in sequence length), which conflicts with the flat, O(1)-per-timestep memory model every layer type above is built around.

## Why this library

There are other embedded neural network options out there -- TensorFlow Lite for Microcontrollers and Microsoft's Embedded Learning Library among them. This one earns a place next to them on a few specific points, not by trying to out-ecosystem them:

* **On-device training, not just inference.** TFLite Micro and ELL are both built around a "train on a workstation with the full framework, freeze the graph, deploy a read-only model" workflow -- neither has a supported path for a deployed device to keep learning. `nn_train()` here works on any owned (non-flash-aliased) model, with a selectable optimizer (`nn_set_optimizer()`: SGD, Momentum, or Adam), so a device can keep training in the field: personalizing a wake-word or gesture model to one user's voice or motion after shipping, adapting to one sensor's drift, or fine-tuning without a round-trip to a server or a full reflash cycle.

* **Small enough to audit, not just small enough to run.** The entire library is two files (`nn.c`/`nn.h`), pure C with no dependency beyond libc/libm -- no C++, no interpreter, no flatbuffers/kernel-registration machinery to pull in or port. For safety-critical or certified embedded work, where someone has to actually read and sign off on what's running, handing a reviewer a library they can get through in an afternoon is a real advantage over auditing a framework's dependency graph. Pure C also means it builds cleanly on toolchains where a C++ requirement is itself an obstacle.

* **Zero-copy, flash-resident deployment as a first-class format.** `nn_load_model_inplace()` aliases a model's weights directly out of a flash-resident buffer -- no RAM copy, no parse step. A small GRU keyword-spotting model (20 inputs, 32 hidden units) quantized to int8 and exported with `export --inplace` comes to 5,572 bytes, loaded straight out of `.rodata` with zero additional RAM for weights. See [Model File Format](#model-file-format) below for the full format and `export`'s `--inplace` flag, just below.

What this library is deliberately not: a place to run attention or transformer-based models (`LAYER_TYPE_ATTENTION`/`LAYER_TYPE_TRANSFORMER` are placeholders with no implementation planned -- see the Features section above for why), and not an attempt to match TFLite Micro's model zoo, vendor board support, or CMSIS-NN-level kernel optimization. The target is small, classic on-device models that need to keep learning after deployment, on whatever MCU toolchain you already have -- not the broadest possible model support.

## Instructions

To build the nn library and the general-purpose model tools (`prune`, `quantize`, `dequantize`, `export`, `import`, `summary`), just type:
```
make
```

Each directory under `examples/` has its own Makefile and is built separately -- building one automatically builds (or reuses) the top-level library first, by recursing into this Makefile for `libnn.a`:
```
cd examples/character_recognition && make  # CNN and plain-FC digit recognition (MNIST)
cd examples/gesture_recognition && make    # RNN gesture classification
cd examples/fall_detection && make         # RNN, GRU, and LSTM fall detection
cd examples/wake_word_detection && make    # GRU wake word detection (Speech Commands)
cd examples/image_classification && make   # CNN cat vs. dog (Kaggle Cats and Dogs + Oxford-IIIT Pet)
cd examples/object_detection && make       # CNN blood cell detection and counting (BCCD)
```

The general-purpose tools below (`prune`, `quantize`, `export`, `summary`, ...) live at the repository root and work on a saved model from *any* example -- run them either from the root with a path into the example's directory (`./prune examples/character_recognition/model.txt 10`), or from inside the example's directory with a relative path back to the tool (`../../prune model.txt 10`).

To train the digit-recognition example, choosing either the CNN or the plain fully-connected (FC) architecture (see `examples/character_recognition/` above for how they compare):
```
cd examples/character_recognition
./train_cnn model.txt
```
or
```
cd examples/character_recognition
./train_fc model.txt
```
The model can be further trained (or fine-tuned) simply by re-running the same training program, which further trains a model file if it already exists -- use a different `<model-file>` for each (e.g. `model_cnn.txt`/`model_fc.txt`) if you want to train and compare both, since a model built by one architecture can't be resumed by the other.

The included example data is the MNIST data set.

To train the RNN gesture-classification example instead (no data file needed, training/validation windows are synthesized on the fly):
```
cd examples/gesture_recognition
./train gesture_model.txt
```

To train the fall-detection example instead (also no data file needed), choosing among the three included recurrent architectures (see `examples/fall_detection/` above for how they compare):
```
cd examples/fall_detection
./train_rnn fall_model_rnn.txt
./train_gru fall_model_gru.txt
./train_lstm fall_model_lstm.txt
```

To train the wake-word-detection example instead (see `examples/wake_word_detection/` above): the first `make` there downloads the real Speech Commands dataset and builds `train.csv`/`validation.csv`/`test.csv` from it, which takes a while the first time.
```
cd examples/wake_word_detection
./train wake_word_model.txt
```

To train the cat-vs-dog image-classification example instead (see `examples/image_classification/` above): the first `make` there downloads the Kaggle Cats and Dogs and Oxford-IIIT Pet datasets (~1.6GB together) and builds `train.csv`/`validation.csv`/`test.csv`/`test_oxford.csv` from them.
```
cd examples/image_classification
./train model.txt
```

To train the blood-cell object-detection example instead (see `examples/object_detection/` above): the first `make` there downloads the BCCD dataset (~8MB) and builds `train.csv`/`validation.csv`/`test.csv` and their `*_boxes.txt` label files from it.
```
cd examples/object_detection
./train model.txt
```

To evaluate the digit-recognition model's performance:
```
cd examples/character_recognition
./test model.txt
```

To evaluate the gesture-classification model instead (prints a confusion matrix against a fresh, never-before-seen batch of synthesized windows):
```
cd examples/gesture_recognition
./test gesture_model.txt
```

To evaluate a fall-detection model instead (prints per-timestep accuracy, recall, false alarms, and detection latency against a fresh, larger batch of synthesized sequences; works the same for any of the three trained models above):
```
cd examples/fall_detection
./test fall_model_rnn.txt
```

To evaluate a wake-word-detection model instead (prints accuracy, false-reject rate, and false-accept rate against the held-out `test.csv` split):
```
cd examples/wake_word_detection
./test wake_word_model.txt
```

To evaluate an image-classification model instead (prints a confusion matrix and accuracy for both held-out test sets, plus accuracy per breed on the Oxford one):
```
cd examples/image_classification
./test model.txt
```

To evaluate an object-detection model instead (prints a threshold sweep on the validation set, then precision, recall, and counting error per class on the held-out test set):
```
cd examples/object_detection
./test model.txt
```

To use a trained digit-recognition model:
```
cd examples/character_recognition
./predict model.txt
```

To prune a model (this example removes the 10 least contributing neurons):

```
./prune examples/character_recognition/model.txt 10
```

To quantize a trained model (which is floating point by default), run the following command:

```
./quantize examples/character_recognition/model.txt model_quantized.txt
```

To export a trained model as a flash-resident binary for a microcontroller target, add `--inplace` so `export` writes the zero-copy "inplace" format instead of the regular binary format. `export` accepts any of the three model formats as input (it auto-detects ASCII, binary, or an existing inplace file), so this works directly on whichever one you have:

```
./export examples/character_recognition/model.txt model_inplace.bin --inplace
```

See [Embedding an inplace model as a C header](#embedding-an-inplace-model-as-a-c-header) below for turning that file into a `.h` you can `#include` and pass to `nn_load_model_inplace()`. Omit `--inplace` to instead get the regular binary format (`nn_load_model_binary()`/`nn_load_model_memory()`).

To check which format a model file is in (along with its version, quantization status, and layer-by-layer layout), run:

```
./summary examples/character_recognition/model.txt
```

## Architecture

The network architecture is a fully connected feed-forward neural network. It is based on floating-point computation. The widths of each layer, the activation function to be used, and the bias for each layer is set as each successive layer is added to the network using the `nn_add_layer()` function call. Multiple layers may be added to construct a deep neural network.

## Demonstration

This embedded neural network library was used to power a handwritten character recognition application running on an STM32H7 microcontroller. Check it out here:

https://www.youtube.com/watch?v=cqjwSkrGtww

## Model File Format

The model can be saved in the following formats:

* **ASCII** - a text file of floating-point values. The first line depicts the number of layers, inclusive of the input and output layers. The construct of each of those layers comprises the next set of lines, one line for each layer. The format of each line is width (in neurons), activation function, and bias. The remaining lines are the weights of each neuron in each layer, for all layers. Since there are no weights associated with the neurons in the input layer, these are skipped, and do not exist in the model file.

* **Binary** - a compact, raw binary encoding of the same information. Every binary model file begins with the 4-byte magic number `NNB1`, followed by the same fields the ASCII format stores (quantized flag, version, layer definitions, weights, and biases), written as raw integers/floats rather than text.

`nn_load_model()` reads a model file's first few bytes and dispatches to the ASCII or binary loader automatically based on the magic number, so any tool that calls it can open either kind of model file without knowing in advance which format it's in. `nn_save_model()` writes binary format when the destination path ends in `.bin` (case-insensitive) and ASCII format otherwise. Every example's `train`/`test` (and `predict`, for the recognition example), plus `prune`, `quantize`, `dequantize`, and `summary`, all use these, so passing e.g. `model.bin` instead of `model.txt` is enough to train, evaluate, prune, (de)quantize, or run inference against a binary model file. `nn_load_model_ascii`/`nn_save_model_ascii` and `nn_load_model_binary`/`nn_save_model_binary` remain available for callers that need to force a specific format regardless of extension (as `import` does, to convert binary back to ASCII).

`nn_model_format(path)` peeks a file's first few bytes (without loading it) to report which of the three formats -- ASCII, binary, or inplace -- it's in; `summary` uses this to print a `Model Format:` line, and `export` uses it to accept any of the three as input (reading an inplace file into a buffer and loading it with `nn_load_model_inplace()`, since that format has to be read from memory rather than a plain path) regardless of which one it writes as output. `export` also accepts an `--inplace` flag to write the inplace format (below) instead of the regular binary format.

For targets with no filesystem, `nn_load_model_memory(data, size)` parses that same binary format directly out of a caller-supplied buffer (e.g. a model baked into flash as a byte array on a microcontroller) instead of reading from a file, with no `FILE*`/`fopen` dependency. It copies the model into its own allocations, so `data` only needs to stay valid for the duration of the call.

* **Inplace** - a third format (magic `NNP1`), written by `nn_save_model_inplace()` and read back with zero copy by `nn_load_model_inplace(data, size)`: instead of parsing weights/biases into freshly malloc'd RAM, the returned model's weight (or weight_quantized) and bias (or bias_quantized) arrays point directly into the caller's buffer. This is the format for microcontroller targets where RAM, not flash, is the tight resource -- the buffer (typically a `static const uint8_t[]` baked into flash) must stay valid for as long as the model is used, and the model it produces is read-only: `nn_train()`, `nn_quantize()`, `nn_dequantize()`, `nn_remove_neuron()`, and `nn_prune_lightest_neuron()` all refuse to run against it. Use `nn_predict()`/`nn_error()` for inference as usual, and release it with `nn_free()` as usual -- it knows not to free the aliased buffers.

## Integration

To use this nn library in your own embedded system, it is only necessary to pull in the nn.c and nn.h files into your project. The other source files in the nn package are intended for data preparation for offline training, as well as examples of training and inference. For microcontroller-class inference-only use, nn_load_model_inplace() (see Model File Format above) is the intended entry point: it runs inference directly out of a flash-resident model with no RAM copy of the weights; `nn_load_model_memory()` is the fallback when the model instead needs to be copied into (and is free to be modified/replaced in) RAM.

The one exception is `audio_features.[ch]` (see Overview above): if your embedded project does on-device audio feature extraction the way `examples/wake_word_detection/` does, pull that pair of files in alongside `nn.c`/`nn.h` too -- unlike every other file in this repository besides those two, it's meant to run on the target itself, not just at training time.

### Embedding an inplace model as a C header

To integrate an inplace-format model into firmware as a flash-resident array, convert the file `nn_save_model_inplace()` wrote with [`xxd -i`](https://linux.die.net/man/1/xxd):

```
xxd -i model_inplace.bin > model_inplace.h
```

This generates a `.h` file with a `unsigned char model_inplace_bin[] = {...};` array and a `unsigned int model_inplace_bin_len` byte count. `#include` it, then hand the array straight to `nn_load_model_inplace()`:

```c
#include "model_inplace.h"

nn_t *nn = nn_load_model_inplace(model_inplace_bin, model_inplace_bin_len);
```

`xxd -i` doesn't add an alignment attribute, so on a toolchain/linker that doesn't already place `.rodata` arrays 4-byte aligned, add one by hand to the generated declaration (`unsigned char model_inplace_bin[] __attribute__((aligned(4))) = {...};`) -- `nn_load_model_inplace()` requires 4-byte alignment (see its comment in nn.c).

## License

Copyright (c) 2019-2026 Cole Design and Development, LLC. All rights reserved.

Licensed under the [Apache License 2.0](./LICENSE).
