# Examples

Eight self-contained examples of constructing, training, and evaluating a network with this library. Each has its own Makefile (`cd examples/<name> && make`, recursing into the top-level Makefile for the shared library), its own README (embedded-system framing, architecture diagram, build/run instructions, sample output), and generates, synthesizes, or downloads its own training data -- nothing here needs to be built from the repository root.

Read in order, they also tell a story: each example needs a little more memory (or a little more realism) than the last to solve its task, and the layer type -- or, for the last one, the extra feature-extraction stage in front of it -- tracks that escalation directly. The last one, `predictive_maintenance`, turns that around: its network is the simplest in the set, because the work has moved into a signal-processing front end and into what it compares each input against -- the same machine's own past. The last, `prosthetic_hand`, keeps that shape -- a plain FC network behind a feature front end -- but on real data from real people, where the hard part is that no two of them are alike: the model is fine-tuned to each wearer on the device itself, and what turns its outputs into motor commands matters as much as the model.

| Example | Layer type | Task | Parameters |
|---|---|---|---|
| [`character_recognition`](character_recognition/README.md) | CNN *vs.* plain FC (both included, for comparison) | Classify a whole 28×28 image at once -- no memory of anything needed | ~100K (CNN, two conv+pool stages) / ~109K (FC) |
| [`gesture_recognition`](gesture_recognition/README.md) | RNN | Classify a short (32-timestep) burst of sensor readings -- needs to remember the last second or so | 388 |
| [`fall_detection`](fall_detection/README.md) | RNN *vs.* GRU *vs.* LSTM (all three included, for comparison) | Continuously monitor a long (150-timestep) stream and flag a rare event -- needs to remember something brief that happened many steps ago, selectively, without it decaying | 337 (RNN) / 977 (GRU) / 1,297 (LSTM) |
| [`wake_word_detection`](wake_word_detection/README.md) | GRU (the winner of a measured RNN *vs.* GRU *vs.* LSTM comparison -- see its README) | Classify a whole (49-frame) real spoken utterance as a wake word or not -- same fixed-window shape as `gesture_recognition`, but on real audio (via a new `audio_features.[ch]` feature-extraction stage) instead of a synthetic sensor stream | ~5,121 |
| [`image_classification`](image_classification/README.md) | CNN | Classify a pet photo as cat or dog -- same whole-image-at-once shape as `character_recognition`, but on ~23,000 real photos from two public datasets, with accuracy broken down per breed | ~25K |
| [`object_detection`](object_detection/README.md) | Fully convolutional CNN (1×1 conv heatmap head) | Find and count every red cell, white cell, and platelet in a microscope photo -- the first example with more than one answer per input: what is in the frame, where, and how many | ~16K |
| [`predictive_maintenance`](predictive_maintenance/README.md) | Plain FC, behind a two-FFT spectral "fingerprint" front end | Tell from a few seconds of vibration whether an industrial machine is healthy or which of seven faults is developing -- months before it fails -- by comparing its frequency fingerprint now with the one it had when installed | 4,792 |
| [`prosthetic_hand`](prosthetic_hand/README.md) | Plain FC, behind a surface-EMG feature front end, fine-tuned on the device | Decide every 50ms, from 8 electrodes around the forearm, whether the wearer of a powered prosthetic hand is trying to open or close it, rotate, flex or extend the wrist, or rest -- calibrated to each wearer in under a minute, measured on intact subjects and amputees | 4,423 |

## [`character_recognition`](character_recognition/README.md)

<img src="character_recognition/architecture_cnn.svg" alt="CNN architecture diagram" width="520"><br>

Recognizes handwritten digits (MNIST) -- the "hello world" of embedded computer vision, and the network behind this library's own STM32H7 hardware demo (see the top-level README). Trains **two** architectures on identical data -- `train_cnn.c` (convolution + pooling ahead of the fully-connected layers) and `train_fc.c` (a plain FC network, no convolution at all) -- so the accuracy a CNN actually buys you over the baseline is something you can measure, not just take on faith. No recurrence involved: the whole image is seen at once.

## [`gesture_recognition`](gesture_recognition/README.md)

<img src="gesture_recognition/architecture.svg" alt="RNN architecture diagram" width="420"><br>

Classifies a short burst of simulated 3-axis accelerometer readings (STILL / SHAKE / TILT_LEFT / TILT_RIGHT) -- the shape of model behind gesture-based wake/control features on wearables and remote controls. A single instantaneous reading can't tell these apart; you need the shape of the motion over a couple dozen timesteps. `LAYER_TYPE_RNN` gives the network a hidden state that persists across calls, so it builds up a picture of the gesture one timestep at a time instead of needing the whole window handed to it at once.

## [`fall_detection`](fall_detection/README.md)

<img src="fall_detection/architecture_lstm.svg" alt="LSTM architecture diagram" width="420"><br>

Continuously monitors a much longer simulated accelerometer stream for a fall -- a fall-detection pendant's actual job. A fall is a multi-phase pattern (free-fall dip, impact spike, then a long stretch of unusual stillness afterward), and confirming it's genuine means remembering the brief early dip across a long, quiet gap. Trains **three** architectures on identical data -- `train_rnn.c`, `train_gru.c`, `train_lstm.c` -- and measures them honestly rather than assuming gating wins: all three reach the same accuracy ceiling on this task, but the gated ones (GRU, LSTM) get there in reliably fewer training epochs. See this example's own README for the full epoch-count comparison.

## [`wake_word_detection`](wake_word_detection/README.md)

Classifies a whole ~1-second spoken utterance as a single wake word ("marvin") or not -- the always-on listener in front of a voice-controlled device. Structurally it's a fixed-window classifier like `gesture_recognition` (one label per whole window, not `fall_detection`'s per-timestep-varying one), but it's the first example built on a real, downloaded dataset (Google Speech Commands) instead of a synthetic generator, which means the model needs a real signal-processing stage -- a new `audio_features.[ch]` at the repository root (framing, FFT, mel filterbank, per-clip-normalized log energy) -- in front of a recurrent layer. An early version of this example compared the same three recurrent architectures `fall_detection` does; on this real, noisier task GRU won clearly (unlike `fall_detection`'s task, where all three tie), so `train.c` here trains GRU alone -- see this example's own README for the measured numbers, why normalization mattered more than architecture choice, and the library-vs-application split its `predict.c` illustrates.

## [`image_classification`](image_classification/README.md)

<img src="image_classification/architecture.svg" alt="Image classification CNN architecture diagram" width="520"><br>

Tells a cat from a dog in a photo (the Kaggle Cats and Dogs and Oxford-IIIT Pet datasets) -- the decision a camera-equipped pet door or feeder has to make on-device before letting the right animal through. Architecturally it's a step back to `character_recognition`'s territory (a CNN looking at one whole image, no memory), but it closes the loop on the story the other examples tell about realism: ordinary photos instead of centered digits, 37 breeds instead of one look per class, and a per-breed accuracy breakdown that shows exactly which cats and dogs the model confuses. Its README reports the measured numbers as they are, including how much room there is left above them.

## [`object_detection`](object_detection/README.md)

<img src="object_detection/architecture.svg" alt="Object detection fully convolutional network architecture diagram" width="520"><br>

Finds and counts the red blood cells, white blood cells, and platelets in a microscope photo of a blood smear (the BCCD dataset) -- the job of a handheld blood analyzer that has to produce a count with no lab and no network. Every other example gives one answer per input; this one gives a list of objects, each with a class and a position. It does that without any change to the library: the network ends in a 1×1 convolution instead of a fully-connected output layer, producing one coarse heatmap per class, and a small decoder (`heatmap.[ch]`) turns each heatmap's peaks into detections. Its README measures what made that work (spreading each training target over neighboring cells, without which the rare classes never trained at all) and what didn't (a 4× range in parameter count), and shows how much of its remaining error is the dataset's own missing labels rather than the model.

## [`predictive_maintenance`](predictive_maintenance/README.md)

<img src="predictive_maintenance/architecture.svg" alt="Predictive maintenance architecture diagram" width="620"><br>

Watches an industrial motor and gearbox through an accelerometer and names which of seven failure modes is developing in it -- imbalance, misalignment, looseness, an outer- or inner-race bearing defect, a damaged gear tooth, or lubrication breakdown -- long before it fails, while a traditional overall-vibration alarm still reads normal. Built around an STM32U585 with an IIS3DWB vibration sensor. Each snapshot becomes a 64-band frequency fingerprint (an FFT of the vibration, plus a second FFT of its envelope that brings out the repetition rate of bearing and gear impacts), and the network sees both that fingerprint and how far it has moved from the one the same machine had at commissioning. Training data comes from a physics-based simulator of the seven faults on many slightly different machines. Its README measures what each part is worth: the warning time against an ISO 10816 vibration alarm, and the false alarms and lost warning time without the comparison to the machine's own past.

## [`prosthetic_hand`](prosthetic_hand/README.md)

<img src="prosthetic_hand/architecture.svg" alt="Prosthetic hand architecture diagram" width="620"><br>

Turns the electrical activity of the forearm muscles, picked up by a ring of 8 surface electrodes in a prosthesis socket, into commands for a powered hand and wrist: open, close, pronate, supinate, flex, extend, or rest. Built around an STM32U585 and a TI ADS1298 analog front end, and trained and tested on real recordings from Ninapro (40 intact subjects, 11 transradial amputees), downloaded by the Makefile one exercise at a time with HTTP range requests. `emg_features.[ch]` computes the classic Hudgins features every 50ms, and `intent_decoder.[ch]` standardizes them with the wearer's calibration, fine-tunes a shared model to the wearer on the device, and turns the network's output into a motor command with confidence rejection and a majority vote. Its README measures the network against the linear discriminant analysis commercial controllers use, what the shared model, calibration time and electrode count are each worth, and -- by replaying whole sessions through the firmware decoder -- how often the hand would do the wrong thing, which is where the gap between intact subjects and amputees shows.
