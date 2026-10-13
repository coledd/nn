<p align="center">
  <img src="figures/banner.svg" alt="Character Recognition: reading handwritten digits on a microcontroller, and measuring what convolution buys over a plain network." width="100%">
</p>

> **Abstract.** Turning a small handwritten image into a digit is the "hello world" of embedded vision, and the same model shape reads handwritten amounts, form fields and dials on devices with no network connection. This example trains two networks on the same 70,000 MNIST digits — a convolutional network (CNN) and a plain fully-connected one — and evaluates both with the same test program, so what the convolution layers buy is measured rather than assumed. On 7,000 held-out digits the CNN scores 98.24% and the fully-connected network 97.26%, with about the same number of parameters (100K and 109K) but eight times the arithmetic per digit for the CNN. The library behind it has run this task live on an STM32H7 microcontroller.

| | |
|---|---|
| **Task** | 10-way classification of a 28 × 28 handwritten digit |
| **Data** | MNIST: 70,000 digits, binarized; 80 / 10 / 10% train / validation / test split |
| **Model** | CNN: 2 × (5 × 5 conv + 2 × 2 max pool) → FC 120 → FC 20 → softmax 10 (100,254 parameters). Baseline: FC 128 → FC 64 → softmax 10 (109,386) |
| **Hardware** | STM32H7 microcontroller (the library's own demonstration) |
| **Result** | Test accuracy 98.24% (CNN) vs. 97.26% (fully connected) |

## 1. The problem

Handwritten-digit recognition is the "hello world" of embedded computer vision, but it's not just a toy: the same shape of model runs behind a smart pen digitizing handwritten forms, a point-of-sale device reading a handwritten amount, a meter reader recognizing a dial, or a postal sorter reading a ZIP code — anywhere a small, fixed-size image has to become a class label on the device, without a round trip to a server.

The question this example answers with numbers is the one every such project faces: what does the *shape* of a network cost and buy, beyond its raw size? A convolutional network shares a small set of weights across the whole image and keeps its spatial structure; a fully-connected one gives every pixel its own weight to every unit. Both are trained here on identical data and scored by the identical `test` program.

## 2. The device

<p align="center">
  <img src="figures/device.svg" alt="A handwriting pad with a stylus and a handwritten 8; the image is scaled and thresholded to a 28 by 28 one-bit grid and passed to an STM32H7 microcontroller running the 100K-parameter CNN, which outputs the digit 8 with its confidence." width="900">
</p>
<p align="center"><sub><b>Figure 1.</b> A representative device: the handwritten input is scaled to the 28 × 28, one-bit-per-pixel form MNIST digits come in, and the microcontroller classifies it. The digit shown is <code>predict.c</code>'s built-in sample. Schematic, not to scale.</sub></p>

The library has powered exactly this application — a handwritten character recognizer running live on an STM32H7 microcontroller [3]. The CNN's 100,254 weights are 392 KB as 32-bit floats; the STM32H743, for example, has 2 MB of flash and 1 MB of RAM. [Section 6](#6-running-on-the-device) covers fitting and running the model.

## 3. The data

MNIST [1, 2] is 70,000 grayscale digits, 28 × 28 pixels, written by about 500 people and size-normalized and centered in the frame. The Makefile downloads the four original files, and `prepare_data` converts them into `samples.csv`: each pixel becomes 1 if its gray level is 128 or more and 0 otherwise, followed by a one-hot label. `split.py` then shuffles all 70,000 into train (80%), validation (10%) and test (10%) sets.

<p align="center">
  <img src="figures/mnist_samples.png" alt="A grid of real MNIST test digits, fourteen of each digit from 0 to 9, shown binarized in black on white; handwriting styles vary widely, for example open and closed 4s and slanted 1s." width="640">
</p>
<p align="center"><sub><b>Figure 2.</b> Real MNIST digits as the networks see them: fourteen of each class, binarized at a gray level of 128 exactly as <code>prepare_data</code> does.</sub></p>

## 4. Method

### 4.1 `train_cnn.c`: convolution and pooling ahead of the fully-connected layers

<p align="center">
  <img src="figures/architecture_cnn.svg" alt="CNN architecture: input 28 by 28 by 1; 5 by 5 convolution with 8 filters and same padding to 28 by 28 by 8; 2 by 2 max pool to 14 by 14 by 8; 5 by 5 convolution with 16 filters to 14 by 14 by 16; 2 by 2 max pool to 7 by 7 by 16; flatten into a 120-unit GELU layer; 30% dropout; a 20-unit GELU layer; 10-way softmax output." width="100%">
</p>
<p align="center"><sub><b>Figure 3.</b> The CNN. Two convolution and pooling stages turn the 28 × 28 image into 7 × 7 × 16 feature maps before the fully-connected layers.</sub></p>

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 28 × 28 × 1 | one channel, one bit per pixel |
| 1 | CNN | 28 × 28 × 8 | 5 × 5 kernel, 8 filters, stride 1, "same" padding, so strokes near the border are seen by every kernel position |
| 2 | Pool | 14 × 14 × 8 | 2 × 2 max pool, stride 2 |
| 3 | CNN | 14 × 14 × 16 | 5 × 5 kernel, 16 filters, "same" padding — features of features: strokes, then shapes |
| 4 | Pool | 7 × 7 × 16 | 2 × 2 max pool, stride 2 |
| 5 | FC | 120 | GELU |
| 6 | Dropout | 120 | 30%, training only |
| 7 | FC | 20 | GELU |
| 8 | Output | 10 | Softmax (cross-entropy loss) |

100,254 trainable parameters. The two convolutions contribute only 3,424 of them (208 + 3,216); the first fully-connected layer, reading the 784 flattened values the second pooling stage leaves, holds 94,200. The convolutions are where the arithmetic goes instead: about 880,000 multiply-accumulates per digit, against about 109,000 for the fully-connected network.

### 4.2 `train_fc.c`: a plain fully-connected network

<p align="center">
  <img src="figures/architecture_fc.svg" alt="Fully-connected architecture: 784 flattened input pixels, dense all-to-all into a 128-unit GELU layer, then a 64-unit GELU layer, then a 10-way softmax output." width="820">
</p>
<p align="center"><sub><b>Figure 4.</b> The fully-connected baseline: every pixel connects to every unit of the first layer.</sub></p>

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 784 | flattened 28 × 28 pixels, no spatial structure |
| 1 | FC | 128 | GELU |
| 2 | FC | 64 | GELU |
| 3 | Output | 10 | Softmax (cross-entropy loss) |

109,386 trainable parameters. Every input pixel connects directly to every unit in the first layer: no shared weights, no spatial locality, no translation invariance. This is the baseline a CNN is normally justified against.

### 4.3 Training and testing

Both programs train with a learning rate of 0.005 and stop once validation error hasn't improved for 5 epochs, saving the best model. `test` scores train and test sets the same way for either model: a digit counts as correct only if the right class's softmax output is at least 0.5 and no other class's is — slightly stricter than taking the most probable class.

## 5. Results

The figures and numbers below come from one run of each program; random initialization and the shuffled split make every run a little different. The CNN led in both pairs of runs measured while writing this example (98.24% vs. 97.26% here; 98.49% vs. 96.13% in an earlier pair).

<p align="center">
  <img src="figures/training.svg" alt="Training and validation cross-entropy per epoch. CNN: validation error falls to its best, 0.058, at epoch 6 and training error stays close to it. Fully connected: training error falls to 0.005 while validation error bottoms out at 0.092 at epoch 5 and then rises." width="900">
</p>
<p align="center"><sub><b>Figure 5.</b> Training curves. The dot marks the epoch whose model was saved (lowest validation error); training stopped five epochs later.</sub></p>

| | Parameters | Multiply-adds per digit | Epochs | Training time | Train accuracy | Test accuracy |
|---|---|---|---|---|---|---|
| `train_cnn` | 100,254 | ~880K | 11 | 8 min 25 s | 99.17% | **98.24%** |
| `train_fc` | 109,386 | ~109K | 10 | 15 s | 99.44% | 97.26% |

Training times are single-threaded on the desktop used for this document.

**What convolution buys.** With slightly fewer parameters, the CNN makes about a third fewer mistakes on unseen digits (123 vs. 192 of 7,000 by `test`'s rule). The training curves show why: the fully-connected network fits its training set *better* (99.44%) but generalizes worse — its validation error turns upward after epoch 5 while its training error keeps falling toward zero. The CNN's shared, local filters can't memorize pixel positions the same way, and its training and validation errors stay together.

**What it costs.** The arithmetic: about eight times as many multiply-adds per digit, and a training run about 35 times longer. For a device reading one digit at a time that is still only milliseconds of work (see section 6), but it's the trade the CNN makes: computation for accuracy, at about the same memory.

<p align="center">
  <img src="figures/confusion.svg" alt="Confusion matrix of the CNN on 7,000 test digits; diagonal counts between 581 and 774; the largest off-diagonal counts are 4 read as 9 (8 times), 8 read as 9 (7), and 2 and 3 read as 7 (6 each)." width="560">
</p>
<p align="center"><sub><b>Figure 6.</b> The CNN's test-set confusion matrix by most probable class (6,893 of 7,000 correct, 98.47%). Diagonal: correct; red cells: mistakes, darker for more.</sub></p>

<p align="center">
  <img src="figures/mistakes.png" alt="Twenty-four misclassified test digits, each labeled with its true and predicted class, for example a 4 read as 9 and a 2 read as 4; many are ambiguous or unusually written." width="760">
</p>
<p align="center"><sub><b>Figure 7.</b> The first 24 test digits the CNN gets wrong, labeled true → predicted. Many are hard for a person too.</sub></p>

The mistakes are the familiar MNIST ones: 4s that close at the top become 9s, 8s and 9s trade loops, and 2s and 3s with a straight tail become 7s. Digits 4 and 9 have the lowest per-class accuracy (97.8% and 97.7%) and 0 and 1 the highest (99.3% and 99.2%).

<p align="center">
  <img src="figures/predict.svg" alt="predict.c's built-in 28 by 28 sample, a blocky 8, beside the CNN's softmax output: 0.989 for 8 and under 0.004 for every other digit." width="760">
</p>
<p align="center"><sub><b>Figure 8.</b> <code>predict</code> on its built-in sample: the input, and the CNN's probability for each digit.</sub></p>

## 6. Running on the device

`predict.c` is the shape of the application code: load a model, fill a 784-float input buffer with the 28 × 28 image (0 or 1 per pixel, row by row), call `nn_predict()`, and take the most probable class. On the device the image comes from the touch panel or camera instead, scaled and centered the way MNIST digits are and thresholded at the same gray level.

The models' weights are 392 KB (CNN) and 427 KB (fully connected) as 32-bit floats. The repository's general tools shrink them for flash: `../../quantize` converts a model to 8-bit integers, about a quarter of the size, and `../../export ... --inplace` writes the zero-copy format that `nn_load_model_inplace()` runs straight out of flash (see the top-level README). By a multiply-add count, the CNN's ~880,000 per digit come to a few milliseconds on a Cortex-M7 at several hundred megahertz, and the fully-connected network's ~109,000 to well under one — an estimate, not a measurement.

## 7. Limitations

- **MNIST is clean.** Its digits are size-normalized, centered and written on blank paper. Real input needs the same preprocessing — crop to the stroke, scale to 20 × 20 inside the 28 × 28 frame, center — and a model meant for a particular pen, camera or form should be fine-tuned on samples from it.
- **Binarized input.** The pixels are thresholded at 128, which keeps the input to one bit per pixel but drops the anti-aliasing at stroke edges.
- **One run each.** The numbers come from a single training run of each network; they shift by a few tenths of a percent from run to run.
- **Digits only.** Letters, multi-digit strings and rejecting non-digits all need more classes, more data, or a detector in front.

## Appendix A. Build and run

```
cd examples/character_recognition
make
```

This downloads the original MNIST files (~11 MB, into `dataset/`; yann.lecun.com no longer serves them, so they come from Google's byte-identical CVDF mirror), converts them to `samples.csv` with `prepare_data`, and splits that into `train.csv` / `validation.csv` / `test.csv` with `split.py` (see the top-level README for its `--train` / `--validation` options).

| Command | What it does |
|---|---|
| `./train_cnn model_cnn.txt` | trains the CNN (resumes if the file exists) |
| `./train_fc model_fc.txt` | trains the fully-connected network; use a different model file than the CNN's |
| `./test model_cnn.txt` | scores either model on the train and test sets |
| `./predict model_cnn.txt` | runs either model on `predict.c`'s built-in sample |

The figures in this document are generated by `tools/docs/character_recognition.py`.

## Appendix B. Program output

Training log (`train_cnn`):
```
Creating new model.
train error, validation error, learning rate
0.22502, 0.09638, 0.00500
0.09891, 0.08375, 0.00500
0.08472, 0.07486, 0.00500
0.07124, 0.06229, 0.00500
0.06613, 0.07641, 0.00500
0.06180, 0.05759, 0.00500
0.05565, 0.07670, 0.00500
0.05654, 0.07180, 0.00500
0.05377, 0.06515, 0.00500
0.05713, 0.06905, 0.00500
0.05127, 0.06704, 0.00500
No validation improvement for 5 epochs (best: 0.05759) -- stopping early.
Final (last epoch) train error: 0.051266, validation error: 0.067037
Best validation error (the model saved to disk): 0.057591
Training epochs: 11
```

`test model_cnn.txt`:
```
Train: 55537/56000 = 99.17%
Test : 6877/7000 = 98.24%
```

`predict model_cnn.txt` (per-class probability for the built-in sample):
```
0: 0.00112
1: 0.00124
2: 0.00112
3: 0.00367
4: 0.00009
5: 0.00264
6: 0.00072
7: 0.00046
8: 0.98879
9: 0.00015
```

Training log (`train_fc`):
```
Creating new model.
train error, validation error, learning rate
0.26216, 0.14512, 0.00500
0.10713, 0.12518, 0.00500
0.07035, 0.10774, 0.00500
0.04911, 0.10364, 0.00500
0.03547, 0.09226, 0.00500
0.02390, 0.09820, 0.00500
0.01793, 0.09486, 0.00500
0.01047, 0.10639, 0.00500
0.00784, 0.10681, 0.00500
0.00491, 0.10380, 0.00500
No validation improvement for 5 epochs (best: 0.09226) -- stopping early.
Final (last epoch) train error: 0.004911, validation error: 0.103801
Best validation error (the model saved to disk): 0.092258
Training epochs: 10
```

`test model_fc.txt`:
```
Train: 55688/56000 = 99.44%
Test : 6808/7000 = 97.26%
```

## References

1. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, 1998.
2. Y. LeCun, C. Cortes and C. J. C. Burges, "The MNIST database of handwritten digits."
3. Handwritten character recognition on an STM32H7 with this library, [video demonstration](https://www.youtube.com/watch?v=cqjwSkrGtww).
