# Character Recognition

Trains a network to recognize handwritten digits (MNIST) -- two architectures on the same data, so you can see what the convolutional layers actually buy you.

## In an embedded system

Handwritten-digit recognition is the "hello world" of embedded computer vision, but it's not just a toy: the same shape of model is what runs behind a smart pen digitizing handwritten forms, a point-of-sale device reading a handwritten amount, a meter reader OCR-ing a dial, or a postal sorter reading a ZIP code -- anywhere a small, fixed-size image needs to become a class label on-device, without round-tripping to a server. This `nn` library's own hardware demo (see the top-level README) is exactly this: a handwritten-character recognizer running live on an STM32H7 microcontroller.

This example trains two different architectures on the identical data, so the tradeoff a real embedded project faces -- accuracy vs. compute/memory budget vs. how much of that budget the *shape* of the network costs you, not just its raw size -- is something you can actually see numbers for, rather than take on faith.

## Model architecture

### `train_cnn.c` -- convolution + pooling ahead of the fully-connected layers

![CNN architecture: Input 28x28x1, 5x5 conv x8 same padding to 28x28x8, 2x2 max pool to 14x14x8, a second 5x5 conv x16 same padding to 14x14x16, a second 2x2 max pool to 7x7x16, flatten into a 120-unit GELU FC layer, 30% dropout, a 20-unit GELU FC layer, then a 10-way softmax output](architecture_cnn.svg)

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 28×28×1 | one channel, raw pixel intensities |
| 1 | CNN | 28×28×8 | 5×5 kernel, 8 filters, stride 1, "same" padding (keeps the full 28×28 instead of shrinking to 24×24, so digit strokes near the border still get seen by every kernel position) |
| 2 | Pool | 14×14×8 | 2×2 max pool, stride 2 |
| 3 | CNN | 14×14×16 | 5×5 kernel, 16 filters, stride 1, "same" padding -- a second round of feature extraction (edges-of-edges, not just edges) over the first stage's pooled output |
| 4 | Pool | 7×7×16 | 2×2 max pool, stride 2 |
| 5 | FC | 120 | GELU |
| 6 | Dropout | 120 | 30%, training only |
| 7 | FC | 20 | GELU |
| 8 | Output | 10 | Softmax (cross-entropy loss) |

~100K trainable parameters -- roughly half of a single-conv-stage version of this same network (~191K), despite adding a whole extra layer. Both convolutions together still contribute almost none of that (208 + 3,216 = 3,424 weights): the first FC layer is still what dominates the parameter count, but a *second* conv+pool stage shrinks what reaches it (7×7×16 = 784, flattened) instead of a single, wider stage growing it -- widening the first stage to 16 filters instead (an alternative that was measured, not just assumed inferior) still leaves a single 14×14×8→flatten, doubling the FC layer's input to 3,136 and its weight count right along with it, for a smaller accuracy gain than adding this second stage does. See [How they compare](#how-they-compare) below for the measured numbers behind that claim.

### `train_fc.c` -- plain fully-connected network, no convolution

![FC architecture: Input 784, dense all-to-all into a 128-unit GELU FC layer, then a 64-unit GELU FC layer, then a 10-way softmax output](architecture_fc.svg)

| Layer | Type | Output shape | Notes |
|---|---|---|---|
| 0 | Input | 784 | flattened 28×28 pixels, no spatial structure |
| 1 | FC | 128 | GELU |
| 2 | FC | 64 | GELU |
| 3 | Output | 10 | Softmax (cross-entropy loss) |

~109K trainable parameters -- fewer than the CNN, despite having no convolution/pooling at all, since every one of its FC layers is smaller. Every input pixel connects directly to every unit in the first FC layer: no shared weights, no spatial locality, no translation invariance. This is the baseline a CNN is normally justified against.

### How they compare

A real run of each (see [Sample output](#sample-output) below for the full logs):

| | Parameters | Epochs to converge | Test accuracy |
|---|---|---|---|
| `train_cnn` | ~100K | 11 | 98.49% |
| `train_fc` | ~109K | 8 | 96.13% |

Your numbers will vary run to run (random weight init, shuffled data), but expect the CNN to consistently edge out the FC network on this task -- the point of having both examples side by side.

`train_cnn.c`'s two-conv-stage shape above isn't the only one that was tried -- measured against two alternatives on the same data (two runs each, since a single run isn't enough to separate a real difference from ordinary run-to-run noise):

| | Parameters | Test accuracy (2 runs) | Avg |
|---|---|---|---|
| Single stage, 8 filters | ~191K | 97.64%, 97.73% | 97.69% |
| Single stage, 16 filters | ~379K | 97.77%, 98.31% | 98.04% |
| **Two stages, 8→16 filters (current)** | **~100K** | **98.44%, 98.49%** | **98.47%** |

Widening the single conv stage to 16 filters does buy back some accuracy over the original 8-filter version, but at roughly double the parameters (since the flattened width feeding the first FC layer doubles right along with it). Adding a second, narrower conv+pool stage instead beat both -- higher accuracy *and* about half the parameters of even the original 8-filter version -- which is why it's what ships here.

## Build and run

```
cd examples/character_recognition
make
```
This also downloads the original MNIST files (~11MB, into `dataset/`; yann.lecun.com no longer serves them, so they come from Google's byte-identical CVDF mirror), converts them to `samples.csv` with `prepare_data` (all 70,000 digits, each pixel 1 if its gray level is 128 or more and 0 otherwise, plus a one-hot label), and splits that into `train.csv`/`validation.csv`/`test.csv` via `split.py` (see the top-level README for that script's `--train`/`--validation` options).

Train the CNN:
```
./train_cnn model_cnn.txt
```
Train the plain FC network (use a different `<model-file>` than the CNN's -- a model built by one architecture can't be resumed by the other):
```
./train_fc model_fc.txt
```
Either can be resumed/fine-tuned by re-running the same command against an existing model file.

Evaluate either (the same `test.c` works for both, since it just reads whatever the model's input/output widths are):
```
./test model_cnn.txt
./test model_fc.txt
```

Run inference against a trained model:
```
./predict model_cnn.txt
```

## Sample output

Training log (`train_cnn`):
```
Creating new model.
train error, validation error, learning rate
0.22387, 0.08833, 0.00500
0.10271, 0.08806, 0.00500
0.08255, 0.05924, 0.00500
0.07669, 0.05936, 0.00500
0.06736, 0.05972, 0.00500
0.06029, 0.05364, 0.00500
0.06054, 0.05408, 0.00500
0.05498, 0.06120, 0.00500
0.05740, 0.06391, 0.00500
0.05614, 0.05674, 0.00500
0.05196, 0.06034, 0.00500
No validation improvement for 5 epochs (best: 0.05364) -- stopping early.
Final (last epoch) train error: 0.051964, validation error: 0.060338
Best validation error (the model saved to disk): 0.053642
Training epochs: 11
```

`test model_cnn.txt`:
```
Train: 55544/56000 = 99.19%
Test : 6894/7000 = 98.49%
```

`predict model_cnn.txt` (per-class probability for one sample digit):
```
0: 0.00104
1: 0.00006
2: 0.00126
3: 0.00078
4: 0.00013
5: 0.00020
6: 0.00405
7: 0.00001
8: 0.99149
9: 0.00099
```

Training log (`train_fc`):
```
Creating new model.
train error, validation error, learning rate
0.26005, 0.15315, 0.00500
0.10576, 0.12386, 0.00500
0.07074, 0.11354, 0.00500
0.04833, 0.11842, 0.00500
0.03595, 0.12634, 0.00500
0.02503, 0.11943, 0.00500
0.01891, 0.11780, 0.00500
0.01401, 0.11388, 0.00500
No validation improvement for 5 epochs (best: 0.11354) -- stopping early.
Final (last epoch) train error: 0.014007, validation error: 0.113876
Best validation error (the model saved to disk): 0.113543
Training epochs: 8
```

`test model_fc.txt`:
```
Train: 54932/56000 = 98.09%
Test : 6729/7000 = 96.13%
```
