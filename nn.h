/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NN_H
#define NN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// NN API Version
#define NN_VERSION_MAJOR 0
#define NN_VERSION_MINOR 1
#define NN_VERSION_PATCH 8
#define NN_VERSION_BUILD 0

typedef enum {
  NN_ERROR_NONE = 0,                    // No error
  NN_ERROR_INVALID_ARGUMENT = -1,       // Invalid argument passed to function
  NN_ERROR_OUT_OF_MEMORY = -2,          // Out of memory
  NN_ERROR_FILE_NOT_FOUND = -3,         // File not found
  NN_ERROR_FILE_READ = -4,              // Error reading file
  NN_ERROR_FILE_WRITE = -5,             // Error writing file
  NN_ERROR_UNSUPPORTED_LAYER = -6,      // Unsupported layer type
  NN_ERROR_UNSUPPORTED_ACTIVATION = -7, // Unsupported activation function
  NN_ERROR_UNSUPPORTED_POOLING = -8,    // Unsupported pooling type
  NN_ERROR_INVALID_LAYER = -9,          // Invalid layer index
  NN_ERROR_INVALID_ACTIVATION = -10,    // Invalid activation function index
  NN_ERROR_INVALID_POOLING = -11,       // Invalid pooling type index
  NN_ERROR_INVALID_INIT = -12,          // Invalid initialization type
  NN_ERROR_INVALID_CONFIG = -13,        // Invalid configuration for layer
  NN_ERROR_INVALID_VERSION = -14,       // Invalid version of the neural network model
  NN_ERROR_QUANTIZATION = -15,          // Error during quantization
  NN_ERROR_DEQUANTIZATION = -16,        // Error during dequantization
  NN_ERROR_TRAINING = -17,              // Error during training
  NN_ERROR_PREDICTION = -18,            // Error during prediction
  NN_ERROR_UNKNOWN = -19,               // Unknown error
  NN_ERROR_READ_ONLY_MODEL = -20        // Attempted to mutate a model loaded with nn_load_model_inplace()
} nn_error_t;

typedef enum {
  ACTIVATION_FUNCTION_TYPE_NONE = 0,
  ACTIVATION_FUNCTION_TYPE_LINEAR,
  ACTIVATION_FUNCTION_TYPE_RELU,
  ACTIVATION_FUNCTION_TYPE_LEAKY_RELU,
  ACTIVATION_FUNCTION_TYPE_ELU,
  ACTIVATION_FUNCTION_TYPE_THRESHOLD,
  ACTIVATION_FUNCTION_TYPE_SIGMOID,
  ACTIVATION_FUNCTION_TYPE_SIGMOID_FAST,
  ACTIVATION_FUNCTION_TYPE_TANH,
  ACTIVATION_FUNCTION_TYPE_TANH_FAST,
  ACTIVATION_FUNCTION_TYPE_GELU,
  ACTIVATION_FUNCTION_TYPE_SILU,
  // Valid only on LAYER_TYPE_OUTPUT (nn_add_layer() rejects it elsewhere):
  // unlike every activation above, softmax is not a per-neuron function of
  // its own preact -- each output depends on every neuron's preact in the
  // layer -- so forward_propagation() computes it as a dedicated two-pass
  // whole-layer normalization instead of through the activation_function[]
  // table, and nn_train()/nn_error() switch to cross-entropy loss for the
  // output layer's error/gradient (the two are almost always paired: their
  // combined gradient collapses to a simple `target - prediction`, unlike
  // softmax paired with any other loss).
  ACTIVATION_FUNCTION_TYPE_SOFTMAX
} activation_function_type_t;

typedef enum {
  LAYER_TYPE_NONE = 0,
  LAYER_TYPE_FC,         // Fully Connected Network Layer
  LAYER_TYPE_CNN,        // Convolutional Neural Network Layer
  LAYER_TYPE_POOL,       // Pooling Layer
  // Long Short-Term Memory Layer. Like LAYER_TYPE_RNN, width (the number of
  // hidden units) is given directly via nn_add_layer()'s `width` argument,
  // `config` is unused (pass NULL), and this layer processes one timestep
  // per nn_train()/nn_predict()/nn_error() call, carrying state across
  // calls -- call nn_reset_state() before a new, independent sequence.
  // Unlike RNN, it carries TWO persistent state vectors: nn->neuron[layer]
  // (the hidden state h, exactly as for RNN) and nn->lstm_cell[layer] (the
  // cell state c, this layer's longer-lived memory, gated rather than
  // overwritten each step -- what makes LSTM more resistant than a plain
  // RNN to vanishing gradients over longer sequences). Each of its four
  // gates (input, forget, cell-candidate, output) has a fixed nonlinearity
  // (sigmoid for input/forget/output, tanh for the cell candidate) rather
  // than a user-selected one -- nn->activation[layer] is stored for this
  // layer (and round-tripped through save/load) but not used. Also like
  // RNN, the recurrent connection is trained as truncated BPTT with a
  // truncation depth of 1 (see LAYER_TYPE_RNN's comment for the rationale);
  // an LSTM's output is not a scalar function of one preact per neuron the
  // way every other layer type's is, though, so its backward pass
  // (nn_lstm_backward() in nn.c) replaces the generic per-neuron
  // activation-derivative multiply every other layer type's backprop uses.
  // Because of that, LAYER_TYPE_LSTM should not be used as the network's
  // final layer -- follow it with a normal FC/OUTPUT layer, the same way
  // the RNN gesture-classification example
  // (examples/gesture_recognition/train.c) follows its RNN layer with an
  // OUTPUT layer. See the comment above forward_propagation()'s
  // LAYER_TYPE_LSTM case in nn.c for the weight layout and full rationale.
  LAYER_TYPE_LSTM,
  // Gated Recurrent Unit Layer. Like LAYER_TYPE_LSTM (one timestep per call,
  // `config` unused, width set directly, nn_reset_state() between
  // sequences, truncated BPTT depth 1, fixed gate nonlinearities not
  // user-selected), but a cheaper middle point between LAYER_TYPE_RNN and
  // LAYER_TYPE_LSTM: only THREE gates (reset, update, candidate) instead of
  // four, and only ONE persistent state -- nn->neuron[layer], the hidden
  // state h, same as RNN -- instead of LSTM's two. The update gate alone
  // does the job LSTM splits across separate input/forget gates (how much
  // of the new candidate to blend in vs. how much of the old state to
  // keep), which is why GRU needs fewer weights and less compute than LSTM
  // while still getting most of the same "hold a signal without it decaying"
  // benefit over a plain RNN. Like LSTM, its backward pass
  // (nn_gru_backward() in nn.c) replaces the generic per-neuron
  // activation-derivative multiply every other layer type's backprop uses,
  // so LAYER_TYPE_GRU should not be used as the network's final layer
  // either. See the comment above forward_propagation()'s LAYER_TYPE_GRU
  // case in nn.c for the weight layout, gate equations, and full rationale.
  LAYER_TYPE_GRU,
  // Recurrent (Elman) Neural Network Layer. Width (the number of hidden
  // units) is given directly via nn_add_layer()'s `width` argument, same as
  // LAYER_TYPE_FC; `config` is unused (pass NULL). Unlike every other layer
  // type, this layer is stateful and processes one timestep per
  // nn_train()/nn_predict()/nn_error() call: nn->neuron[layer] holds this
  // layer's hidden state and persists across calls (each call reads it as
  // the *previous* timestep's state before overwriting it with the new
  // one), so a caller runs a whole sequence by calling once per timestep.
  // Call nn_reset_state() before starting a new, independent sequence --
  // otherwise the previous sequence's final hidden state leaks into the
  // next one. The recurrent connection is trained as truncated BPTT with a
  // truncation depth of 1: the previous hidden state is treated as a
  // constant for gradient purposes (like any earlier layer's activations),
  // and no gradient flows further back through time than one step. This
  // keeps memory use flat regardless of sequence length, matching this
  // library's embedded-systems, one-sample-at-a-time training model. See
  // the comment above forward_propagation()'s LAYER_TYPE_RNN case in nn.c
  // for the weight layout and full rationale.
  LAYER_TYPE_RNN,
  LAYER_TYPE_ATTENTION,  // Attention Layer - Not yet implemented
  LAYER_TYPE_TRANSFORMER,// Transformer Layer - Not yet implemented
  LAYER_TYPE_INPUT,      // Input Layer
  LAYER_TYPE_OUTPUT,     // Output Layer
  LAYER_TYPE_DROPOUT     // Dropout Layer (training-only regularization; a no-op pass-through at inference)
} layer_type_t;

typedef enum {
  POOLING_TYPE_NONE = 0,
  POOLING_TYPE_MIN,
  POOLING_TYPE_MAX,
  POOLING_TYPE_AVG,
} pooling_type_t;

// Selects how nn_train() turns a computed gradient (weight_adj[layer]/a
// bias's own gradient) into a weight/bias update -- see nn_set_optimizer().
typedef enum {
  NN_OPTIMIZER_SGD = 0,  // param += grad * rate. No persistent state; today's original behavior.
  NN_OPTIMIZER_MOMENTUM, // Classic momentum: one persistent "velocity" accumulator per weight/bias.
  NN_OPTIMIZER_ADAM,     // Adam: two persistent accumulators (first/second moment) per weight/bias.
} nn_optimizer_t;

typedef enum {
  NN_INIT_NONE = 0,      // No initialization
  NN_INIT_ZEROS,         // Initialize weights to zero
  NN_INIT_ONES,          // Initialize weights to one
  NN_INIT_RANDOM,        // Random initialization
  NN_INIT_XAVIER,        // Xavier initialization
  NN_INIT_HE,            // He initialization
} nn_init_t;

typedef struct {
  uint16_t in_h;         // Input height
  uint16_t in_w;         // Input width
  uint8_t in_channels;  // # Input feature maps
  uint8_t out_channels; // # Output feature maps
  uint8_t kernel_size;   // Kernel width and height (square)
  uint8_t stride;        // Stride
  uint8_t padding;       // Padding (same for all sides)
  uint8_t dilation;      // Dilation (same for both dimensions)
  nn_init_t weight_init; // How to initialize weights
  nn_init_t bias_init;   // How to initialize biases
  uint16_t out_h;        // Cached output height, computed once by nn_add_layer -- do not set this yourself
  uint16_t out_w;        // Cached output width, computed once by nn_add_layer -- do not set this yourself
} cnn_t;

typedef struct {
  uint16_t in_h;         // Input height (must match the previous layer's output geometry)
  uint16_t in_w;         // Input width
  uint8_t channels;      // # feature-map channels (must match the previous layer's # output channels)
  uint8_t pool_size;     // Pooling window width and height (square)
  uint8_t stride;        // Stride
  pooling_type_t pooling_type; // POOLING_TYPE_MIN, POOLING_TYPE_MAX, or POOLING_TYPE_AVG
  uint16_t out_h;        // Cached output height, computed once by nn_add_layer -- do not set this yourself
  uint16_t out_w;        // Cached output width, computed once by nn_add_layer -- do not set this yourself
} pool_t;

typedef struct {
  // Probability, in [0, 1), of dropping (zeroing) each unit during training.
  // Width is derived automatically from the previous layer (a DROPOUT layer
  // is a same-width pass-through) -- the `width` argument passed to
  // nn_add_layer() for a DROPOUT layer is ignored, same as for CNN/POOL.
  // Like a POOL layer, a DROPOUT layer has no activation function of its
  // own: add it with ACTIVATION_FUNCTION_TYPE_LINEAR (see nn_dropout_forward()
  // in nn.c).
  float rate;
} dropout_t;

typedef struct {
  bool quantized;         // Indicates if the network is quantized
  uint8_t version_major;  // Major version of the network model
  uint8_t version_minor;  // Minor version of the network model
  uint8_t version_patch;  // Patch level of the network model
  uint8_t version_build;  // Build number of the network model
  uint32_t depth;         // Number of layers, including the input and the output layers
  uint32_t *width;        // Number of neurons in each layer (can vary from layer to layer)
  uint8_t *layer_type;    // Type of each layer
  uint8_t *activation;    // Activation function used for each layer
  void **config;          // Configuration for each layer (ex: CNN parameters)
  float **neuron;         // Output value for each neuron in each layer
  float **loss;           // Error derivative for each neuron in each layer
  float **preact;         // Neuron values before activation function is applied for each neuron in each layer
  float **weight_scale;   // Scale for each weight-row (neuron or CNN kernel) in each layer
  // weight/weight_quantized/weight_adj: one contiguous, row-major buffer per
  // layer (weight[layer][row * row_len + col]), not an array of separately
  // allocated rows -- better cache locality/vectorization than a jagged
  // array, and far fewer allocations. "row"/"row_len" mean neuron/prev-width
  // for FC & OUTPUT layers, or kernel-index/kernel_size^2 for CNN layers
  // (see the private quantized_layer_shape() helper in nn.c, which every
  // internal user of these fields calls to get the right row/row_len/count
  // for a given layer).
  float **weight;             // Weight for each neuron/kernel in each layer
  int8_t **weight_quantized;  // Quantized weight for each neuron/kernel in each layer
  float **weight_adj;         // Adjustment of each weight for each neuron/kernel in each layer
  float *bias_scale;      // Scale for each bias in each layer
  float **bias;           // Bias for each neuron
  int8_t **bias_quantized;// Quantized bias for each neuron
  int **pool_argmax;      // Per POOL-MAX/MIN layer: winning input index for each output neuron (NULL otherwise)
  // Per DROPOUT layer only (NULL otherwise): the per-neuron scale applied to
  // that neuron by the most recent training forward pass -- either 0.0f (this
  // neuron was dropped) or 1/(1-rate) (kept, inverted-dropout scaling).
  // Written by nn_dropout_forward() and consumed by nn_dropout_backward() to
  // route the same per-neuron scaling through the gradient; both are only
  // exercised in training mode (see forward_propagation()'s `training` flag
  // in nn.c) -- at inference a DROPOUT layer is a pure pass-through and this
  // is never populated or read.
  float **dropout_scale;
  // One bool per layer (only meaningful for a DROPOUT layer; unused
  // elsewhere, same convention as bias_scale). False until
  // nn_dropout_forward() samples a mask into dropout_scale[layer] above;
  // nn_reset_state() clears it back to false. In a network with no RNN/GRU/
  // LSTM layer, nn_dropout_forward() ignores this and resamples every
  // training call regardless (ordinary per-sample dropout). In a network
  // that DOES have one -- where every nn_train()/nn_predict()/nn_error()
  // call is one timestep of an ongoing sequence, not an independent sample
  // -- nn_dropout_forward() instead holds the same mask for the whole
  // sequence once this is true, only resampling right after
  // nn_reset_state() (the same call that already marks a new sequence
  // starting for RNN/GRU/LSTM state) sets it back to false. Without this,
  // dropout would inject a different random mask into a recurrent layer's
  // hidden state on every single frame instead of cleanly regularizing it.
  bool *dropout_mask_valid;
  // Per RNN layer only (NULL otherwise): a copy of this layer's hidden state
  // (nn->neuron[layer]) as it was just *before* the most recent forward
  // pass overwrote it with the new timestep's state. forward_propagation()
  // writes this every call (training or not); nn_train() reads it to
  // compute the recurrent weight's gradient, since by the time backprop
  // runs, nn->neuron[layer] itself already holds the new (not previous)
  // state. See the comment above forward_propagation()'s LAYER_TYPE_RNN
  // case in nn.c.
  float **rnn_hidden_prev;
  // Per LSTM layer only (NULL otherwise): this layer's cell state c, a
  // second persistent state vector alongside nn->neuron[layer]'s hidden
  // state h (see LAYER_TYPE_LSTM's comment above). Zeroed by nn_add_layer()/
  // nn_load_model_inplace()/nn_reset_state(); updated in place by
  // forward_propagation()'s LAYER_TYPE_LSTM case every call.
  float **lstm_cell;
  // Per LSTM layer only (NULL otherwise): forward_propagation() caches this
  // timestep's previous cell/hidden state and every gate's activation here
  // for nn_train()'s backward pass (nn_lstm_backward() in nn.c), which runs
  // after forward_propagation() already overwrote lstm_cell[layer]/
  // neuron[layer] with the new state -- same reason rnn_hidden_prev exists
  // for LAYER_TYPE_RNN. Each layer's buffer is 7 * width[layer] floats,
  // laid out as seven width[layer]-wide segments: cell_prev, hidden_prev,
  // then the input/forget/cell-candidate/output gates' activations (i.e.
  // already through their sigmoid/tanh nonlinearity), then tanh(this
  // timestep's cell state) -- see the comment above
  // forward_propagation()'s LAYER_TYPE_LSTM case in nn.c for the exact
  // offsets.
  float **lstm_cache;
  // Per LSTM layer only (NULL otherwise): the four gates' preact-space
  // gradients computed by nn_lstm_backward(), one width[layer]-wide segment
  // per gate in the same input/forget/cell-candidate/output order as this
  // layer's weight rows (see quantized_layer_shape()) -- 4 * width[layer]
  // floats total. This is what nn_train()'s weight/bias-update loops read
  // for an LSTM layer instead of nn->loss[layer] (which, for an LSTM layer
  // only, is repurposed as scratch space by the backprop loop rather than
  // holding a per-neuron loss value -- see nn_train()'s LAYER_TYPE_LSTM
  // handling in nn.c).
  float **lstm_gate_grad;
  // Per GRU layer only (NULL otherwise): forward_propagation() caches this
  // timestep's previous hidden state and each gate's activation here for
  // nn_train()'s backward pass (nn_gru_backward() in nn.c), which runs
  // after forward_propagation() already overwrote neuron[layer] with the
  // new state -- same reason lstm_cache exists for LAYER_TYPE_LSTM. Each
  // layer's buffer is 5 * width[layer] floats, laid out as five
  // width[layer]-wide segments: hidden_prev, then the reset/update/
  // candidate gates' activations (already through their sigmoid/tanh
  // nonlinearity), then the candidate gate's raw recurrent contribution
  // (before the reset gate multiplies it -- needed to compute the reset
  // gate's own gradient) -- see the comment above forward_propagation()'s
  // LAYER_TYPE_GRU case in nn.c for the exact offsets. Unlike LSTM, GRU has
  // only one persistent state (neuron[layer], the hidden state h), so there
  // is no separate "gru_cell" field the way there's an lstm_cell.
  float **gru_cache;
  // Per GRU layer only (NULL otherwise): the three gates' preact-space
  // gradients computed by nn_gru_backward(), one width[layer]-wide segment
  // per gate in the same reset/update/candidate order as this layer's
  // weight rows (see quantized_layer_shape()) -- 3 * width[layer] floats
  // total. Read by nn_train()'s weight/bias-update loops for a GRU layer
  // instead of nn->loss[layer] (repurposed as scratch space for this layer
  // type, same as for LSTM -- see nn_train()'s LAYER_TYPE_GRU handling in nn.c).
  float **gru_gate_grad;
  // Which optimizer nn_train() uses to turn a gradient into a weight/bias
  // update -- see nn_optimizer_t and nn_set_optimizer(). Defaults to
  // NN_OPTIMIZER_SGD (nn_init()'s zero-value default), matching this
  // library's original behavior exactly with zero extra memory. Never set
  // this directly -- use nn_set_optimizer(), which also (re)allocates the
  // four moment buffers below to match.
  nn_optimizer_t optimizer;
  // Hyperparameters for the optimizer above, set by nn_set_optimizer():
  // `optimizer_momentum` is the velocity decay for NN_OPTIMIZER_MOMENTUM,
  // and Adam's "beta1" for NN_OPTIMIZER_ADAM (the same underlying idea --
  // an exponential moving average of the gradient -- so one field serves
  // both); `optimizer_beta2`/`optimizer_epsilon` are Adam-only. Unused, and
  // left at whatever nn_set_optimizer() last wrote, while optimizer ==
  // NN_OPTIMIZER_SGD.
  float optimizer_momentum;
  float optimizer_beta2;
  float optimizer_epsilon;
  // Adam's step counter (t in the usual Adam bias-correction formulas),
  // incremented once per nn_train() call -- shared across every layer's
  // Adam update within that call, not per-weight. Reset to 0 by
  // nn_set_optimizer() (including switching back to the same optimizer),
  // so a fresh Adam run always gets its own proper warmup. Unused for
  // SGD/MOMENTUM.
  uint32_t adam_step;
  // Per mutable (non-quantized) layer with weights (FC/OUTPUT/CNN/RNN/LSTM/
  // GRU -- NULL for POOL/DROPOUT/INPUT, same layers weight_adj skips): the
  // optimizer's persistent per-weight state, exactly the same shape as
  // weight[layer] (rows*row_len via quantized_layer_shape()). weight_moment1
  // is NN_OPTIMIZER_MOMENTUM's velocity, or NN_OPTIMIZER_ADAM's first
  // moment (m); weight_moment2 is NN_OPTIMIZER_ADAM's second moment (v)
  // only. Both NULL under NN_OPTIMIZER_SGD -- that's the whole point of
  // making the optimizer selectable: a model that never opts into MOMENTUM/
  // ADAM carries none of this extra RAM. Like weight_adj, this state is
  // never written to a saved model file (see nn_save_model_ascii()/
  // nn_save_model_binary()) -- reloading a saved model always restarts the
  // optimizer's accumulators from zero, and quantizing a model
  // (nn_quantize()) always frees them (a quantized model can't train
  // anyway; nn_dequantize() reallocates fresh, zeroed ones if `optimizer`
  // still calls for them). nn_remove_neuron() resets (does not attempt to
  // migrate) the two affected layers' accumulators to zero rather than
  // trying to preserve per-weight history across a reshape.
  float **weight_moment1;
  float **weight_moment2;
  // Same as weight_moment1/weight_moment2 above, but sized to bias_count
  // (one entry per bias, via quantized_layer_shape()) instead of a full
  // weight row -- the bias-side counterpart nn_train()'s bias-update loop
  // reads/writes instead of nn->bias_adj (biases have no separate "adj"
  // scratch array the way weights do; the gradient is applied directly).
  float **bias_moment1;
  float **bias_moment2;
  // Non-NULL only for a model returned by nn_load_model_inplace(): the two
  // activation buffers every non-recurrent layer's neuron[]/preact[] point
  // into, alternating layer by layer, instead of each layer owning its own
  // (preact[layer] == neuron[layer] for those layers). An inference-only
  // forward pass reads nothing but the layer just below the one it's
  // computing, so this is all the RAM it needs. It does mean a hidden
  // layer's neuron[] only holds its own values until two layers later
  // overwrite them; nn_predict()'s returned output stays valid until the
  // next call, as for any model. Never set this yourself.
  float *activation_buffer;
  // True only for a model returned by nn_load_model_inplace(): weight,
  // weight_quantized, weight_scale, and bias/bias_quantized then point
  // directly into the caller's (read-only, e.g. flash-resident) buffer
  // instead of owned heap allocations, and must never be written through.
  // nn_free() checks this to know which per-layer buffers it must NOT free,
  // and nn_train()/nn_quantize()/nn_dequantize()/nn_remove_neuron()/
  // nn_prune_lightest_neuron() check it to refuse to mutate the model at
  // all. Never set this yourself.
  bool immutable;
} nn_t;

uint32_t nn_version(void);
nn_t *nn_init(void);
void nn_free(nn_t *nn);
nn_error_t nn_add_layer(nn_t *nn, layer_type_t layer_type, int width, int activation, void *config);
nn_error_t nn_save_model_ascii(nn_t *nn, const char *path);
nn_error_t nn_save_model_binary(nn_t *nn, const char *path);
nn_t *nn_load_model_ascii(const char *path);
nn_t *nn_load_model_binary(const char *path);
// Loads a model from a binary-format buffer already resident in memory (e.g.
// a model baked into flash as a byte array on a microcontroller with no
// filesystem) instead of from a file. See the comment above its definition
// in nn.c for buffer-lifetime and inference-only usage notes.
nn_t *nn_load_model_memory(const uint8_t *data, size_t size);
// Writes a neural-net model in the "inplace" format (magic "NNP1"): a
// zero-copy-friendly variant of the binary format, meant to be paired with
// nn_load_model_inplace(). See the comment above that function's definition
// in nn.c for the full format layout and rationale.
nn_error_t nn_save_model_inplace(nn_t *nn, const char *path);
// Loads a model from an "inplace"-format buffer with zero-copy weight/bias
// aliasing directly into `data` -- no RAM copy of the (dominant) weight
// arrays -- instead of parsing them into freshly malloc'd storage like
// nn_load_model_memory() does. Intended for microcontroller targets where
// the model lives in flash and RAM is tight; activations, too, take two
// buffers shared by the whole network instead of two per layer (see nn_t's
// activation_buffer). See the comment above its definition in nn.c for
// buffer-lifetime, alignment, and read-only usage requirements.
nn_t *nn_load_model_inplace(const uint8_t *data, size_t size);
// Loads a model from an "inplace"-format buffer the same way
// nn_load_model_inplace() does, except every weight/bias array is copied
// into a freshly owned allocation instead of aliased into `data` -- a
// normal, fully mutable model, usable with nn_train(), nn_quantize(),
// nn_dequantize(), nn_remove_neuron(), and nn_prune_lightest_neuron() (at
// the RAM cost nn_load_model_inplace() exists to avoid). Use this instead
// of nn_load_model_inplace() when you need to modify an inplace-format
// model rather than just run inference against it; `data` only needs to
// stay valid for the duration of this call.
nn_t *nn_load_model_inplace_copy(const uint8_t *data, size_t size);

typedef enum {
  NN_MODEL_FORMAT_UNKNOWN = 0, // Could not open the file
  NN_MODEL_FORMAT_ASCII,
  NN_MODEL_FORMAT_BINARY,      // magic "NNB1" -- nn_load_model_binary()/nn_load_model_memory()
  NN_MODEL_FORMAT_INPLACE,     // magic "NNP1" -- nn_load_model_inplace()
} nn_model_format_t;

// Peeks a model file's first few bytes to report which of the three formats
// (see the "Model File Format" section of the README) it's in, without
// loading the model. A file that opens but doesn't match either binary
// magic number is assumed to be ASCII -- same convention nn_load_model()
// itself uses, so this never disagrees with what nn_load_model() would
// actually load the file as.
nn_model_format_t nn_model_format(const char *path);
// Loads a model file, auto-detecting ascii vs. binary from the binary
// format's magic number -- use this instead of nn_load_model_{ascii,binary}
// when the caller doesn't know (or care) which format a model file is in.
// Note this does not handle the "inplace" format (see nn_model_format()
// above and nn_load_model_inplace()): that format is meant to be read
// directly from a memory buffer (e.g. flash), not opened as a path here.
nn_t *nn_load_model(const char *path);
// Saves a model file, writing binary format if `path` ends in ".bin"
// (case-insensitive) and ascii format otherwise.
nn_error_t nn_save_model(nn_t *nn, const char *path);
float nn_error(nn_t *nn, float *inputs, float *targets);
float nn_train(nn_t *nn, float *inputs, float *targets, float rate);
float *nn_predict(nn_t *nn, float *inputs);
nn_error_t nn_remove_neuron(nn_t *nn, int layer, int neuron_index);
float nn_get_total_neuron_weight(nn_t *nn, int layer, int neuron_index);
bool nn_prune_lightest_neuron(nn_t *nn);
void nn_pool2d(char *src, char *dest, int filter_size, int stride, pooling_type_t pooling_type, int x_in, int y_in);
void nn_conv2d(nn_t *nn, int layer);
void nn_pool_forward(nn_t *nn, int layer);
nn_error_t nn_quantize(nn_t *nn);
nn_error_t nn_dequantize(nn_t *nn);
// Zeros every RNN/LSTM/GRU layer's persistent state -- nn->neuron[layer]
// (the hidden state, for all three) and, for an LSTM layer only,
// nn->lstm_cell[layer] (its cell state) too -- GRU, like RNN, has just the
// one state. Also invalidates any DROPOUT layer's held mask (see
// dropout_mask_valid's comment above) -- in a network with a recurrent
// layer, this is the same "a new, independent sequence starts here"
// boundary dropout needs to refresh at, same as RNN/GRU/LSTM state. Call
// this before feeding the first timestep of a new, independent sequence
// into a model that has an RNN, LSTM, or GRU layer -- otherwise the final
// state left over from whatever sequence was last run through the model
// (or uninitialized-but-zeroed state, for a freshly constructed/loaded
// model) carries over into the new sequence. Safe to call on an immutable
// (nn_load_model_inplace()) model too: it only touches the always-owned
// neuron[]/lstm_cell[]/dropout_mask_valid buffers, never the aliased
// weight/bias arrays.
void nn_reset_state(nn_t *nn);
// Selects the optimizer nn_train() uses from here on, and (re)allocates
// every existing layer's weight_moment1/weight_moment2/bias_moment1/
// bias_moment2 to match it (freeing whichever this optimizer doesn't need,
// zeroing whichever it does) -- see nn_t's optimizer/weight_moment1/etc.
// comments for the full explanation and memory cost of each choice (SGD:
// none; MOMENTUM: +1 float per weight/bias; ADAM: +2 floats per weight/
// bias). Also resets adam_step to 0, so switching optimizers (even back to
// the same one) always starts any new Adam run with a fresh warmup rather
// than picking up a stale step count. `momentum` is the velocity decay for
// MOMENTUM, or Adam's beta1 for ADAM (typical: 0.9); `beta2`/`epsilon` are
// ADAM-only (typical: 0.999/1e-8) and ignored otherwise -- pass 0 for both
// when selecting SGD or MOMENTUM. Call this once, before training (or
// again later to switch), not inside a per-sample training loop. Refuses
// (NN_ERROR_READ_ONLY_MODEL) on an immutable (nn_load_model_inplace())
// model, same as nn_train() itself -- there is nothing to train there.
nn_error_t nn_set_optimizer(nn_t *nn, nn_optimizer_t optimizer, float momentum, float beta2, float epsilon);

#endif /* NN_H */
