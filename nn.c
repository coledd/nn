/*
 * Neural Network library
 * Copyright (c) 2019-2026 Cole Design and Development, LLC
 * https://coledd.com
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ctype.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn.h"

// Private functions

// First 4 bytes of every binary model file, so nn_load_model() can tell a
// binary model file apart from an ascii one by content rather than by
// trusting the file extension.
#define NN_BINARY_MAGIC_LEN 4
static const uint8_t NN_BINARY_MAGIC[NN_BINARY_MAGIC_LEN] = {'N', 'N', 'B', '1'};

// First 4 bytes of every "inplace"-format model (see nn_load_model_inplace()),
// distinct from NN_BINARY_MAGIC above so the two formats can never be
// confused for one another.
#define NN_INPLACE_MAGIC_LEN 4
static const uint8_t NN_INPLACE_MAGIC[NN_INPLACE_MAGIC_LEN] = {'N', 'N', 'P', '1'};

typedef float (*activation_function_t)(float a, bool derivative);

// Null activation function
static float activation_function_none(float a, bool derivative)
{
  return 0;
}

// Linear activation function (aka identity activation function)
static float activation_function_linear(float a, bool derivative)
{
  if (derivative)
    return 1;
  return a;
}

// Rectified Linear Unit (ReLU) activation function
static float activation_function_relu(float a, bool derivative)
{
  if (a >= 0)
    return (derivative ? 1 : a);
  return 0;
}

// Leaky Rectified Linear Unit (Leaky ReLU) activation function
static float activation_function_leaky_relu(float a, bool derivative)
{
  if (a > 0)
    return (derivative ? 1 : a);
  return (derivative ? 0.01f : a * 0.01f);
}

// Exponential Linear Unit (ELU) activation function
static float activation_function_elu(float a, bool derivative)
{
  if (a >= 0)
    return (derivative ? 1 : a);
  // For a < 0, elu(a) = expf(a) - 1, so elu'(a) = expf(a) (not elu(a) itself,
  // i.e. not expf(a) - 1).
  return (derivative ? expf(a) : expf(a) - 1);
}

// Threshold activation function
static float activation_function_threshold(float a, bool derivative)
{
  if (derivative)
    return 0;
  return a > 0;
}

// Sigmoid activation function (aka Logistic, aka Soft Step)
static float activation_function_sigmoid(float a, bool derivative)
{
  if (derivative) {
    float f = activation_function_sigmoid(a, false);
    return (f * (1.0f - f));
  }
  return 1.0f / (1.0f + expf(-a));
}

// Sigmoid activation function using a lookup table
static float activation_function_sigmoid_fast(float a, bool derivative)
{
  // Sigmoid outputs
  const float s[] = {0.0f,      0.000045f, 0.000123f, 0.000335f, 0.000911f,
                     0.002473f, 0.006693f, 0.017986f, 0.047426f, 0.119203f,
                     0.268941f, 0.500000f, 0.731059f, 0.880797f, 0.952574f,
                     0.982014f, 0.993307f, 0.997527f, 0.999089f, 0.999665f,
                     0.999877f, 0.999955f, 1.0f};
  // Derivative of the sigmoid
  const float ds[] = {0.0f,      0.000045f, 0.000123f, 0.000335f, 0.000910f,
                      0.002467f, 0.006648f, 0.017663f, 0.045177f, 0.104994f,
                      0.196612f, 0.250000f, 0.196612f, 0.104994f, 0.045177f,
                      0.017663f, 0.006648f, 0.002466f, 0.000910f, 0.000335f,
                      0.000123f, 0.000045f, 0.0f};
  int index;
  float fraction = 0;

  if (a < -11.0f)
    a = -11.0f;
  else if (a > 11.0f)
    a = 11.0f;
  index = (int)floorf(a) + 11;
  if (index < 0) {
    index = 0;
  } else if (index > 21) {
    index = 21;
  } else {
    fraction = a - floorf(a);
  }
  if (derivative) {
    return ds[index] + (ds[index + 1] - ds[index]) * fraction;
  }
  return s[index] + (s[index + 1] - s[index]) * fraction;
}

// Tanh activation function
static float activation_function_tanh(float a, bool derivative)
{
  if (derivative)
    return 1.0f - activation_function_tanh(a, false) * activation_function_tanh(a, false);
  return (2.0f / (1.0f + expf(-2.0f * a))) - 1.0f;
}

// Fast Tanh activation function
static float activation_function_tanh_fast(float a, bool derivative)
{
  if (derivative)
    return 1.0f / ((1.0f + fabsf(a)) * (1.0f + fabsf(a)));
  return a / (1.0f + fabsf(a));
}

// Gaussian Error Linear Unit (GELU) activation function
static float activation_function_gelu(float a, bool derivative)
{
  const float inv_sqrt2 = 0.70710678118654752440f;
  const float inv_sqrt2pi = 0.39894228040143267794f;
  float cdf = 0.5f * (1.0f + erff(a * inv_sqrt2));
  if (derivative) {
    float pdf = inv_sqrt2pi * expf(-0.5f * a * a);
    return cdf + a * pdf;
  }
  return a * cdf;
}

// Sigmoid Linear Unit (SiLU) activation function (aka Swish)
static float activation_function_silu(float a, bool derivative)
{
  float s = activation_function_sigmoid(a, false);
  if (derivative)
    return s + a * s * (1.0f - s);
  return a * s;
}

// These must be in the same order as the enum activation_function_type
// Softmax cannot be expressed as this table's per-neuron f(a, derivative):
// each output depends on every neuron's preact in the layer, not just its
// own (see forward_propagation()'s dedicated two-pass computation and
// nn_train()'s fused cross-entropy gradient for the actual math). This
// entry only exists so the array's length stays in sync with
// activation_function_type_t; nn_add_layer() restricts
// ACTIVATION_FUNCTION_TYPE_SOFTMAX to LAYER_TYPE_OUTPUT, whose forward/backward
// bypass this table for that entry entirely, so it is never actually called.
static float activation_function_softmax_unreachable(float a, bool derivative)
{
  (void)derivative;
  return a;
}

static activation_function_t activation_function[] = {
    activation_function_none,
    activation_function_linear,
    activation_function_relu,
    activation_function_leaky_relu,
    activation_function_elu,
    activation_function_threshold,
    activation_function_sigmoid,
    activation_function_sigmoid_fast,
    activation_function_tanh,
    activation_function_tanh_fast,
    activation_function_gelu,
    activation_function_silu,
    activation_function_softmax_unreachable};

// Computes the error given a cost function
// The loss function is a basic mean-square error (MSE)
static float error(float a, float b)
{
  return 0.5f * (a - b) * (a - b);
}

// Computes derivative of the error through the derivative of the cost function
static float error_derivative(float a, float b)
{
  return a - b;
}

// Per-class cross-entropy term -target*log(pred), used instead of error()
// for the output layer's reported error whenever its activation is softmax
// (see the comment on ACTIVATION_FUNCTION_TYPE_SOFTMAX in nn.h). `pred` is
// clamped away from 0 first: softmax can legitimately drive a class's
// probability arbitrarily close to (but never exactly) 0, and log(0) is
// -inf, which would turn into a NaN total the moment a target of 0
// multiplies it (the common case for every non-target class in a one-hot
// target vector).
static float cross_entropy_term(float target, float pred)
{
  const float eps = 1e-7f;
  return -target * logf(fmaxf(pred, eps));
}

// Quantization/dequantization treat each layer as a set of "rows" that share
// a single layer-wide bias scale: for a CNN layer a row is a kernel
// (out_channels * in_channels of them, row_len = kernel_size^2 wide), with
// bias_count = out_channels; for FC/OUTPUT layers a row is a neuron
// (width[L] of them, row_len = width[L-1] wide), with bias_count =
// width[L]. An RNN layer's row is also one neuron (width[L] of them), but
// row_len = width[L-1] + width[L]: each row holds that neuron's
// input-to-hidden weights (against the previous layer's width[L-1] outputs)
// immediately followed by its hidden-to-hidden/recurrent weights (against
// this same layer's own width[L] previous-timestep hidden state) --
// concatenated into one flat row so every piece of generic per-row
// machinery below (quantization, save/load, Xavier init, ...) handles RNN
// layers automatically, with no separate weight array of its own. See the
// comment above forward_propagation()'s LAYER_TYPE_RNN case for how the two
// halves of a row are actually used. Pooling and dropout layers have no
// rows at all -- callers must check nn->layer_type[L] against
// LAYER_TYPE_POOL/LAYER_TYPE_DROPOUT themselves before using this.
static void quantized_layer_shape(nn_t *nn, int L, int *rows, int *row_len, int *bias_count)
{
  if (nn->layer_type[L] == LAYER_TYPE_CNN) {
    cnn_t *c = nn->config[L];
    *rows = c->out_channels * c->in_channels;
    *row_len = c->kernel_size * c->kernel_size;
    *bias_count = c->out_channels;
  } else if (nn->layer_type[L] == LAYER_TYPE_RNN) {
    *rows = (int)nn->width[L];
    *row_len = (int)nn->width[L - 1] + (int)nn->width[L];
    *bias_count = (int)nn->width[L];
  } else if (nn->layer_type[L] == LAYER_TYPE_LSTM) {
    // Four gates (input, forget, cell-candidate, output), each with its own
    // row per hidden unit -- see LAYER_TYPE_LSTM's comment in nn.h and
    // forward_propagation()'s LAYER_TYPE_LSTM case for the row layout and
    // gate order. Each row is still row_len_in + hidden wide, exactly like
    // LAYER_TYPE_RNN's single gate.
    *rows = 4 * (int)nn->width[L];
    *row_len = (int)nn->width[L - 1] + (int)nn->width[L];
    *bias_count = 4 * (int)nn->width[L];
  } else if (nn->layer_type[L] == LAYER_TYPE_GRU) {
    // Three gates (reset, update, candidate), each with its own row per
    // hidden unit -- see LAYER_TYPE_GRU's comment in nn.h and
    // forward_propagation()'s LAYER_TYPE_GRU case for the row layout and
    // gate order. Each row is still row_len_in + hidden wide, exactly like
    // LAYER_TYPE_RNN's single gate and LAYER_TYPE_LSTM's four.
    *rows = 3 * (int)nn->width[L];
    *row_len = (int)nn->width[L - 1] + (int)nn->width[L];
    *bias_count = 3 * (int)nn->width[L];
  } else {
    *rows = (int)nn->width[L];
    *row_len = (int)nn->width[L - 1];
    *bias_count = (int)nn->width[L];
  }
}

// (Re)allocates layer `layer`'s four optimizer moment buffers
// (weight_moment1/2, bias_moment1/2) to match nn->optimizer, freeing
// whichever of the four this optimizer doesn't need and zeroing whichever
// it does -- see their comment in nn.h. A no-op (all four end up NULL) for
// NN_OPTIMIZER_SGD, or for a layer type with no weights of its own (POOL/
// DROPOUT). Used by nn_add_layer() (a new layer, if an optimizer needing
// this state is already selected), nn_set_optimizer() (every existing
// layer, when switching optimizers), nn_dequantize() (every layer, since
// nn_quantize() always frees this state first), and nn_remove_neuron()
// (the two reshaped layers -- which resets rather than migrates this state,
// see its own comment). Returns false only on a real allocation failure.
static bool nn_optimizer_alloc_layer(nn_t *nn, int layer)
{
  free(nn->weight_moment1[layer]); nn->weight_moment1[layer] = NULL;
  free(nn->weight_moment2[layer]); nn->weight_moment2[layer] = NULL;
  free(nn->bias_moment1[layer]); nn->bias_moment1[layer] = NULL;
  free(nn->bias_moment2[layer]); nn->bias_moment2[layer] = NULL;
  if (nn->optimizer == NN_OPTIMIZER_SGD)
    return true;
  if (nn->layer_type[layer] == LAYER_TYPE_POOL || nn->layer_type[layer] == LAYER_TYPE_DROPOUT)
    return true; // no weights/bias to optimize
  int rows, row_len, bias_count;
  quantized_layer_shape(nn, layer, &rows, &row_len, &bias_count);
  nn->weight_moment1[layer] = (float *)calloc((size_t)rows * row_len, sizeof(float));
  nn->bias_moment1[layer] = (float *)calloc((size_t)bias_count, sizeof(float));
  if (!nn->weight_moment1[layer] || !nn->bias_moment1[layer])
    return false;
  if (nn->optimizer == NN_OPTIMIZER_ADAM) {
    nn->weight_moment2[layer] = (float *)calloc((size_t)rows * row_len, sizeof(float));
    nn->bias_moment2[layer] = (float *)calloc((size_t)bias_count, sizeof(float));
    if (!nn->weight_moment2[layer] || !nn->bias_moment2[layer])
      return false;
  }
  return true;
}

// Applies one optimizer step to `total` contiguous elements of a weight or
// bias buffer (`param[idx] += f(grad[idx])`, `grad` already carrying
// whatever sign convention makes a plain "+=" move toward lower loss --
// exactly the quantity nn_train() already computes into weight_adj/a bias's
// own gradient today), using whichever of SGD/MOMENTUM/ADAM nn->optimizer
// currently selects. The switch is hoisted out to run once per call (i.e.
// once per layer, not once per weight), same rationale as the quantized/
// float branch hoisting elsewhere in this file. `moment1`/`moment2` are
// that layer's own weight_moment1[i]/weight_moment2[i] (or
// bias_moment1[i]/bias_moment2[i] for a bias update) -- unused (and may be
// NULL) under NN_OPTIMIZER_SGD.
static void nn_optimizer_apply(nn_t *nn, float *param, const float *grad, float *moment1, float *moment2, int total, float rate)
{
  switch (nn->optimizer) {
    case NN_OPTIMIZER_MOMENTUM:
      for (int idx = 0; idx < total; idx++) {
        moment1[idx] = nn->optimizer_momentum * moment1[idx] + grad[idx];
        param[idx] += moment1[idx] * rate;
      }
      break;
    case NN_OPTIMIZER_ADAM: {
      const float beta1 = nn->optimizer_momentum;
      const float beta2 = nn->optimizer_beta2;
      const float epsilon = nn->optimizer_epsilon;
      const float bias_correction1 = 1.0f - powf(beta1, (float)nn->adam_step);
      const float bias_correction2 = 1.0f - powf(beta2, (float)nn->adam_step);
      for (int idx = 0; idx < total; idx++) {
        moment1[idx] = beta1 * moment1[idx] + (1.0f - beta1) * grad[idx];
        moment2[idx] = beta2 * moment2[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
        float m_hat = moment1[idx] / bias_correction1;
        float v_hat = moment2[idx] / bias_correction2;
        param[idx] += rate * m_hat / (sqrtf(v_hat) + epsilon);
      }
      break;
    }
    case NN_OPTIMIZER_SGD:
    default:
      for (int idx = 0; idx < total; idx++)
        param[idx] += grad[idx] * rate;
      break;
  }
}

// 2D pooling layer forward pass over the previous layer's (float) feature
// maps. Supports MIN/MAX ("winner take all", cached for backprop routing)
// and AVG (uniform reduction) pooling. Pooling has no weights/bias and no
// activation function of its own -- preact[layer] simply mirrors neuron[layer]
// so the generic activation-derivative machinery in nn_train() (with
// nn->activation[layer] set to ACTIVATION_FUNCTION_TYPE_LINEAR) is a no-op.
void nn_pool_forward(nn_t *nn, int layer)
{
  pool_t *pool = nn->config[layer];
  const int channels = pool->channels;
  const int psize = pool->pool_size;
  const int stride = pool->stride;
  const int x_out = pool->out_w;
  const int y_out = pool->out_h;
  const int plane_out = x_out * y_out;
  const int plane_in = pool->in_w * pool->in_h;
  // Sanity-check shapes
  if ((channels <= 0) || ((uint32_t)channels * plane_out != nn->width[layer])) {
    fprintf(stderr, "pool2d: inconsistent shape (channels=%d, plane_out=%d, width[%d]=%u)\n", channels, plane_out, layer, nn->width[layer]);
    return;
  }
  const float *in = nn->neuron[layer - 1];
  float *out = nn->neuron[layer];
  float *pre = nn->preact[layer];
  int *argmax = nn->pool_argmax[layer]; // NULL unless MIN/MAX pooling

  for (int c = 0; c < channels; ++c) {
    for (int oy = 0; oy < y_out; ++oy) {
      const int in_y = oy * stride;
      for (int ox = 0; ox < x_out; ++ox) {
        const int in_x = ox * stride;
        const int oidx = c * plane_out + oy * x_out + ox;
        float result = 0.0f;
        int winner = -1;
        switch (pool->pooling_type) {
          case POOLING_TYPE_MAX: {
            float best = -FLT_MAX;
            for (int ky = 0; ky < psize; ++ky) {
              for (int kx = 0; kx < psize; ++kx) {
                int in_idx = c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx);
                if (in[in_idx] > best) {
                  best = in[in_idx];
                  winner = in_idx;
                }
              }
            }
            result = best;
            break;
          }
          case POOLING_TYPE_MIN: {
            float best = FLT_MAX;
            for (int ky = 0; ky < psize; ++ky) {
              for (int kx = 0; kx < psize; ++kx) {
                int in_idx = c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx);
                if (in[in_idx] < best) {
                  best = in[in_idx];
                  winner = in_idx;
                }
              }
            }
            result = best;
            break;
          }
          case POOLING_TYPE_AVG: {
            float sum = 0.0f;
            for (int ky = 0; ky < psize; ++ky)
              for (int kx = 0; kx < psize; ++kx)
                sum += in[c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx)];
            result = sum / (float)(psize * psize);
            break;
          }
          case POOLING_TYPE_NONE:
          default:
            result = 0.0f;
            break;
        }
        out[oidx] = result;
        pre[oidx] = result;
        if (argmax)
          argmax[oidx] = winner;
      }
    }
  }
}

// Routes/distributes a pooling layer's loss (dE/d(pool output), already
// computed in nn->loss[layer]) back into grad_in, which must be zeroed by the
// caller and sized to the pooling layer's input width (i.e. the previous
// layer's width). MIN/MAX pooling route the full gradient to the single
// cached winning input position; AVG pooling distributes it evenly across
// every input position in the window.
static void nn_pool_backward(nn_t *nn, int layer, float *grad_in)
{
  pool_t *pool = nn->config[layer];
  const int channels = pool->channels;
  const int psize = pool->pool_size;
  const int stride = pool->stride;
  const int x_out = pool->out_w;
  const int y_out = pool->out_h;
  const int plane_out = x_out * y_out;
  const int plane_in = pool->in_w * pool->in_h;
  const float *loss = nn->loss[layer];
  const int *argmax = nn->pool_argmax[layer];

  for (int c = 0; c < channels; ++c) {
    for (int oy = 0; oy < y_out; ++oy) {
      const int in_y = oy * stride;
      for (int ox = 0; ox < x_out; ++ox) {
        const int in_x = ox * stride;
        const int oidx = c * plane_out + oy * x_out + ox;
        const float g = loss[oidx];
        switch (pool->pooling_type) {
          case POOLING_TYPE_MAX:
          case POOLING_TYPE_MIN:
            grad_in[argmax[oidx]] += g;
            break;
          case POOLING_TYPE_AVG: {
            const float share = g / (float)(psize * psize);
            for (int ky = 0; ky < psize; ++ky)
              for (int kx = 0; kx < psize; ++kx)
                grad_in[c * plane_in + (in_y + ky) * pool->in_w + (in_x + kx)] += share;
            break;
          }
          case POOLING_TYPE_NONE:
          default:
            break;
        }
      }
    }
  }
}

// Routes a CNN layer's loss (dE/d(preact), already computed in nn->loss[layer])
// back into grad_in, which must be zeroed by the caller and sized to the CNN
// layer's input width (i.e. the previous layer's width) -- same contract as
// nn_pool_backward() above. This is the transpose of nn_conv2d()'s forward
// pass: each output position's loss is scattered back across the input
// positions its kernel window read from, weighted by that same kernel.
// Padding is handled the same way nn_conv2d() handles it -- ky/kx are
// clipped to the sub-range of the kernel that overlaps a real (unpadded)
// input position, so out-of-bounds (padding) positions are simply never
// written to grad_in instead of needing an actual padded buffer.
static void nn_conv_backward(nn_t *nn, int layer, float *grad_in)
{
  cnn_t *cnn = nn->config[layer];
  const int in_c = cnn->in_channels;
  const int out_c = cnn->out_channels;
  const int ksize = cnn->kernel_size;
  const int x_out = cnn->out_w;
  const int y_out = cnn->out_h;
  const int plane_out = x_out * y_out;
  const int in_plane = cnn->in_w * cnn->in_h;
  const int row_len = ksize * ksize;

  for (int oc = 0; oc < out_c; ++oc) {
    const float *loss = nn->loss[layer] + oc * plane_out;
    for (int oy = 0; oy < y_out; ++oy) {
      const int in_y0 = oy * cnn->stride - cnn->padding;
      const int ky_start = in_y0 < 0 ? -in_y0 : 0;
      const int ky_end = (in_y0 + ksize > cnn->in_h) ? (cnn->in_h - in_y0) : ksize;
      for (int ox = 0; ox < x_out; ++ox) {
        const int in_x0 = ox * cnn->stride - cnn->padding;
        const int kx_start = in_x0 < 0 ? -in_x0 : 0;
        const int kx_end = (in_x0 + ksize > cnn->in_w) ? (cnn->in_w - in_x0) : ksize;
        const float delta = loss[oy * x_out + ox];
        for (int ic = 0; ic < in_c; ++ic) {
          const float *kptr = nn->weight[layer] + (oc * in_c + ic) * row_len;
          float *dst_base = grad_in + ic * in_plane;
          for (int ky = ky_start; ky < ky_end; ++ky) {
            float *drow = dst_base + (in_y0 + ky) * cnn->in_w + in_x0;
            const float *wrow = kptr + ky * ksize;
            for (int kx = kx_start; kx < kx_end; ++kx)
              drow[kx] += delta * wrow[kx];
          }
        }
      }
    }
  }
}

// True if `nn` contains a stateful recurrent layer (RNN/GRU/LSTM) anywhere
// in its stack. Those layer types always process one timestep per
// nn_train()/nn_predict()/nn_error() call (see their comments in nn.h), a
// property of the whole network rather than of any one layer's position --
// nn_dropout_forward() below uses this to decide whether it's regularizing
// an independent sample (ordinary dropout: resample every call) or one
// frame of an ongoing sequence (hold one mask per sequence instead). depth
// is small in practice (a handful of layers), so this cheap O(depth) scan
// on every dropout forward call isn't worth caching.
static bool nn_is_recurrent_layer_type(uint8_t layer_type)
{
  return layer_type == LAYER_TYPE_RNN || layer_type == LAYER_TYPE_GRU || layer_type == LAYER_TYPE_LSTM;
}

static bool nn_has_recurrent_layer(const nn_t *nn)
{
  for (int layer = 1; layer < (int)nn->depth; layer++) {
    if (nn_is_recurrent_layer_type(nn->layer_type[layer]))
      return true;
  }
  return false;
}

// Dropout layer forward pass: a same-width pass-through of the previous
// layer's output. At inference (training == false) every unit simply
// passes through unchanged. During training, each unit is independently
// zeroed with probability `rate` (nn->config[layer]->rate); a surviving
// unit is scaled by 1/(1-rate) ("inverted dropout", the standard
// convention -- it keeps the expected output magnitude the same as at
// inference, so nothing needs adjusting when dropout is later turned off).
// The per-neuron scale actually applied (0 or 1/(1-rate)) is cached in
// nn->dropout_scale[layer] for nn_dropout_backward() to reuse.
// Like nn_pool_forward(), preact[layer] simply mirrors neuron[layer]
// (dropout has no activation function of its own) -- add a DROPOUT layer
// with ACTIVATION_FUNCTION_TYPE_LINEAR so the generic activation-derivative
// machinery in nn_train() is a no-op.
static void nn_dropout_forward(nn_t *nn, int layer, bool training)
{
  const int width = (int)nn->width[layer];
  const float *in = nn->neuron[layer - 1];
  float *out = nn->neuron[layer];
  float *pre = nn->preact[layer];

  if (!training) {
    memcpy(out, in, (size_t)width * sizeof(float));
    memcpy(pre, in, (size_t)width * sizeof(float));
    return;
  }
  // In a network with a recurrent layer, every call is one timestep of an
  // ongoing sequence, not an independent sample -- hold the same mask
  // already cached in dropout_scale[layer] for the whole sequence instead
  // of drawing a fresh one every timestep, which would inject a different
  // random mask into the recurrent layer's hidden state on every single
  // frame rather than cleanly regularizing it. dropout_mask_valid[layer]
  // is invalidated by nn_reset_state(), the same call that already marks a
  // new, independent sequence starting for RNN/GRU/LSTM state -- see its
  // comment in nn.h. A network with no recurrent layer never takes this
  // branch (nn_has_recurrent_layer() is false), so this is a no-op change
  // for a purely feedforward network's dropout, which still resamples
  // every call exactly as before.
  if (nn->dropout_mask_valid[layer] && nn_has_recurrent_layer(nn)) {
    const float *scale = nn->dropout_scale[layer];
    for (int j = 0; j < width; j++) {
      out[j] = in[j] * scale[j];
      pre[j] = out[j];
    }
    return;
  }
  dropout_t *dropout = nn->config[layer];
  const float rate = dropout->rate;
  const float inv_keep = 1.0f / (1.0f - rate); // rate < 1 is enforced by nn_add_layer()
  float *scale = nn->dropout_scale[layer];
  for (int j = 0; j < width; ++j) {
    const bool keep = (rand() / (float)RAND_MAX) >= rate;
    scale[j] = keep ? inv_keep : 0.0f;
    out[j] = in[j] * scale[j];
    pre[j] = out[j];
  }
  nn->dropout_mask_valid[layer] = true;
}

// Routes a dropout layer's loss (dE/d(preact), already computed in
// nn->loss[layer]) back into grad_in, which must be zeroed by the caller and
// sized to the dropout layer's input width (same contract as
// nn_pool_backward()/nn_conv_backward() above). Since dropout is an
// elementwise, same-width pass-through, this simply re-applies the same
// per-neuron scale nn_dropout_forward() cached for this forward pass: a
// dropped unit (scale 0) blocks its gradient entirely, and a surviving unit
// passes its gradient through scaled by 1/(1-rate), consistent with the
// forward multiply.
static void nn_dropout_backward(nn_t *nn, int layer, float *grad_in)
{
  const int width = (int)nn->width[layer];
  const float *loss = nn->loss[layer];
  const float *scale = nn->dropout_scale[layer];
  for (int j = 0; j < width; ++j)
    grad_in[j] += loss[j] * scale[j];
}

// Backward pass for one LSTM layer's own internal gates, given `dh` -- the
// already-gathered incoming gradient dL/dh_t (dh[j] = dE/d(neuron[layer][j]),
// this timestep's hidden-state output), routed here exactly the way the
// generic case in nn_train()'s backprop loop computes `sum` before applying
// its trailing activation-derivative multiply. An LSTM's output is not a
// scalar function of one preact per neuron the way every other layer
// type's is (h_t = o_t * tanh(c_t), itself built from four gates), so
// there's no single "own activation derivative" to multiply by -- this
// function IS that missing piece, computing all four gates' preact-space
// gradients (written into nn->lstm_gate_grad[layer], one width[layer]-wide
// segment per gate in input/forget/cell-candidate/output order) from dh,
// this timestep's cached gate values (nn->lstm_cache[layer]), and its
// just-committed cell state.
//
// Truncated BPTT, depth 1 (same simplification as LAYER_TYPE_RNN -- see its
// comment in nn.h): only this timestep's own local contribution to
// dL/d(gate preacts) is computed here; nn->lstm_cache[layer]'s cached
// cell_prev/hidden_prev are read as given constants, and no gradient is
// propagated into the previous timestep's cell or hidden state.
static void nn_lstm_backward(nn_t *nn, int layer, const float *dh)
{
  const int hidden = (int)nn->width[layer];
  const float *cache = nn->lstm_cache[layer];
  const float *c_prev = cache + 0 * hidden;
  const float *gate_i = cache + 2 * hidden;
  const float *gate_f = cache + 3 * hidden;
  const float *gate_g = cache + 4 * hidden;
  const float *gate_o = cache + 5 * hidden;
  const float *tanh_c = cache + 6 * hidden;
  float *grad = nn->lstm_gate_grad[layer]; // segments: [0,h)=i, [h,2h)=f, [2h,3h)=g, [3h,4h)=o

  for (int j = 0; j < hidden; j++) {
    // dh's contribution to the cell state, through h_t = o_t * tanh(c_t).
    // No additional term from a later timestep's cell gradient -- that's
    // exactly the truncation described above.
    float dc = dh[j] * gate_o[j] * (1.0f - tanh_c[j] * tanh_c[j]);
    grad[0 * hidden + j] = dc * gate_g[j] * gate_i[j] * (1.0f - gate_i[j]);       // d(input gate preact)
    grad[1 * hidden + j] = dc * c_prev[j] * gate_f[j] * (1.0f - gate_f[j]);       // d(forget gate preact)
    grad[2 * hidden + j] = dc * gate_i[j] * (1.0f - gate_g[j] * gate_g[j]);       // d(cell-candidate preact)
    grad[3 * hidden + j] = dh[j] * tanh_c[j] * gate_o[j] * (1.0f - gate_o[j]);    // d(output gate preact)
  }
}

// Backward pass for one GRU layer's own internal gates, given `dh` -- the
// already-gathered incoming gradient dL/dh_t -- the same way
// nn_lstm_backward() is used (see its comment for the general rationale;
// this is that same missing piece for GRU's h_t = (1-z_t)*h_prev + z_t*n_t).
// Writes the three gates' preact-space gradients into
// nn->gru_gate_grad[layer] (segments: [0,h)=reset, [h,2h)=update,
// [2h,3h)=candidate), using this timestep's cached gate values
// (nn->gru_cache[layer]).
//
// The candidate gate's recurrent contribution is reset-gated (n_t =
// tanh(W_n x_t + r_t * (U_n h_prev) + b_n)), which is why
// nn->gru_cache[layer] also caches n_recur_raw (U_n h_prev, before the
// reset-gate multiply) -- it's what the reset gate's own gradient (dr_t =
// dn_preact * n_recur_raw) is computed from, and nn_train()'s
// weight-adjustment loop needs r_t itself (also cached) to correctly scale
// the candidate gate's recurrent weight gradients.
//
// Truncated BPTT, depth 1 (same simplification as LAYER_TYPE_RNN/LSTM --
// see their comments in nn.h): only this timestep's own local contribution
// is computed; nn->gru_cache[layer]'s cached hidden_prev is read as a given
// constant, and no gradient is propagated into the previous timestep.
static void nn_gru_backward(nn_t *nn, int layer, const float *dh)
{
  const int hidden = (int)nn->width[layer];
  const float *cache = nn->gru_cache[layer];
  const float *h_prev = cache + 0 * hidden;
  const float *gate_r = cache + 1 * hidden;
  const float *gate_z = cache + 2 * hidden;
  const float *gate_n = cache + 3 * hidden;
  const float *n_recur_raw = cache + 4 * hidden;
  float *grad = nn->gru_gate_grad[layer]; // segments: [0,h)=reset, [h,2h)=update, [2h,3h)=candidate

  for (int j = 0; j < hidden; j++) {
    float dz = dh[j] * (gate_n[j] - h_prev[j]);                      // dL/d(update gate output)
    float dn = dh[j] * gate_z[j];                                    // dL/d(candidate output)
    float dn_preact = dn * (1.0f - gate_n[j] * gate_n[j]);           // through candidate's tanh
    float dr = dn_preact * n_recur_raw[j];                           // dL/d(reset gate output), via n's r-gating
    grad[0 * hidden + j] = dr * gate_r[j] * (1.0f - gate_r[j]);      // d(reset gate preact)
    grad[1 * hidden + j] = dz * gate_z[j] * (1.0f - gate_z[j]);      // d(update gate preact)
    grad[2 * hidden + j] = dn_preact;                                // d(candidate preact)
  }
}

// `training` selects DROPOUT layer behavior: true (only from nn_train())
// randomly zeroes units (inverted-dropout scaling on the survivors) and
// caches the per-neuron scale used in nn->dropout_scale for the backward
// pass; false (from nn_predict()/nn_error()) makes every DROPOUT layer a
// pure pass-through, as is standard at inference.
static void forward_propagation(nn_t *nn, bool training)
{
  float sum;
  int i, j, k;

  // Calculate neuron values in each layer
  for (i = 1; i < (int)nn->depth; i++) {
    switch(nn->layer_type[i]) {
      case LAYER_TYPE_FC:
      case LAYER_TYPE_OUTPUT:
        // Fully Connected Layer. The quantized/float branch and the
        // per-neuron weight_scale multiply are hoisted out of the innermost
        // dot-product loop (checked/applied once per neuron rather than
        // once per weight) so the accumulation loop is a plain, easily
        // vectorized float multiply-add either way.
        {
          const int row_len = (int)nn->width[i - 1]; // flat weight buffer stride for this layer
          const int width_i = (int)nn->width[i];
          // First pass: compute every neuron's preact (dot product + bias)
          // for this layer, same as before. The activation itself is
          // applied in a second pass below instead of inline here, because
          // softmax (unlike every other activation) needs every neuron's
          // preact already computed before it can normalize any one of
          // them -- see the comment on ACTIVATION_FUNCTION_TYPE_SOFTMAX in nn.h.
          if (nn->quantized) {
            for (j = 0; j < width_i; j++) {
              sum = 0.0f;
              const int8_t *wrow = nn->weight_quantized[i] + j * row_len;
              for (k = 0; k < row_len; k++) {
                sum += nn->neuron[i - 1][k] * (float)wrow[k];
              }
              nn->preact[i][j] = sum * nn->weight_scale[i][j] + (float)nn->bias_quantized[i][j] * nn->bias_scale[i];
            }
          } else {
            for (j = 0; j < width_i; j++) {
              sum = 0.0f;
              const float *wrow = nn->weight[i] + j * row_len;
              for (k = 0; k < row_len; k++) {
                sum += nn->neuron[i - 1][k] * wrow[k];
              }
              nn->preact[i][j] = sum + nn->bias[i][j];
            }
          }
          // Second pass: apply the activation. Softmax (only ever valid
          // here on LAYER_TYPE_OUTPUT -- enforced by nn_add_layer()) is a
          // numerically-stable whole-layer normalization: subtracting the
          // layer's max preact before exponentiating changes no ratios
          // (exp(x-m)/sum(exp(x-m)) == exp(x)/sum(exp(x)) for any constant
          // m) but keeps the largest exponent at exp(0)=1 instead of
          // risking expf() overflowing to inf for a large preact.
          if (nn->activation[i] == ACTIVATION_FUNCTION_TYPE_SOFTMAX) {
            float maxv = nn->preact[i][0];
            for (j = 1; j < width_i; j++)
              if (nn->preact[i][j] > maxv)
                maxv = nn->preact[i][j];
            float sumexp = 0.0f;
            for (j = 0; j < width_i; j++) {
              float e = expf(nn->preact[i][j] - maxv);
              nn->neuron[i][j] = e; // temporarily unnormalized; divided through below
              sumexp += e;
            }
            for (j = 0; j < width_i; j++)
              nn->neuron[i][j] /= sumexp;
          } else {
            for (j = 0; j < width_i; j++)
              nn->neuron[i][j] = activation_function[nn->activation[i]](nn->preact[i][j], false);
          }
        }
        break;
      case LAYER_TYPE_CNN:
        // Convolutional Neural Network Layer
        nn_conv2d(nn, i);
        break;
      case LAYER_TYPE_POOL:
        // Pooling Layer
        nn_pool_forward(nn, i);
        break;
      case LAYER_TYPE_DROPOUT:
        // Dropout Layer
        nn_dropout_forward(nn, i, training);
        break;
      case LAYER_TYPE_RNN:
        // Recurrent (Elman) Neural Network Layer. Each row of this layer's
        // flat weight buffer is [input-to-hidden weights (row_len_in of
        // them) | hidden-to-hidden/recurrent weights (hidden of them)] --
        // see quantized_layer_shape()'s comment. The recurrent half is
        // multiplied against this same layer's own hidden state from the
        // *previous* timestep, i.e. whatever nn->neuron[i] already holds
        // when this call begins (left there by the previous
        // nn_train()/nn_predict()/nn_error() call, or zeroed by
        // nn_add_layer()/nn_reset_state() for the very first timestep of a
        // sequence) -- NOT the value being computed this call.
        //
        // That previous state is cached into rnn_hidden_prev[i] up front
        // (nn_train()'s backward pass needs it after this function returns,
        // by which point neuron[i] has already been overwritten below with
        // the new state). The two-pass structure below (compute every
        // neuron's preact first, then apply the activation) is what makes
        // this safe: the recurrent dot product only ever reads neuron[i]/
        // rnn_hidden_prev[i], never writes it, so it doesn't matter that
        // neuron[i] is about to become this timestep's output.
        {
          const int row_len_in = (int)nn->width[i - 1]; // input-to-hidden half of each row
          const int hidden = (int)nn->width[i];          // == recurrent half's width == this layer's width
          const int row_len = row_len_in + hidden;        // total flat row length (see quantized_layer_shape())
          float *prev = nn->rnn_hidden_prev[i];
          memcpy(prev, nn->neuron[i], (size_t)hidden * sizeof(float));
          if (nn->quantized) {
            for (j = 0; j < hidden; j++) {
              sum = 0.0f;
              const int8_t *wrow = nn->weight_quantized[i] + j * row_len;
              for (k = 0; k < row_len_in; k++)
                sum += nn->neuron[i - 1][k] * (float)wrow[k];
              for (k = 0; k < hidden; k++)
                sum += prev[k] * (float)wrow[row_len_in + k];
              nn->preact[i][j] = sum * nn->weight_scale[i][j] + (float)nn->bias_quantized[i][j] * nn->bias_scale[i];
            }
          } else {
            for (j = 0; j < hidden; j++) {
              sum = 0.0f;
              const float *wrow = nn->weight[i] + j * row_len;
              for (k = 0; k < row_len_in; k++)
                sum += nn->neuron[i - 1][k] * wrow[k];
              for (k = 0; k < hidden; k++)
                sum += prev[k] * wrow[row_len_in + k];
              nn->preact[i][j] = sum + nn->bias[i][j];
            }
          }
          // Softmax is restricted to LAYER_TYPE_OUTPUT by nn_add_layer(), so
          // (unlike the FC/OUTPUT case above) there is no whole-layer
          // normalization branch to consider here -- every RNN activation is
          // a plain per-neuron function of its own preact.
          for (j = 0; j < hidden; j++)
            nn->neuron[i][j] = activation_function[nn->activation[i]](nn->preact[i][j], false);
        }
        break;
      case LAYER_TYPE_LSTM:
        // Long Short-Term Memory Layer. Like LAYER_TYPE_RNN, this processes
        // one timestep per call with state persisted across calls -- but
        // TWO state vectors instead of one: nn->neuron[i] (the hidden state
        // h, exactly as for RNN) and nn->lstm_cell[i] (the cell state c).
        // This layer's flat weight buffer has 4*hidden rows (see
        // quantized_layer_shape()): rows [0,hidden) are the input gate's,
        // [hidden,2*hidden) the forget gate's, [2*hidden,3*hidden) the cell
        // candidate's, and [3*hidden,4*hidden) the output gate's -- each
        // row_len = row_len_in + hidden wide (input-to-hidden columns
        // followed by hidden-to-hidden/recurrent columns, exactly like
        // LAYER_TYPE_RNN's single gate). Gate nonlinearities are fixed
        // (sigmoid for input/forget/output, tanh for the cell candidate)
        // rather than user-selectable -- nn->activation[i] is stored for
        // this layer but not read here.
        //
        // nn->lstm_cache[i] caches this timestep's previous cell/hidden
        // state and every gate's activation (needed by nn_train()'s
        // backward pass, nn_lstm_backward(), which runs after this
        // function returns -- by which point nn->neuron[i]/lstm_cell[i]
        // already hold the NEW state) as seven width[i]-wide segments, in
        // this order: cell_prev, hidden_prev, gate_i, gate_f, gate_g,
        // gate_o, tanh_c.
        {
          const int row_len_in = (int)nn->width[i - 1];
          const int hidden = (int)nn->width[i];
          const int row_len = row_len_in + hidden;
          float *cache = nn->lstm_cache[i];
          float *c_prev = cache + 0 * hidden;
          float *h_prev = cache + 1 * hidden;
          float *gate_i = cache + 2 * hidden;
          float *gate_f = cache + 3 * hidden;
          float *gate_g = cache + 4 * hidden;
          float *gate_o = cache + 5 * hidden;
          float *tanh_c = cache + 6 * hidden;
          memcpy(c_prev, nn->lstm_cell[i], (size_t)hidden * sizeof(float));
          memcpy(h_prev, nn->neuron[i], (size_t)hidden * sizeof(float));
          // The quantized/float branch is hoisted out per-layer (not
          // per-gate/per-weight), same rationale as the FC/OUTPUT and RNN
          // cases above.
          if (nn->quantized) {
            for (j = 0; j < hidden; j++) {
              float pre_i = 0.0f, pre_f = 0.0f, pre_g = 0.0f, pre_o = 0.0f;
              const int8_t *w_i = nn->weight_quantized[i] + (0 * hidden + j) * row_len;
              const int8_t *w_f = nn->weight_quantized[i] + (1 * hidden + j) * row_len;
              const int8_t *w_g = nn->weight_quantized[i] + (2 * hidden + j) * row_len;
              const int8_t *w_o = nn->weight_quantized[i] + (3 * hidden + j) * row_len;
              for (k = 0; k < row_len_in; k++) {
                float x = nn->neuron[i - 1][k];
                pre_i += x * (float)w_i[k];
                pre_f += x * (float)w_f[k];
                pre_g += x * (float)w_g[k];
                pre_o += x * (float)w_o[k];
              }
              for (k = 0; k < hidden; k++) {
                float hp = h_prev[k];
                pre_i += hp * (float)w_i[row_len_in + k];
                pre_f += hp * (float)w_f[row_len_in + k];
                pre_g += hp * (float)w_g[row_len_in + k];
                pre_o += hp * (float)w_o[row_len_in + k];
              }
              pre_i = pre_i * nn->weight_scale[i][0 * hidden + j] + (float)nn->bias_quantized[i][0 * hidden + j] * nn->bias_scale[i];
              pre_f = pre_f * nn->weight_scale[i][1 * hidden + j] + (float)nn->bias_quantized[i][1 * hidden + j] * nn->bias_scale[i];
              pre_g = pre_g * nn->weight_scale[i][2 * hidden + j] + (float)nn->bias_quantized[i][2 * hidden + j] * nn->bias_scale[i];
              pre_o = pre_o * nn->weight_scale[i][3 * hidden + j] + (float)nn->bias_quantized[i][3 * hidden + j] * nn->bias_scale[i];
              gate_i[j] = activation_function_sigmoid(pre_i, false);
              gate_f[j] = activation_function_sigmoid(pre_f, false);
              gate_g[j] = activation_function_tanh(pre_g, false);
              gate_o[j] = activation_function_sigmoid(pre_o, false);
              float c_new = gate_f[j] * c_prev[j] + gate_i[j] * gate_g[j];
              nn->lstm_cell[i][j] = c_new;
              tanh_c[j] = activation_function_tanh(c_new, false);
              nn->neuron[i][j] = gate_o[j] * tanh_c[j];
              nn->preact[i][j] = nn->neuron[i][j]; // no single "preact" applies here -- mirrored for consistency only, never read for this layer type
            }
          } else {
            for (j = 0; j < hidden; j++) {
              float pre_i = 0.0f, pre_f = 0.0f, pre_g = 0.0f, pre_o = 0.0f;
              const float *w_i = nn->weight[i] + (0 * hidden + j) * row_len;
              const float *w_f = nn->weight[i] + (1 * hidden + j) * row_len;
              const float *w_g = nn->weight[i] + (2 * hidden + j) * row_len;
              const float *w_o = nn->weight[i] + (3 * hidden + j) * row_len;
              for (k = 0; k < row_len_in; k++) {
                float x = nn->neuron[i - 1][k];
                pre_i += x * w_i[k];
                pre_f += x * w_f[k];
                pre_g += x * w_g[k];
                pre_o += x * w_o[k];
              }
              for (k = 0; k < hidden; k++) {
                float hp = h_prev[k];
                pre_i += hp * w_i[row_len_in + k];
                pre_f += hp * w_f[row_len_in + k];
                pre_g += hp * w_g[row_len_in + k];
                pre_o += hp * w_o[row_len_in + k];
              }
              pre_i += nn->bias[i][0 * hidden + j];
              pre_f += nn->bias[i][1 * hidden + j];
              pre_g += nn->bias[i][2 * hidden + j];
              pre_o += nn->bias[i][3 * hidden + j];
              gate_i[j] = activation_function_sigmoid(pre_i, false);
              gate_f[j] = activation_function_sigmoid(pre_f, false);
              gate_g[j] = activation_function_tanh(pre_g, false);
              gate_o[j] = activation_function_sigmoid(pre_o, false);
              float c_new = gate_f[j] * c_prev[j] + gate_i[j] * gate_g[j];
              nn->lstm_cell[i][j] = c_new;
              tanh_c[j] = activation_function_tanh(c_new, false);
              nn->neuron[i][j] = gate_o[j] * tanh_c[j];
              nn->preact[i][j] = nn->neuron[i][j]; // no single "preact" applies here -- mirrored for consistency only, never read for this layer type
            }
          }
        }
        break;
      case LAYER_TYPE_GRU:
        // Gated Recurrent Unit Layer. Like LAYER_TYPE_LSTM, this processes
        // one timestep per call with state persisted across calls, but with
        // only ONE state vector -- nn->neuron[i], the hidden state h, same
        // as LAYER_TYPE_RNN -- and three gates instead of LSTM's four. This
        // layer's flat weight buffer has 3*hidden rows (see
        // quantized_layer_shape()): rows [0,hidden) are the reset gate's,
        // [hidden,2*hidden) the update gate's, [2*hidden,3*hidden) the
        // candidate gate's -- each row_len = row_len_in + hidden wide,
        // exactly like LAYER_TYPE_RNN's single gate. Reset/update use a
        // sigmoid; the candidate uses tanh, with its recurrent contribution
        // scaled by the reset gate *before* the bias is added:
        //   r = sigmoid(W_r x + U_r h_prev + b_r)
        //   z = sigmoid(W_z x + U_z h_prev + b_z)
        //   n = tanh(W_n x + r * (U_n h_prev) + b_n)
        //   h_new = (1 - z) * h_prev + z * n
        // (z close to 1 means "take the new candidate"; z close to 0 means
        // "keep the old state" -- the single update gate does the job LSTM
        // splits across separate input/forget gates.)
        //
        // nn->gru_cache[i] caches this timestep's previous hidden state and
        // every gate's activation (needed by nn_train()'s backward pass,
        // nn_gru_backward(), which runs after this function returns -- by
        // which point nn->neuron[i] already holds the NEW state) as five
        // width[i]-wide segments, in this order: hidden_prev, gate_r,
        // gate_z, gate_n, n_recur_raw (U_n h_prev, cached before the reset
        // gate multiplies it -- needed for the reset gate's own gradient).
        {
          const int row_len_in = (int)nn->width[i - 1];
          const int hidden = (int)nn->width[i];
          const int row_len = row_len_in + hidden;
          float *cache = nn->gru_cache[i];
          float *h_prev = cache + 0 * hidden;
          float *gate_r = cache + 1 * hidden;
          float *gate_z = cache + 2 * hidden;
          float *gate_n = cache + 3 * hidden;
          float *n_recur_raw = cache + 4 * hidden;
          memcpy(h_prev, nn->neuron[i], (size_t)hidden * sizeof(float));
          // The quantized/float branch is hoisted out per-layer (not
          // per-gate/per-weight), same rationale as the FC/OUTPUT, RNN, and
          // LSTM cases above.
          if (nn->quantized) {
            for (j = 0; j < hidden; j++) {
              float pre_r = 0.0f, pre_z = 0.0f, pre_n = 0.0f, n_recur = 0.0f;
              const int8_t *w_r = nn->weight_quantized[i] + (0 * hidden + j) * row_len;
              const int8_t *w_z = nn->weight_quantized[i] + (1 * hidden + j) * row_len;
              const int8_t *w_n = nn->weight_quantized[i] + (2 * hidden + j) * row_len;
              for (k = 0; k < row_len_in; k++) {
                float x = nn->neuron[i - 1][k];
                pre_r += x * (float)w_r[k];
                pre_z += x * (float)w_z[k];
                pre_n += x * (float)w_n[k];
              }
              for (k = 0; k < hidden; k++) {
                float hp = h_prev[k];
                pre_r += hp * (float)w_r[row_len_in + k];
                pre_z += hp * (float)w_z[row_len_in + k];
                n_recur += hp * (float)w_n[row_len_in + k];
              }
              pre_r = pre_r * nn->weight_scale[i][0 * hidden + j] + (float)nn->bias_quantized[i][0 * hidden + j] * nn->bias_scale[i];
              pre_z = pre_z * nn->weight_scale[i][1 * hidden + j] + (float)nn->bias_quantized[i][1 * hidden + j] * nn->bias_scale[i];
              n_recur = n_recur * nn->weight_scale[i][2 * hidden + j];
              gate_r[j] = activation_function_sigmoid(pre_r, false);
              gate_z[j] = activation_function_sigmoid(pre_z, false);
              n_recur_raw[j] = n_recur;
              pre_n = pre_n * nn->weight_scale[i][2 * hidden + j] + gate_r[j] * n_recur + (float)nn->bias_quantized[i][2 * hidden + j] * nn->bias_scale[i];
              gate_n[j] = activation_function_tanh(pre_n, false);
              float h_new = (1.0f - gate_z[j]) * h_prev[j] + gate_z[j] * gate_n[j];
              nn->neuron[i][j] = h_new;
              nn->preact[i][j] = h_new; // no single "preact" applies here -- mirrored for consistency only, never read for this layer type
            }
          } else {
            for (j = 0; j < hidden; j++) {
              float pre_r = 0.0f, pre_z = 0.0f, pre_n = 0.0f, n_recur = 0.0f;
              const float *w_r = nn->weight[i] + (0 * hidden + j) * row_len;
              const float *w_z = nn->weight[i] + (1 * hidden + j) * row_len;
              const float *w_n = nn->weight[i] + (2 * hidden + j) * row_len;
              for (k = 0; k < row_len_in; k++) {
                float x = nn->neuron[i - 1][k];
                pre_r += x * w_r[k];
                pre_z += x * w_z[k];
                pre_n += x * w_n[k];
              }
              for (k = 0; k < hidden; k++) {
                float hp = h_prev[k];
                pre_r += hp * w_r[row_len_in + k];
                pre_z += hp * w_z[row_len_in + k];
                n_recur += hp * w_n[row_len_in + k];
              }
              pre_r += nn->bias[i][0 * hidden + j];
              pre_z += nn->bias[i][1 * hidden + j];
              gate_r[j] = activation_function_sigmoid(pre_r, false);
              gate_z[j] = activation_function_sigmoid(pre_z, false);
              n_recur_raw[j] = n_recur;
              pre_n += gate_r[j] * n_recur + nn->bias[i][2 * hidden + j];
              gate_n[j] = activation_function_tanh(pre_n, false);
              float h_new = (1.0f - gate_z[j]) * h_prev[j] + gate_z[j] * gate_n[j];
              nn->neuron[i][j] = h_new;
              nn->preact[i][j] = h_new; // no single "preact" applies here -- mirrored for consistency only, never read for this layer type
            }
          }
        }
        break;
      case LAYER_TYPE_ATTENTION:
        // Attention Layer
        // TODO
        break;
      case LAYER_TYPE_TRANSFORMER:
        // Transformer Layer
        // TODO
        break;
      case LAYER_TYPE_INPUT:
      case LAYER_TYPE_NONE:
      default:
        // No operation for this layer
        break;
    }
  }
}

// Public functions

uint32_t nn_version(void)
{
  return ((uint32_t)NN_VERSION_MAJOR << 24) |
         ((uint32_t)NN_VERSION_MINOR << 16) |
         ((uint32_t)NN_VERSION_PATCH << 8) |
         (uint32_t)NN_VERSION_BUILD;
}

nn_t *nn_init(void)
{
  nn_t *nn = (nn_t *)malloc(sizeof(nn_t));
  if (nn == NULL)
    return NULL;
  // Mark as non-quantized by default
  nn->quantized = false;
  // Populate version from macros
  nn->version_major = NN_VERSION_MAJOR;
  nn->version_minor = NN_VERSION_MINOR;
  nn->version_patch = NN_VERSION_PATCH;
  nn->version_build = NN_VERSION_BUILD;
  nn->depth = 0;
  nn->layer_type = NULL;
  nn->width = NULL;
  nn->activation = NULL;
  nn->config = NULL;
  // Floats-only pointers are NULL initially
  nn->neuron = NULL;
  nn->loss = NULL;
  nn->preact = NULL;
  nn->weight = NULL;
  nn->weight_adj = NULL;
  nn->bias = NULL;
  // Quantization-related pointers start NULL
  nn->weight_quantized = NULL;
  nn->weight_scale = NULL;
  nn->bias_quantized = NULL;
  nn->bias_scale = NULL;
  nn->pool_argmax = NULL;
  nn->dropout_scale = NULL;
  nn->dropout_mask_valid = NULL;
  nn->rnn_hidden_prev = NULL;
  nn->lstm_cell = NULL;
  nn->lstm_cache = NULL;
  nn->lstm_gate_grad = NULL;
  nn->gru_cache = NULL;
  nn->gru_gate_grad = NULL;
  // Optimizer defaults to plain SGD (value 0), matching this library's
  // original behavior exactly -- no persistent per-weight state, and the
  // hyperparameter fields/moment buffers below are simply unused until
  // nn_set_optimizer() selects something else.
  nn->optimizer = NN_OPTIMIZER_SGD;
  nn->optimizer_momentum = 0.0f;
  nn->optimizer_beta2 = 0.0f;
  nn->optimizer_epsilon = 0.0f;
  nn->adam_step = 0;
  nn->weight_moment1 = NULL;
  nn->weight_moment2 = NULL;
  nn->bias_moment1 = NULL;
  nn->bias_moment2 = NULL;
  nn->activation_buffer = NULL;
  nn->immutable = false;
  return nn;
}

// False only for a layer whose neuron[]/preact[] point into a model's shared
// activation_buffer (see its comment in nn.h) rather than at the layer's own
// allocations: every non-recurrent layer of a model loaded with
// nn_load_model_inplace(). Recurrent layers always own theirs, since their
// neuron[] is persistent state.
static bool nn_owns_activations(const nn_t *nn, int layer)
{
  return !nn->activation_buffer || nn_is_recurrent_layer_type(nn->layer_type[layer]);
}

void nn_free(nn_t *nn)
{
  if (nn == NULL)
    return;
  // Free weight and bias related arrays layer by layer (float side)
  // There are no weights/biases for layer 0, so start from layer 1.
  if (!nn->quantized) {
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      // weight/weight_adj are each one flat buffer per layer (or NULL for
      // POOL); free(NULL) is a no-op so this is safe uniformly across
      // every layer type. Except: for a model loaded with
      // nn_load_model_inplace(), weight[layer]/bias[layer] alias the
      // caller's buffer instead of being owned allocations -- freeing them
      // would be undefined behavior, so immutable gates those two.
      if (!nn->immutable) {
        free(nn->weight[layer]);
        free(nn->bias[layer]);
      }
      free(nn->weight_adj[layer]);
      free(nn->weight_scale[layer]);
      // Optimizer moment buffers: NULL (free() no-ops) unless
      // nn_set_optimizer() selected MOMENTUM/ADAM -- see their comment in nn.h.
      free(nn->weight_moment1[layer]);
      free(nn->weight_moment2[layer]);
      free(nn->bias_moment1[layer]);
      free(nn->bias_moment2[layer]);
      free(nn->config[layer]);
      if (nn_owns_activations(nn, layer)) {
        free(nn->neuron[layer]);
        free(nn->preact[layer]);
      }
      free(nn->loss[layer]);
    }
    free(nn->weight);
    free(nn->weight_adj);
    free(nn->weight_scale);
    free(nn->weight_moment1);
    free(nn->weight_moment2);
    free(nn->bias);
    free(nn->bias_scale);
    free(nn->bias_moment1);
    free(nn->bias_moment2);
    free(nn->neuron);
    free(nn->loss);
    free(nn->preact);
    free(nn->layer_type);
    free(nn->width);
    free(nn->activation);
    free(nn->config);
  }
  // Free quantized side arrays if allocated
  if (nn->quantized) {
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      // weight_quantized is one flat buffer per layer (or NULL for POOL);
      // free(NULL) is a no-op so this is safe uniformly. Guard the whole
      // group on the top-level pointer, though: a load failure part-way
      // through reading a quantized file can leave weight_quantized (and
      // its siblings) already torn down and NULLed by the caller, in which
      // case there's nothing left here to index into.
      // As above (float-side comment), weight_quantized[layer]/
      // weight_scale[layer]/bias_quantized[layer] alias the caller's buffer
      // for a model loaded with nn_load_model_inplace() and must not be freed.
      if (nn->weight_quantized && !nn->immutable) {
        free(nn->weight_quantized[layer]);
        free(nn->weight_scale[layer]);
        free(nn->bias_quantized[layer]);
      }
      free(nn->config[layer]);
      if (nn_owns_activations(nn, layer)) {
        free(nn->neuron[layer]);
        free(nn->preact[layer]);
      }
      free(nn->loss[layer]);
    }
    free(nn->weight_quantized);
    free(nn->weight_scale);
    free(nn->bias_quantized);
    free(nn->bias_scale);
    free(nn->neuron);
    free(nn->loss);
    free(nn->preact);
    free(nn->layer_type);
    free(nn->width);
    free(nn->activation);
    free(nn->config);
  }
  // Free the pooling argmax cache (independent of quantized state)
  if (nn->pool_argmax) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->pool_argmax[layer]);
    free(nn->pool_argmax);
  }
  // Free the dropout per-neuron scale cache (independent of quantized state,
  // same as pool_argmax above)
  if (nn->dropout_scale) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->dropout_scale[layer]);
    free(nn->dropout_scale);
  }
  // Flat, one-bool-per-layer (not nested like dropout_scale above), same
  // convention as bias_scale -- a single free, no per-layer loop.
  free(nn->dropout_mask_valid);
  // Free the RNN previous-hidden-state cache (independent of quantized
  // state, same as pool_argmax/dropout_scale above)
  if (nn->rnn_hidden_prev) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->rnn_hidden_prev[layer]);
    free(nn->rnn_hidden_prev);
  }
  // Free the LSTM cell state and its backward-pass caches (independent of
  // quantized state, same as pool_argmax/dropout_scale/rnn_hidden_prev above)
  if (nn->lstm_cell) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->lstm_cell[layer]);
    free(nn->lstm_cell);
  }
  if (nn->lstm_cache) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->lstm_cache[layer]);
    free(nn->lstm_cache);
  }
  if (nn->lstm_gate_grad) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->lstm_gate_grad[layer]);
    free(nn->lstm_gate_grad);
  }
  // Free the GRU backward-pass caches (independent of quantized state, same
  // as the LSTM arrays above). GRU has no separate persistent-state array
  // of its own (unlike lstm_cell) -- its one state lives in neuron[] like RNN.
  if (nn->gru_cache) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->gru_cache[layer]);
    free(nn->gru_cache);
  }
  if (nn->gru_gate_grad) {
    for (int layer = 1; layer < (int)nn->depth; layer++)
      free(nn->gru_gate_grad[layer]);
    free(nn->gru_gate_grad);
  }
  free(nn->activation_buffer);
  free(nn);
}

nn_error_t nn_add_layer(nn_t *nn, layer_type_t layer_type, int width, int activation, void *config)
{
  cnn_t *cnn = NULL;
  pool_t *pool = NULL;
  int out_w = 0, out_h = 0; // CNN/POOL output spatial dims, computed once below

  // Validate the CNN config before mutating `nn` at all, so a rejected call
  // leaves the network completely unchanged (in particular, still safe to
  // keep using or to nn_free()).
  if (layer_type == LAYER_TYPE_CNN) {
    if (config == NULL) {
      return NN_ERROR_INVALID_CONFIG;
    }
    cnn_t *cnn_check = (cnn_t *)config;
    // dilation is accepted and round-tripped through save/load, but neither
    // the output-size formula below nor nn_conv2d()'s actual convolution
    // loop implements it -- silently accepting a non-default value would
    // produce a different (and wrong, from the caller's expectation) result
    // instead of what was asked for. Reject rather than silently ignore,
    // until it's genuinely implemented.
    if (cnn_check->dilation != 1) {
      fprintf(stderr, "nn_add_layer: CNN dilation is not implemented (got dilation=%u; only dilation=1 is supported)\n", cnn_check->dilation);
      return NN_ERROR_INVALID_CONFIG;
    }
    // Padding must leave at least one valid output position in each spatial
    // dimension, or the output-size formula below underflows.
    if ((int)cnn_check->in_h + 2 * (int)cnn_check->padding < (int)cnn_check->kernel_size ||
        (int)cnn_check->in_w + 2 * (int)cnn_check->padding < (int)cnn_check->kernel_size) {
      fprintf(stderr, "nn_add_layer: CNN padding too large for kernel_size/input dims (in_h=%u, in_w=%u, kernel_size=%u, padding=%u)\n",
              cnn_check->in_h, cnn_check->in_w, cnn_check->kernel_size, cnn_check->padding);
      return NN_ERROR_INVALID_CONFIG;
    }
  } else if (layer_type == LAYER_TYPE_DROPOUT) {
    if (config == NULL) {
      return NN_ERROR_INVALID_CONFIG;
    }
    dropout_t *dropout_check = (dropout_t *)config;
    // rate == 1 would mean every unit is always dropped, making the
    // inverted-dropout scale 1/(1-rate) divide by zero.
    if (!(dropout_check->rate >= 0.0f) || !(dropout_check->rate < 1.0f)) {
      fprintf(stderr, "nn_add_layer: DROPOUT rate must be in [0, 1) (got %g)\n", (double)dropout_check->rate);
      return NN_ERROR_INVALID_CONFIG;
    }
    // A DROPOUT layer's width is derived from the previous layer (it's a
    // same-width pass-through), so it cannot be the very first layer added.
    if (nn->depth == 0) {
      fprintf(stderr, "nn_add_layer: DROPOUT cannot be the first layer (no previous layer to derive its width from)\n");
      return NN_ERROR_INVALID_CONFIG;
    }
  }
  // Softmax depends on every neuron's preact in the layer (see the comment
  // on ACTIVATION_FUNCTION_TYPE_SOFTMAX in nn.h), which forward_propagation()
  // and nn_train() only handle for LAYER_TYPE_OUTPUT -- reject it elsewhere
  // rather than silently computing something else.
  if (activation == ACTIVATION_FUNCTION_TYPE_SOFTMAX && layer_type != LAYER_TYPE_OUTPUT) {
    fprintf(stderr, "nn_add_layer: SOFTMAX activation is only valid on LAYER_TYPE_OUTPUT (got layer_type=%d)\n", (int)layer_type);
    return NN_ERROR_UNSUPPORTED_ACTIVATION;
  }

  // Increase depth by one
  nn->depth++;
  nn->layer_type = (uint8_t *)realloc(nn->layer_type, nn->depth * sizeof(*nn->layer_type));
  if (nn->layer_type == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  // nn->depth - 1 is the index of the new layer that we are adding
  nn->layer_type[nn->depth - 1] = (uint8_t)layer_type;
  nn->width = (uint32_t *)realloc(nn->width, nn->depth * sizeof(*nn->width));
  if (nn->width == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->width[nn->depth - 1] = (uint32_t)width;
  if (layer_type == LAYER_TYPE_CNN) {
    // config's validity (non-NULL, padding=0, dilation=1) was already
    // checked above before any of the mutation up to this point.
    cnn = (cnn_t *)config;
    out_w = ((cnn->in_w + 2 * cnn->padding - cnn->kernel_size) / cnn->stride) + 1;
    out_h = ((cnn->in_h + 2 * cnn->padding - cnn->kernel_size) / cnn->stride) + 1;
    nn->width[nn->depth - 1] = cnn->out_channels * out_w * out_h;
  } else if (layer_type == LAYER_TYPE_POOL) {
    if (config == NULL) {
      return NN_ERROR_INVALID_CONFIG;
    }
    pool = (pool_t *)config;
    out_w = ((pool->in_w - pool->pool_size) / pool->stride) + 1;
    out_h = ((pool->in_h - pool->pool_size) / pool->stride) + 1;
    nn->width[nn->depth - 1] = pool->channels * out_w * out_h;
  } else if (layer_type == LAYER_TYPE_DROPOUT) {
    // Pass-through: same width as the previous layer. nn->depth was already
    // validated to be >= 1 (pre-increment) above, so nn->depth - 2 >= 0 here.
    nn->width[nn->depth - 1] = nn->width[nn->depth - 2];
  }
  nn->activation = (uint8_t *)realloc(nn->activation, nn->depth * sizeof(*nn->activation));
  if (nn->activation == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->activation[nn->depth - 1] = (uint8_t)activation;
  nn->config = (void **)realloc(nn->config, nn->depth * sizeof(*nn->config));
  if (nn->config == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->config[nn->depth - 1] = NULL;
  if (layer_type == LAYER_TYPE_CNN) {
    nn->config[nn->depth - 1] = (void *)malloc(sizeof(cnn_t));
    if (nn->config[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    // Copy the CNN configuration
    memcpy(nn->config[nn->depth - 1], config, sizeof(cnn_t));
    // Cache the output dims so nn_conv2d() and nn_train() never need to
    // re-derive them via division on every sample/epoch.
    ((cnn_t *)nn->config[nn->depth - 1])->out_w = (uint16_t)out_w;
    ((cnn_t *)nn->config[nn->depth - 1])->out_h = (uint16_t)out_h;
  } else if (layer_type == LAYER_TYPE_POOL) {
    nn->config[nn->depth - 1] = (void *)malloc(sizeof(pool_t));
    if (nn->config[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    // Copy the pooling configuration
    memcpy(nn->config[nn->depth - 1], config, sizeof(pool_t));
    // Cache the output dims for the same reason as the CNN case above.
    ((pool_t *)nn->config[nn->depth - 1])->out_w = (uint16_t)out_w;
    ((pool_t *)nn->config[nn->depth - 1])->out_h = (uint16_t)out_h;
  } else if (layer_type == LAYER_TYPE_DROPOUT) {
    nn->config[nn->depth - 1] = (void *)malloc(sizeof(dropout_t));
    if (nn->config[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    // Copy the dropout configuration (no output dims to cache -- width was
    // already derived directly from the previous layer above).
    memcpy(nn->config[nn->depth - 1], config, sizeof(dropout_t));
  }
  nn->neuron = (float **)realloc(nn->neuron, nn->depth * sizeof(float *));
  if (nn->neuron == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->loss = (float **)realloc(nn->loss, nn->depth * sizeof(float *));
  if (nn->loss == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->preact = (float **)realloc(nn->preact, nn->depth * sizeof(float *));
  if (nn->preact == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight = (float **)realloc(nn->weight, (nn->depth) * sizeof(float *));
  if (nn->weight == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight_adj = (float **)realloc(nn->weight_adj, (nn->depth) * sizeof(float *));
  if (nn->weight_adj == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight_scale = (float **)realloc(nn->weight_scale, (nn->depth) * sizeof(float *));
  if (nn->weight_scale == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->bias = (float **)realloc(nn->bias, (nn->depth) * sizeof(float *));
  if (nn->bias == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->bias_scale = (float *)realloc(nn->bias_scale, (nn->depth) * sizeof(float));
  if (nn->bias_scale == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->pool_argmax = (int **)realloc(nn->pool_argmax, (nn->depth) * sizeof(int *));
  if (nn->pool_argmax == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->pool_argmax[nn->depth - 1] = NULL;
  nn->dropout_scale = (float **)realloc(nn->dropout_scale, (nn->depth) * sizeof(float *));
  if (nn->dropout_scale == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->dropout_scale[nn->depth - 1] = NULL;
  nn->dropout_mask_valid = (bool *)realloc(nn->dropout_mask_valid, (nn->depth) * sizeof(bool));
  if (nn->dropout_mask_valid == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->dropout_mask_valid[nn->depth - 1] = false;
  nn->rnn_hidden_prev = (float **)realloc(nn->rnn_hidden_prev, (nn->depth) * sizeof(float *));
  if (nn->rnn_hidden_prev == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->rnn_hidden_prev[nn->depth - 1] = NULL;
  nn->lstm_cell = (float **)realloc(nn->lstm_cell, (nn->depth) * sizeof(float *));
  if (nn->lstm_cell == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->lstm_cell[nn->depth - 1] = NULL;
  nn->lstm_cache = (float **)realloc(nn->lstm_cache, (nn->depth) * sizeof(float *));
  if (nn->lstm_cache == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->lstm_cache[nn->depth - 1] = NULL;
  nn->lstm_gate_grad = (float **)realloc(nn->lstm_gate_grad, (nn->depth) * sizeof(float *));
  if (nn->lstm_gate_grad == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->lstm_gate_grad[nn->depth - 1] = NULL;
  nn->gru_cache = (float **)realloc(nn->gru_cache, (nn->depth) * sizeof(float *));
  if (nn->gru_cache == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->gru_cache[nn->depth - 1] = NULL;
  nn->gru_gate_grad = (float **)realloc(nn->gru_gate_grad, (nn->depth) * sizeof(float *));
  if (nn->gru_gate_grad == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->gru_gate_grad[nn->depth - 1] = NULL;
  nn->weight_moment1 = (float **)realloc(nn->weight_moment1, (nn->depth) * sizeof(float *));
  if (nn->weight_moment1 == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight_moment1[nn->depth - 1] = NULL;
  nn->weight_moment2 = (float **)realloc(nn->weight_moment2, (nn->depth) * sizeof(float *));
  if (nn->weight_moment2 == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->weight_moment2[nn->depth - 1] = NULL;
  nn->bias_moment1 = (float **)realloc(nn->bias_moment1, (nn->depth) * sizeof(float *));
  if (nn->bias_moment1 == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->bias_moment1[nn->depth - 1] = NULL;
  nn->bias_moment2 = (float **)realloc(nn->bias_moment2, (nn->depth) * sizeof(float *));
  if (nn->bias_moment2 == NULL)
    return NN_ERROR_OUT_OF_MEMORY;
  nn->bias_moment2[nn->depth - 1] = NULL;
  // For layer 0, we do not allocate neuron/loss/preact (input is provided externally)
  if (nn->depth > 1) {
    nn->neuron[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
    if (nn->neuron[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    nn->loss[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
    if (nn->loss[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    nn->preact[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
    if (nn->preact[nn->depth - 1] == NULL)
      return NN_ERROR_OUT_OF_MEMORY;
    if (layer_type == LAYER_TYPE_POOL || layer_type == LAYER_TYPE_DROPOUT) {
      // Pooling and dropout both have no learnable parameters
      nn->weight[nn->depth - 1] = NULL;
      nn->weight_adj[nn->depth - 1] = NULL;
      nn->weight_scale[nn->depth - 1] = NULL;
      nn->bias[nn->depth - 1] = NULL;
      if (layer_type == LAYER_TYPE_POOL) {
        // MIN/MAX pooling need to remember which input position "won" each
        // output, so backprop can route the gradient to only that position.
        if (pool->pooling_type == POOLING_TYPE_MAX || pool->pooling_type == POOLING_TYPE_MIN) {
          nn->pool_argmax[nn->depth - 1] = (int *)malloc(nn->width[nn->depth - 1] * sizeof(int));
          if (nn->pool_argmax[nn->depth - 1] == NULL)
            return NN_ERROR_OUT_OF_MEMORY;
        }
      } else {
        // DROPOUT: nn_dropout_forward() fills this in on every training
        // forward pass with the per-neuron scale it applied (0 or
        // 1/(1-rate)), so nn_dropout_backward() can route the gradient the
        // same way. Allocated here (not lazily) so a later OOM can't happen
        // mid-training; unused (never populated or read) for a model that's
        // only ever run through nn_predict()/nn_error().
        nn->dropout_scale[nn->depth - 1] = (float *)malloc(nn->width[nn->depth - 1] * sizeof(float));
        if (nn->dropout_scale[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
      }
    } else {
      // CNN, FC, OUTPUT, and RNN layers all store weight/weight_adj as one
      // flat, row-major buffer of rows*row_len elements: a "row" is a kernel
      // for CNN layers (row_len = kernel_size^2, one bias per output
      // channel), a neuron for FC/OUTPUT layers (row_len = previous layer's
      // width, one bias per neuron), or a neuron for RNN layers (row_len =
      // previous layer's width + this layer's own width, one bias per
      // neuron) -- see quantized_layer_shape().
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, nn->depth - 1, &rows, &row_len, &bias_count);
      float range;
      if (layer_type == LAYER_TYPE_CNN) {
        // Xavier (Glorot) initialisation: fan_in/fan_out count every
        // connection a kernel weight participates in, i.e. across all
        // input/output channels, not just its own row_len.
        range = sqrtf(6.0f / (float)(cnn->in_channels * row_len + cnn->out_channels * row_len));
      } else {
        // Xavier (Glorot) initialization
        range = sqrtf(6.0f / (float)(rows + row_len));
      }
      nn->weight[nn->depth - 1] = (float *)malloc((size_t)rows * row_len * sizeof(float));
      nn->weight_adj[nn->depth - 1] = (float *)malloc((size_t)rows * row_len * sizeof(float));
      nn->weight_scale[nn->depth - 1] = (float *)malloc((size_t)rows * sizeof(float));
      nn->bias[nn->depth - 1] = (float *)malloc((size_t)bias_count * sizeof(float));
      if (!nn->weight[nn->depth - 1] || !nn->weight_adj[nn->depth - 1] || !nn->weight_scale[nn->depth - 1] || !nn->bias[nn->depth - 1])
        return NN_ERROR_OUT_OF_MEMORY;
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < row_len; ++c) {
          nn->weight[nn->depth - 1][r * row_len + c] = range * 2.0f * ((rand() / (float)RAND_MAX) - 0.5f);
          nn->weight_adj[nn->depth - 1][r * row_len + c] = 0.0f;
        }
        nn->weight_scale[nn->depth - 1][r] = 0.0f; /* filled during quantise */
      }
      for (int b = 0; b < bias_count; ++b)
        nn->bias[nn->depth - 1][b] = 0.0f;
      if (layer_type == LAYER_TYPE_RNN) {
        // nn->neuron[nn->depth - 1] (malloc'd, not zeroed, just above) IS
        // this layer's hidden state -- it must start at all-zeros so the
        // very first timestep's recurrent dot product (forward_propagation()'s
        // LAYER_TYPE_RNN case) reads a defined "no history yet" state
        // instead of uninitialized memory. nn_reset_state() re-zeros this
        // the same way to start a new sequence later.
        memset(nn->neuron[nn->depth - 1], 0, (size_t)bias_count * sizeof(float));
        // Cache of this layer's hidden state from just before the most
        // recent forward pass overwrote it -- see forward_propagation()'s
        // LAYER_TYPE_RNN case and nn_train()'s recurrent weight_adj
        // computation.
        nn->rnn_hidden_prev[nn->depth - 1] = (float *)malloc((size_t)bias_count * sizeof(float));
        if (nn->rnn_hidden_prev[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
      } else if (layer_type == LAYER_TYPE_LSTM) {
        // bias_count == 4 * hidden here (see quantized_layer_shape()); the
        // actual per-gate hidden width is nn->width[nn->depth - 1].
        const int hidden = (int)nn->width[nn->depth - 1];
        // neuron[nn->depth-1] doubles as this layer's hidden state h (same
        // as RNN, and for the same reason) and must start at all-zeros.
        memset(nn->neuron[nn->depth - 1], 0, (size_t)hidden * sizeof(float));
        // The cell state c, this layer's second (longer-lived) piece of
        // persistent memory -- also starts at all-zeros. nn_reset_state()
        // re-zeros both this and neuron[] to start a new sequence.
        nn->lstm_cell[nn->depth - 1] = (float *)malloc((size_t)hidden * sizeof(float));
        if (nn->lstm_cell[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
        memset(nn->lstm_cell[nn->depth - 1], 0, (size_t)hidden * sizeof(float));
        // Cache of this timestep's previous cell/hidden state and every
        // gate's activation, for nn_train()'s backward pass -- see its
        // layout comment where lstm_cache is declared in nn.h.
        nn->lstm_cache[nn->depth - 1] = (float *)malloc(7 * (size_t)hidden * sizeof(float));
        if (nn->lstm_cache[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
        // Backward-pass output: each of the four gates' preact-space
        // gradient, one width[nn->depth-1]-wide segment per gate (see
        // nn_lstm_backward()).
        nn->lstm_gate_grad[nn->depth - 1] = (float *)malloc(4 * (size_t)hidden * sizeof(float));
        if (nn->lstm_gate_grad[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
      } else if (layer_type == LAYER_TYPE_GRU) {
        // bias_count == 3 * hidden here (see quantized_layer_shape()); the
        // actual per-gate hidden width is nn->width[nn->depth - 1].
        const int hidden = (int)nn->width[nn->depth - 1];
        // neuron[nn->depth-1] doubles as this layer's hidden state h (same
        // as RNN/LSTM, and for the same reason) and must start at all-zeros.
        // Unlike LSTM, GRU has no second persistent-state array -- just this.
        memset(nn->neuron[nn->depth - 1], 0, (size_t)hidden * sizeof(float));
        // Cache of this timestep's previous hidden state and every gate's
        // activation, for nn_train()'s backward pass -- see its layout
        // comment where gru_cache is declared in nn.h.
        nn->gru_cache[nn->depth - 1] = (float *)malloc(5 * (size_t)hidden * sizeof(float));
        if (nn->gru_cache[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
        // Backward-pass output: each of the three gates' preact-space
        // gradient, one width[nn->depth-1]-wide segment per gate (see
        // nn_gru_backward()).
        nn->gru_gate_grad[nn->depth - 1] = (float *)malloc(3 * (size_t)hidden * sizeof(float));
        if (nn->gru_gate_grad[nn->depth - 1] == NULL)
          return NN_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  // If an optimizer needing persistent per-weight state was already
  // selected (nn_set_optimizer() called before this layer was added), give
  // the new layer its own moment buffers too -- see
  // nn_optimizer_alloc_layer()'s comment. A no-op under NN_OPTIMIZER_SGD
  // (the default), and only runs at model-construction time, never inside
  // the hot training/inference path. Layer 0 (the INPUT layer) never has
  // weights of its own -- quantized_layer_shape() assumes a previous layer
  // exists, so this is skipped for it the same way weight/weight_adj/bias
  // themselves are above.
  if (nn->depth > 1 && !nn_optimizer_alloc_layer(nn, (int)nn->depth - 1))
    return NN_ERROR_OUT_OF_MEMORY;
  return NN_ERROR_NONE;
}

// Returns the total error of the network given a set of inputs and target outputs
float nn_error(nn_t *nn, float *inputs, float *targets)
{
  int i, j;
  float err = 0.0f;

  // Layer 0's neuron pointers simply reference the input array
  nn->neuron[0] = inputs;
  // training=false: any DROPOUT layer is a pass-through here, so nn_error()
  // reports a stable, reproducible figure (e.g. for validation/test error)
  // rather than one perturbed by dropout's per-call randomness.
  forward_propagation(nn, false);
  // Sum error on the final (output) layer: cross-entropy if this network
  // uses a softmax output (see the comment on ACTIVATION_FUNCTION_TYPE_SOFTMAX
  // in nn.h -- softmax and MSE are not a matched pair), MSE otherwise.
  i = (int)nn->depth - 1;
  if (nn->activation[i] == ACTIVATION_FUNCTION_TYPE_SOFTMAX) {
    for (j = 0; j < (int)nn->width[i]; j++) {
      err += cross_entropy_term(targets[j], nn->neuron[i][j]);
    }
  } else {
    for (j = 0; j < (int)nn->width[i]; j++) {
      err += error(targets[j], nn->neuron[i][j]);
    }
  }
  return err;
}

// Trains a nn with a given input and target output at a specified learning
// rate. The rate (or step size) controls how far in the search space to move
// against the gradient in each iteration of the algorithm. This function
// assumes quantized==false Returns the total error between the target and the
// output of the neural network.
float nn_train(nn_t *nn, float *inputs, float *targets, float rate)
{
  float sum;
  int i, j, k;
  float err;

  if (nn->immutable) {
    // A model loaded with nn_load_model_inplace() has its weights aliased
    // into the caller's (typically flash-resident, read-only) buffer --
    // there is nothing writable here for gradient descent to update.
    fprintf(stderr, "nn_train: cannot train a read-only, flash-resident model loaded with nn_load_model_inplace()\n");
    return NAN;
  }
  if (nn->quantized) {
    // Cannot train a quantized network, so convert to a floating point model first.
    nn_dequantize(nn);
  }
  nn->neuron[0] = inputs;
  // training=true: any DROPOUT layer actually drops/scales units here (see
  // forward_propagation()).
  forward_propagation(nn, true);
  // Capture this sample's pre-update error now, while neuron[] still reflects
  // the forward pass above. This is the conventional "training loss" and lets
  // us avoid a second, redundant forward_propagation() call at the end of
  // this function (nn->neuron[] is not touched again until the next forward
  // pass, so this is equivalent to what a trailing nn_error() call would have
  // computed from the pre-update weights -- except for the contribution of
  // any DROPOUT layer, which nn_error() always runs as a pass-through).
  i = (int)nn->depth - 1;
  err = 0.0f;
  const bool softmax_output = (nn->activation[i] == ACTIVATION_FUNCTION_TYPE_SOFTMAX);
  if (softmax_output) {
    for (j = 0; j < (int)nn->width[i]; j++) {
      err += cross_entropy_term(targets[j], nn->neuron[i][j]);
    }
  } else {
    for (j = 0; j < (int)nn->width[i]; j++) {
      err += error(targets[j], nn->neuron[i][j]);
    }
  }
  // Perform back propagation using gradient descent, which is an optimization
  // algorithm that follows the negative gradient of the objective function to
  // find the minimum of the function. Start at the output layer, and work
  // backward toward the input layer, adjusting weights along the way. Calculate
  // the error aka loss aka delta at the output.
  // Compute output layer loss. This must include the output layer's own
  // activation derivative here (not later), so that loss[] always
  // uniformly represents -dE/d(preact) at every layer -- the propagation
  // step below then never needs to re-derive a layer's own derivative from
  // its neighbor's loss.
  if (softmax_output) {
    // Softmax + cross-entropy: the two Jacobians (softmax's own, and
    // cross-entropy's derivative w.r.t. softmax's output) cancel into this
    // simple difference -- see the comment on ACTIVATION_FUNCTION_TYPE_SOFTMAX
    // in nn.h. There is no separate activation-derivative factor to apply
    // here (unlike every other activation below): softmax's true derivative
    // is a full cross-neuron Jacobian, not a per-neuron scalar, so
    // activation_function[...](preact, true) cannot represent it anyway.
    for (j = 0; j < (int)nn->width[i]; j++) {
      nn->loss[i][j] = error_derivative(targets[j], nn->neuron[i][j]);
    }
  } else {
    for (j = 0; j < (int)nn->width[i]; j++) {
      nn->loss[i][j] = error_derivative(targets[j], nn->neuron[i][j]) * activation_function[nn->activation[i]](nn->preact[i][j], true);
    }
  }
  // Backpropagate loss into earlier layers
  for (i = nn->depth - 2; i > 0; i--) {
    if (nn->layer_type[i] == LAYER_TYPE_LSTM) {
      // An LSTM layer's own backward pass (nn_lstm_backward()) needs
      // dL/dh_t -- the raw gradient flowing in from layer i+1 -- routed
      // through its four internal gates instead of the generic
      // "* activation_derivative(preact[i][j])" multiply every branch
      // below finishes with (an LSTM's output is not a scalar function of
      // one preact per neuron the way every other layer type's is -- see
      // nn_lstm_backward()'s comment). So: gather that raw gradient first,
      // using the exact same per-(layer_type[i+1]) rules as the branches
      // below, just without their trailing activation-derivative multiply,
      // then hand it to nn_lstm_backward() instead.
      //
      // nn->loss[i] is repurposed here as scratch space for that gathered
      // gradient -- for an LSTM layer, it never holds a per-neuron loss
      // value the way it does for every other layer type;
      // nn->lstm_gate_grad[i] (computed by nn_lstm_backward() below) takes
      // over that role for the bias/weight-adjustment loops further down.
      float *dh = nn->loss[i];
      if (nn->layer_type[i + 1] == LAYER_TYPE_POOL) {
        memset(dh, 0, nn->width[i] * sizeof(float));
        nn_pool_backward(nn, i + 1, dh);
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_CNN) {
        memset(dh, 0, nn->width[i] * sizeof(float));
        nn_conv_backward(nn, i + 1, dh);
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_DROPOUT) {
        memset(dh, 0, nn->width[i] * sizeof(float));
        nn_dropout_backward(nn, i + 1, dh);
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_RNN) {
        const int row_len = (int)nn->width[i] + (int)nn->width[i + 1];
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < (int)nn->width[i + 1]; k++)
            sum += nn->loss[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_LSTM) {
        const int hidden_next = (int)nn->width[i + 1];
        const int row_len = (int)nn->width[i] + hidden_next;
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < 4 * hidden_next; k++)
            sum += nn->lstm_gate_grad[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_GRU) {
        const int hidden_next = (int)nn->width[i + 1];
        const int row_len = (int)nn->width[i] + hidden_next;
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < 3 * hidden_next; k++)
            sum += nn->gru_gate_grad[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      } else {
        const int row_len = (int)nn->width[i];
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < (int)nn->width[i + 1]; k++)
            sum += nn->loss[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      }
      nn_lstm_backward(nn, i, dh);
    } else if (nn->layer_type[i] == LAYER_TYPE_GRU) {
      // Same idea as the LSTM-owning branch just above (see its comment for
      // the full rationale): gather dL/dh_t using the same per-(next-layer-
      // type) rules, without any trailing activation-derivative multiply,
      // then hand it to nn_gru_backward() instead. nn->loss[i] is reused as
      // scratch space the same way.
      float *dh = nn->loss[i];
      if (nn->layer_type[i + 1] == LAYER_TYPE_POOL) {
        memset(dh, 0, nn->width[i] * sizeof(float));
        nn_pool_backward(nn, i + 1, dh);
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_CNN) {
        memset(dh, 0, nn->width[i] * sizeof(float));
        nn_conv_backward(nn, i + 1, dh);
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_DROPOUT) {
        memset(dh, 0, nn->width[i] * sizeof(float));
        nn_dropout_backward(nn, i + 1, dh);
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_RNN) {
        const int row_len = (int)nn->width[i] + (int)nn->width[i + 1];
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < (int)nn->width[i + 1]; k++)
            sum += nn->loss[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_LSTM) {
        const int hidden_next = (int)nn->width[i + 1];
        const int row_len = (int)nn->width[i] + hidden_next;
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < 4 * hidden_next; k++)
            sum += nn->lstm_gate_grad[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      } else if (nn->layer_type[i + 1] == LAYER_TYPE_GRU) {
        const int hidden_next = (int)nn->width[i + 1];
        const int row_len = (int)nn->width[i] + hidden_next;
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < 3 * hidden_next; k++)
            sum += nn->gru_gate_grad[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      } else {
        const int row_len = (int)nn->width[i];
        for (j = 0; j < (int)nn->width[i]; j++) {
          sum = 0.0f;
          for (k = 0; k < (int)nn->width[i + 1]; k++)
            sum += nn->loss[i + 1][k] * nn->weight[i + 1][k * row_len + j];
          dh[j] = sum;
        }
      }
      nn_gru_backward(nn, i, dh);
    } else if (nn->layer_type[i + 1] == LAYER_TYPE_POOL) {
      // Pooling has no weight matrix -- route/distribute the gradient
      // directly according to the pooling type instead of the generic
      // weighted-sum formula below. Accumulate routed gradients straight
      // into loss[i] (zeroed first), then apply layer i's own activation
      // derivative in place, same as the generic case does.
      memset(nn->loss[i], 0, nn->width[i] * sizeof(float));
      nn_pool_backward(nn, i + 1, nn->loss[i]);
      for (j = 0; j < (int)nn->width[i]; j++) {
        nn->loss[i][j] *= activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    } else if (nn->layer_type[i + 1] == LAYER_TYPE_CNN) {
      // Same idea as the POOL case above, but routed through the CNN
      // layer's kernels (nn_conv_backward()) instead of a pooling rule --
      // the generic weighted-sum formula below assumes a flat FC-style
      // weight matrix, which a CNN layer's weight buffer is not.
      memset(nn->loss[i], 0, nn->width[i] * sizeof(float));
      nn_conv_backward(nn, i + 1, nn->loss[i]);
      for (j = 0; j < (int)nn->width[i]; j++) {
        nn->loss[i][j] *= activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    } else if (nn->layer_type[i + 1] == LAYER_TYPE_DROPOUT) {
      // Same idea again: a DROPOUT layer has no weight matrix (it's a
      // same-width pass-through), so the generic weighted-sum formula below
      // does not apply -- route through nn_dropout_backward() instead,
      // which re-applies the same per-neuron scale the forward pass used.
      memset(nn->loss[i], 0, nn->width[i] * sizeof(float));
      nn_dropout_backward(nn, i + 1, nn->loss[i]);
      for (j = 0; j < (int)nn->width[i]; j++) {
        nn->loss[i][j] *= activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    } else if (nn->layer_type[i + 1] == LAYER_TYPE_RNN) {
      // An RNN layer's flat weight buffer has a row per its own neuron, but
      // (unlike the plain FC/OUTPUT case below) each row is row_len =
      // width[i] + width[i+1] wide: the first width[i] columns are the
      // input-to-hidden weights against OUR neurons (what we need here),
      // and the remaining width[i+1] columns are that layer's own
      // recurrent/hidden-to-hidden weights, applied against ITS previous
      // hidden state -- not ours. Only the input-to-hidden columns
      // contribute to our loss; the recurrent columns are deliberately
      // skipped (truncated BPTT depth 1 -- see LAYER_TYPE_RNN's comment in
      // nn.h and forward_propagation()'s LAYER_TYPE_RNN case).
      const int row_len = (int)nn->width[i] + (int)nn->width[i + 1];
      for (j = 0; j < (int)nn->width[i]; j++) {
        sum = 0.0f;
        for (k = 0; k < (int)nn->width[i + 1]; k++) {
          sum += nn->loss[i + 1][k] * nn->weight[i + 1][k * row_len + j];
        }
        nn->loss[i][j] = sum * activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    } else if (nn->layer_type[i + 1] == LAYER_TYPE_LSTM) {
      // Same idea as the RNN-next-layer case above, but layer i+1's flat
      // weight buffer has 4*width[i+1] rows (one per gate per hidden unit
      // -- see quantized_layer_shape()) and its gate gradients live in
      // lstm_gate_grad[i+1], not loss[i+1]. Only the input-to-hidden
      // columns of each of the four gates' rows contribute to our loss;
      // the recurrent columns are deliberately skipped, same rationale as
      // the RNN case (truncated BPTT depth 1).
      const int hidden_next = (int)nn->width[i + 1];
      const int row_len = (int)nn->width[i] + hidden_next;
      for (j = 0; j < (int)nn->width[i]; j++) {
        sum = 0.0f;
        for (k = 0; k < 4 * hidden_next; k++) {
          sum += nn->lstm_gate_grad[i + 1][k] * nn->weight[i + 1][k * row_len + j];
        }
        nn->loss[i][j] = sum * activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    } else if (nn->layer_type[i + 1] == LAYER_TYPE_GRU) {
      // Same idea as the LSTM-next-layer case above, but layer i+1's flat
      // weight buffer has 3*width[i+1] rows (one per gate per hidden unit
      // -- see quantized_layer_shape()) and its gate gradients live in
      // gru_gate_grad[i+1], not loss[i+1]. Only the input-to-hidden columns
      // of each of the three gates' rows contribute to our loss; the
      // recurrent columns are deliberately skipped, same rationale as the
      // RNN/LSTM cases (truncated BPTT depth 1).
      const int hidden_next = (int)nn->width[i + 1];
      const int row_len = (int)nn->width[i] + hidden_next;
      for (j = 0; j < (int)nn->width[i]; j++) {
        sum = 0.0f;
        for (k = 0; k < 3 * hidden_next; k++) {
          sum += nn->gru_gate_grad[i + 1][k] * nn->weight[i + 1][k * row_len + j];
        }
        nn->loss[i][j] = sum * activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    } else {
      // Layer i+1's flat weight buffer has a row per its own neuron, each
      // row_len = width[i] wide (its previous layer's width, i.e. ours).
      const int row_len = (int)nn->width[i];
      for (j = 0; j < (int)nn->width[i]; j++) {
        sum = 0.0f;
        for (k = 0; k < (int)nn->width[i + 1]; k++) {
          // loss[i+1][k] already has layer i+1's own activation derivative
          // baked in (it was applied when loss[i+1] was computed, whether
          // at output-layer init above or in this same branch one
          // recursion level up) -- do not re-apply it here.
          sum += nn->loss[i + 1][k] * nn->weight[i + 1][k * row_len + j];
        }
        // The chain rule dictates that we should multiply the summed loss by the
        // derivative of the activation at the current neuron, not only during
        // weight updates, but immediately when calculating loss[i][j].
        nn->loss[i][j] = sum * activation_function[nn->activation[i]](nn->preact[i][j], true);
      }
    }
  }
  // Adam's step counter: one increment per nn_train() call, shared across
  // every layer's update within this call (see its comment in nn.h). Cheap
  // to bump unconditionally; only ever read back under NN_OPTIMIZER_ADAM.
  nn->adam_step++;
  // Update biases (optimizer step -- SGD/MOMENTUM/ADAM, see nn_optimizer_apply())
  for (i = 1; i < (int)nn->depth; i++) {
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      cnn_t *cnn = nn->config[i];
      int out_c  = cnn->out_channels;
      int x_out  = cnn->out_w;
      int y_out  = cnn->out_h;
      int plane  = x_out * y_out;
      float db[256]; // out_channels is a uint8_t (max 255)
      for (j = 0; j < out_c; ++j) {
        db[j] = 0.0f;
        for (k = 0; k < plane; ++k)
          db[j] += nn->loss[i][j * plane + k];
      }
      nn_optimizer_apply(nn, nn->bias[i], db, nn->bias_moment1[i], nn->bias_moment2[i], out_c, rate);
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL || nn->layer_type[i] == LAYER_TYPE_DROPOUT) {
      // Pooling and dropout both have no bias
    } else if (nn->layer_type[i] == LAYER_TYPE_LSTM) {
      // 4*hidden biases (one per gate per hidden unit); the gradient for
      // each comes from lstm_gate_grad[i], not loss[i] (which is repurposed
      // as scratch space for this layer type -- see the backprop loop above).
      nn_optimizer_apply(nn, nn->bias[i], nn->lstm_gate_grad[i], nn->bias_moment1[i], nn->bias_moment2[i], 4 * (int)nn->width[i], rate);
    } else if (nn->layer_type[i] == LAYER_TYPE_GRU) {
      // 3*hidden biases (one per gate per hidden unit); the gradient for
      // each comes from gru_gate_grad[i], not loss[i] (which is repurposed
      // as scratch space for this layer type -- see the backprop loop above).
      nn_optimizer_apply(nn, nn->bias[i], nn->gru_gate_grad[i], nn->bias_moment1[i], nn->bias_moment2[i], 3 * (int)nn->width[i], rate);
    } else {
      // FC / output layers
      nn_optimizer_apply(nn, nn->bias[i], nn->loss[i], nn->bias_moment1[i], nn->bias_moment2[i], (int)nn->width[i], rate);
    }
  }
  // Calculate the weight adjustments. Note that their update is delayed until
  // after full backprop traversal. The weights cannot be updated while
  // back-propagating, because back propagating each layer depends on the next
  // layer's weights. So we save the weight adjustments in a temporary array and
  // apply them all at once later.
  // Compute weight adjustments (store in weight_adj)
  for (i = (int)nn->depth - 1; i > 0; i--) {
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      // Convolution gradient
      cnn_t *cnn = nn->config[i];
      int in_c   = cnn->in_channels;
      int out_c  = cnn->out_channels;
      int ksize  = cnn->kernel_size;
      int x_out  = cnn->out_w;
      int y_out  = cnn->out_h;
      int plane_out = x_out * y_out;
      int in_plane  = cnn->in_w * cnn->in_h;
      for (int oc = 0; oc < out_c; ++oc) {
        for (int ic = 0; ic < in_c; ++ic) {
          float *adj = nn->weight_adj[i] + (oc * in_c + ic) * ksize * ksize;
          memset(adj, 0, ksize * ksize * sizeof(float));
          const float *base = nn->neuron[i - 1] + ic * in_plane;
          for (int oy = 0; oy < y_out; ++oy) {
            // Clip the kernel-tap range to the sub-window that overlaps a
            // real (unpadded) input row/column, same technique as
            // nn_conv2d()/nn_conv_backward() -- a tap outside this range
            // multiplied an implicit zero (padding) during the forward
            // pass, and so contributes nothing to this weight's gradient.
            int in_y0 = oy * cnn->stride - cnn->padding;
            int ky_start = in_y0 < 0 ? -in_y0 : 0;
            int ky_end = (in_y0 + ksize > cnn->in_h) ? (cnn->in_h - in_y0) : ksize;
            for (int ox = 0; ox < x_out; ++ox) {
              int in_x0 = ox * cnn->stride - cnn->padding;
              int kx_start = in_x0 < 0 ? -in_x0 : 0;
              int kx_end = (in_x0 + ksize > cnn->in_w) ? (cnn->in_w - in_x0) : ksize;
              float delta = nn->loss[i][oc * plane_out + oy * x_out + ox];
              for (int ky = ky_start; ky < ky_end; ++ky) {
                const float *src = base + (in_y0 + ky) * cnn->in_w + in_x0;
                for (int kx = kx_start; kx < kx_end; ++kx) {
                  adj[ky * ksize + kx] += delta * src[kx];
                }
              }
            }
          }
        }
      }
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL || nn->layer_type[i] == LAYER_TYPE_DROPOUT) {
      // Pooling and dropout both have no weights
    } else if (nn->layer_type[i] == LAYER_TYPE_RNN) {
      // Same row layout as forward_propagation()'s LAYER_TYPE_RNN case:
      // input-to-hidden columns first (gradient w.r.t. our previous layer's
      // current-timestep output, exactly like the FC/OUTPUT case below),
      // then hidden-to-hidden/recurrent columns (gradient w.r.t. THIS
      // layer's own hidden state from the *previous* timestep --
      // rnn_hidden_prev[i], cached by forward_propagation() before it got
      // overwritten with this timestep's state; nn->neuron[i] itself no
      // longer holds that value by this point in nn_train()). The recurrent
      // weight is trained (this is its only gradient term -- truncated BPTT
      // depth 1), but no gradient is propagated further back through it.
      const int row_len_in = (int)nn->width[i - 1];
      const int hidden = (int)nn->width[i];
      const int row_len = row_len_in + hidden;
      const float *prev = nn->rnn_hidden_prev[i];
      for (j = 0; j < hidden; j++) {
        for (k = 0; k < row_len_in; k++)
          nn->weight_adj[i][j * row_len + k] = nn->loss[i][j] * nn->neuron[i - 1][k];
        for (k = 0; k < hidden; k++)
          nn->weight_adj[i][j * row_len + row_len_in + k] = nn->loss[i][j] * prev[k];
      }
    } else if (nn->layer_type[i] == LAYER_TYPE_LSTM) {
      // Same idea as the RNN case above, but for all 4*hidden rows (one per
      // gate per hidden unit -- see quantized_layer_shape()) at once: the
      // gradient for row `j` (regardless of which gate it belongs to) comes
      // from lstm_gate_grad[i][j], and h_prev (this layer's hidden state
      // from the *previous* timestep, cached in lstm_cache[i] before this
      // call's forward pass overwrote it) is shared across every gate/row,
      // not just the one it happens to belong to.
      const int row_len_in = (int)nn->width[i - 1];
      const int hidden = (int)nn->width[i];
      const int row_len = row_len_in + hidden;
      const float *h_prev = nn->lstm_cache[i] + 1 * hidden;
      const float *grad = nn->lstm_gate_grad[i];
      for (j = 0; j < 4 * hidden; j++) {
        for (k = 0; k < row_len_in; k++)
          nn->weight_adj[i][j * row_len + k] = grad[j] * nn->neuron[i - 1][k];
        for (k = 0; k < hidden; k++)
          nn->weight_adj[i][j * row_len + row_len_in + k] = grad[j] * h_prev[k];
      }
    } else if (nn->layer_type[i] == LAYER_TYPE_GRU) {
      // Same idea as the LSTM case above (rows [0,hidden)=reset,
      // [hidden,2*hidden)=update, [2*hidden,3*hidden)=candidate -- see
      // quantized_layer_shape()), EXCEPT the candidate gate's recurrent
      // columns need an extra factor of the reset gate's own output r[j]:
      // its preact is W_n x + r*(U_n h_prev) + b_n (see
      // forward_propagation()'s LAYER_TYPE_GRU case), so
      // d(preact)/d(U_n[j,k]) = r[j] * h_prev[k], not just h_prev[k] the
      // way it is for the reset/update gates' own recurrent weights.
      const int row_len_in = (int)nn->width[i - 1];
      const int hidden = (int)nn->width[i];
      const int row_len = row_len_in + hidden;
      const float *h_prev = nn->gru_cache[i] + 0 * hidden;
      const float *gate_r = nn->gru_cache[i] + 1 * hidden;
      const float *grad = nn->gru_gate_grad[i];
      for (j = 0; j < 3 * hidden; j++) {
        for (k = 0; k < row_len_in; k++)
          nn->weight_adj[i][j * row_len + k] = grad[j] * nn->neuron[i - 1][k];
        // j / hidden: 0=reset, 1=update, 2=candidate; j % hidden: which
        // hidden unit's row within that gate.
        float recur_scale = (j / hidden == 2) ? grad[j] * gate_r[j % hidden] : grad[j];
        for (k = 0; k < hidden; k++)
          nn->weight_adj[i][j * row_len + row_len_in + k] = recur_scale * h_prev[k];
      }
    } else {
      // FC / output layers
      const int row_len = (int)nn->width[i - 1];
      for (j = 0; j < (int)nn->width[i]; j++)
        for (k = 0; k < row_len; k++)
          nn->weight_adj[i][j * row_len + k] = nn->loss[i][j] * nn->neuron[i - 1][k];
    }
  }
  // Apply weight adjustments (optimizer step -- SGD/MOMENTUM/ADAM, see
  // nn_optimizer_apply()). Once weight_adj is filled, this pass is
  // identical mechanically regardless of which layer type produced it --
  // just a different element count -- so every branch below only computes
  // `total` and defers the actual update to the shared helper.
  for (i = (int)nn->depth - 1; i > 0; i--) {
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      cnn_t *cnn = nn->config[i];
      int kernels = cnn->out_channels * cnn->in_channels;
      int k_elems = cnn->kernel_size * cnn->kernel_size;
      int total = kernels * k_elems;
      nn_optimizer_apply(nn, nn->weight[i], nn->weight_adj[i], nn->weight_moment1[i], nn->weight_moment2[i], total, rate);
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL || nn->layer_type[i] == LAYER_TYPE_DROPOUT) {
      // Pooling and dropout both have no weights
    } else if (nn->layer_type[i] == LAYER_TYPE_RNN) {
      // Row length includes both the input-to-hidden and recurrent halves
      // (see quantized_layer_shape()); weight_adj was filled with both
      // above, so a single flat update over the whole row-major buffer
      // updates both halves correctly.
      int total = (int)nn->width[i] * ((int)nn->width[i - 1] + (int)nn->width[i]);
      nn_optimizer_apply(nn, nn->weight[i], nn->weight_adj[i], nn->weight_moment1[i], nn->weight_moment2[i], total, rate);
    } else if (nn->layer_type[i] == LAYER_TYPE_LSTM) {
      // Same idea as RNN above, but 4x the rows (one per gate per hidden
      // unit -- see quantized_layer_shape()).
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, i, &rows, &row_len, &bias_count);
      int total = rows * row_len;
      nn_optimizer_apply(nn, nn->weight[i], nn->weight_adj[i], nn->weight_moment1[i], nn->weight_moment2[i], total, rate);
    } else if (nn->layer_type[i] == LAYER_TYPE_GRU) {
      // Same idea as RNN/LSTM above, but 3x the rows (one per gate per
      // hidden unit -- see quantized_layer_shape()).
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, i, &rows, &row_len, &bias_count);
      int total = rows * row_len;
      nn_optimizer_apply(nn, nn->weight[i], nn->weight_adj[i], nn->weight_moment1[i], nn->weight_moment2[i], total, rate);
    } else {
      // FC / output layers
      int total = (int)nn->width[i] * (int)nn->width[i - 1];
      nn_optimizer_apply(nn, nn->weight[i], nn->weight_adj[i], nn->weight_moment1[i], nn->weight_moment2[i], total, rate);
    }
  }
  // Return the pre-update error computed above
  return err;
}

// Returns an output prediction given an input.
float *nn_predict(nn_t *nn, float *inputs)
{
  nn->neuron[0] = inputs;
  // training=false: any DROPOUT layer is a pass-through at inference.
  forward_propagation(nn, false);
  // Return the output layer
  return nn->neuron[nn->depth - 1];
}

// Zeros every RNN layer's hidden state (see LAYER_TYPE_RNN's comment in
// nn.h). Call this before feeding the first timestep of a new, independent
// sequence -- otherwise the previous sequence's final hidden state would
// leak into the next one.
void nn_reset_state(nn_t *nn)
{
  if (!nn)
    return;
  for (int layer = 1; layer < (int)nn->depth; layer++) {
    if (nn->layer_type[layer] == LAYER_TYPE_RNN || nn->layer_type[layer] == LAYER_TYPE_LSTM ||
        nn->layer_type[layer] == LAYER_TYPE_GRU)
      memset(nn->neuron[layer], 0, (size_t)nn->width[layer] * sizeof(float));
    if (nn->layer_type[layer] == LAYER_TYPE_LSTM)
      memset(nn->lstm_cell[layer], 0, (size_t)nn->width[layer] * sizeof(float));
    // Also invalidates any DROPOUT layer's held mask (see
    // dropout_mask_valid's comment in nn.h) -- this call is already the
    // "a new, independent sequence starts here" signal for RNN/GRU/LSTM
    // state above, so it's the same boundary a recurrent network's dropout
    // mask needs to refresh at.
    if (nn->layer_type[layer] == LAYER_TYPE_DROPOUT)
      nn->dropout_mask_valid[layer] = false;
  }
}

nn_error_t nn_set_optimizer(nn_t *nn, nn_optimizer_t optimizer, float momentum, float beta2, float epsilon)
{
  if (!nn) {
    return NN_ERROR_INVALID_ARGUMENT;
  }
  if (nn->immutable) {
    // Same reasoning as nn_train() itself: nothing writable to optimize.
    return NN_ERROR_READ_ONLY_MODEL;
  }
  nn->optimizer = optimizer;
  nn->optimizer_momentum = momentum;
  nn->optimizer_beta2 = beta2;
  nn->optimizer_epsilon = epsilon;
  // Any prior optimizer's accumulated history no longer applies (the
  // buffers are about to be freed/reallocated below anyway) -- and a fresh
  // Adam run should always get its own proper warmup rather than picking up
  // a stale step count, even when switching back to ADAM after a detour
  // through another optimizer.
  nn->adam_step = 0;
  // A quantized model has no moment buffers at all right now (nn_quantize()
  // already freed them) and can't train until dequantized -- nn_dequantize()
  // will call nn_optimizer_alloc_layer() itself at that point, using
  // whatever `optimizer`/hyperparameters were just set here. Nothing further
  // to allocate immediately in that case.
  if (nn->quantized) {
    return NN_ERROR_NONE;
  }
  for (int layer = 1; layer < (int)nn->depth; layer++) {
    if (!nn_optimizer_alloc_layer(nn, layer)) {
      return NN_ERROR_OUT_OF_MEMORY;
    }
  }
  return NN_ERROR_NONE;
}

// Loads a neural net model from a file.
nn_t *nn_load_model_ascii(const char *path)
{
  FILE *file = fopen(path, "r");
  if (!file)
    return NULL;
  nn_t *nn = nn_init();
  if (!nn) {
    fclose(file);
    return NULL;
  }
  // Read quantized flag
  int quant_flag = 0;
  if (fscanf(file, "%d\n", &quant_flag) != 1) {
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  nn->quantized = (quant_flag != 0);
  // Read model version, use %hhu since version_* are uint8_t.
  if (fscanf(file, "%hhu %hhu %hhu %hhu\n", &nn->version_major, &nn->version_minor, &nn->version_patch, &nn->version_build) != 4) {
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  // Read depth
  int depth = 0;
  if (fscanf(file, "%d\n", &depth) != 1) {
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  // Call nn_add_layer for each (layer_type, width, activation)
  for (int i = 0; i < depth; i++) {
    int layer_type, w, act;
    // peek layer_type first to know if cnn_t follows
    if (fscanf(file, "%d", &layer_type) != 1) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
    if (fscanf(file, "%d %d", &w, &act) != 2) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
    cnn_t ctmp;
    pool_t ptmp;
    dropout_t dtmp;
    void *cptr = NULL;
    if (layer_type == LAYER_TYPE_CNN) {
      if (fscanf(file, " %hu %hu %hhu %hhu %hhu %hhu %hhu %hhu", &ctmp.in_h, &ctmp.in_w, &ctmp.in_channels, &ctmp.out_channels, &ctmp.kernel_size, &ctmp.stride, &ctmp.padding, &ctmp.dilation) != 8) {
        fclose(file);
        nn_free(nn);
        return NULL;
      }
      cptr = &ctmp;
      // nn_add_layer will recompute width
      w = 0;
    } else if (layer_type == LAYER_TYPE_POOL) {
      int pt;
      if (fscanf(file, " %hu %hu %hhu %hhu %hhu %d", &ptmp.in_h, &ptmp.in_w, &ptmp.channels, &ptmp.pool_size, &ptmp.stride, &pt) != 6) {
        fclose(file);
        nn_free(nn);
        return NULL;
      }
      ptmp.pooling_type = (pooling_type_t)pt;
      cptr = &ptmp;
      // nn_add_layer will recompute width
      w = 0;
    } else if (layer_type == LAYER_TYPE_DROPOUT) {
      if (fscanf(file, " %f", &dtmp.rate) != 1) {
        fclose(file);
        nn_free(nn);
        return NULL;
      }
      cptr = &dtmp;
      // nn_add_layer will recompute width
      w = 0;
    }
    // Consume '\n'
    fgetc(file);
    if (nn_add_layer(nn, layer_type, w, act, cptr) != 0) {
      fclose(file);
      nn_free(nn);
      return NULL;
    }
  }
  // Note: neuron/loss/preact for every layer >= 1 were already allocated by
  // nn_add_layer() inside the layer-construction loop above; re-allocating
  // them here would just leak those allocations.
  // If float mode, read floats into nn->weight / nn->bias
  if (!nn->quantized) {
    for (int layer = 1; layer < nn->depth; layer++) {
      float dummy_scale;
      // Skip bias_scale (0)
      if (fscanf(file, "%f\n", &dummy_scale) != 1) {
        goto cleanup_float;
      }
      if (nn->layer_type[layer] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[layer];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        // Extra dummy line that the saver emits before the kernels
        if (fscanf(file, "%f\n", &dummy_scale) != 1)
          goto cleanup_float;
        // kernels
        for (int k = 0; k < kernels; ++k) {
          // per-kernel weight-scale placeholder
          if (fscanf(file, "%f\n", &dummy_scale) != 1)
            goto cleanup_float;
          for (int e = 0; e < k_elems; ++e) {
            if (fscanf(file, "%f\n", &nn->weight[layer][k * k_elems + e]) != 1)
              goto cleanup_float;
          }
        }
        // One bias per output channel
        for (int oc = 0; oc < c->out_channels; ++oc) {
          if (fscanf(file, "%f\n", &nn->bias[layer][oc]) != 1)
            goto cleanup_float;
        }
      } else if (nn->layer_type[layer] == LAYER_TYPE_POOL || nn->layer_type[layer] == LAYER_TYPE_DROPOUT) {
        // Pooling and dropout both have no weights/bias beyond the placeholder line already consumed above
      } else {
        // Fully-connected / output / RNN layer. row_len comes from
        // quantized_layer_shape() (not a hardcoded width[layer-1]) so this
        // handles an RNN layer's wider row (input-to-hidden + recurrent
        // columns, see that function's comment) the same as FC/OUTPUT's
        // plain row -- both are just a flat row_len-wide read per neuron.
        int rows, row_len, bias_count;
        quantized_layer_shape(nn, layer, &rows, &row_len, &bias_count);
        // `rows` (not width[layer]) is the right loop bound here: for
        // FC/OUTPUT/RNN it happens to equal width[layer], but for LSTM
        // rows == 4*width[layer] (four gates' worth of rows -- see that
        // function's comment) and bias_count == rows too, so iterating
        // `rows` reads every gate's weights and biases instead of just the
        // first quarter of them.
        for (int i = 0; i < rows; i++) {
          // Skip weight_scale (0)
          if (fscanf(file, "%f\n", &dummy_scale) != 1)
            goto cleanup_float;
          for (int j = 0; j < row_len; j++) {
            if (fscanf(file, "%f\n", &nn->weight[layer][i * row_len + j]) != 1)
              goto cleanup_float;
          }
          if (fscanf(file, "%f\n", &nn->bias[layer][i]) != 1)
            goto cleanup_float;
        }
      }
    }
    fclose(file);
    return nn;
  cleanup_float:
    fclose(file);
    nn_free(nn);
    return NULL;
  }
  // Otherwise: quantized == true
  // Free all float-side allocations made by nn_add_layer (weight, weight_adj, bias)
  for (int layer = 1; layer < nn->depth; layer++) {
    // weight/weight_adj are each one flat buffer per layer (or NULL for
    // POOL); free(NULL) is a no-op.
    free(nn->weight[layer]);
    free(nn->weight_adj[layer]);
    free(nn->bias[layer]);
  }
  free(nn->weight);
  free(nn->weight_adj);
  free(nn->bias);
  // Note: we keep nn->neuron/loss/preact (allocated above) for use in nn_predict
  // Null out float-side pointers that won't be used for weights/biases
  nn->weight = NULL;
  nn->weight_adj = NULL;
  nn->bias = NULL;
  // weight_scale and bias_scale will be replaced by quantized arrays
  for (int layer = 1; layer < nn->depth; layer++)
    free(nn->weight_scale[layer]);
  free(nn->weight_scale);
  free(nn->bias_scale);
  nn->weight_scale = NULL;
  nn->bias_scale = NULL;
  // Allocate top-level arrays for quantized model
  nn->weight_quantized = (int8_t **)malloc(sizeof(int8_t *) * nn->depth);
  nn->weight_scale = (float **)malloc(sizeof(float *) * nn->depth);
  nn->bias_quantized = (int8_t **)malloc(sizeof(int8_t *) * nn->depth);
  nn->bias_scale = (float *)malloc(sizeof(float) * nn->depth);
  if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized || !nn->bias_scale) {
    goto cleanup_quant_top;
  }
  // Initialize layer 0 entries
  nn->weight_quantized[0] = NULL;
  nn->weight_scale[0] = NULL;
  nn->bias_quantized[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  // Read quantized data, layer by layer. Each layer's weight_quantized is
  // now one flat, row-major buffer (not an array of per-row pointers), so
  // there is no per-row allocation bookkeeping to unwind on failure --
  // every layer's fields are either NULL or one single valid allocation.
  int layer = 0;
  for (layer = 1; layer < nn->depth; layer++) {
    if (nn->layer_type[layer] == LAYER_TYPE_POOL || nn->layer_type[layer] == LAYER_TYPE_DROPOUT) {
      // Pooling and dropout both have no weights/bias to read
      nn->weight_quantized[layer] = NULL;
      nn->weight_scale[layer] = NULL;
      nn->bias_quantized[layer] = NULL;
      nn->bias_scale[layer] = 0.0f;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, layer, &rows, &row_len, &bias_count);
    nn->weight_quantized[layer] = (int8_t *)malloc((size_t)rows * row_len * sizeof(int8_t));
    nn->weight_scale[layer] = (float *)malloc(sizeof(float) * rows);
    nn->bias_quantized[layer] = (int8_t *)malloc(sizeof(int8_t) * bias_count);
    if (!nn->weight_quantized[layer] || !nn->weight_scale[layer] || !nn->bias_quantized[layer]) {
      goto cleanup_quant_per_layer;
    }
    // For each row (neuron or kernel), read weight_scale + quantized weights
    for (int row = 0; row < rows; row++) {
      if (fscanf(file, "%f\n", &nn->weight_scale[layer][row]) != 1) {
        goto cleanup_quant_per_layer;
      }
      for (int w = 0; w < row_len; w++) {
        int int_w;
        if (fscanf(file, "%d\n", &int_w) != 1) {
          goto cleanup_quant_per_layer;
        }
        nn->weight_quantized[layer][row * row_len + w] = (int8_t)int_w;
      }
    }
    // Read bias_scale[layer]
    if (fscanf(file, "%f\n", &nn->bias_scale[layer]) != 1) {
      goto cleanup_quant_per_layer;
    }
    // Read each bias (one per output channel for CNN, one per neuron for FC/OUTPUT)
    for (int b = 0; b < bias_count; b++) {
      int int_b;
      if (fscanf(file, "%d\n", &int_b) != 1) {
        goto cleanup_quant_per_layer;
      }
      nn->bias_quantized[layer][b] = (int8_t)int_b;
    }
  }
  fclose(file);
  return nn;
  // If we failed before allocating the top-level quantized arrays:
cleanup_quant_top:
  free(nn->weight_quantized);
  free(nn->weight_scale);
  free(nn->bias_quantized);
  free(nn->bias_scale);
  // NULL these out before nn_free(): nn_free()'s quantized branch would
  // otherwise index through these same (now-freed) top-level pointers
  // again, since it doesn't know we already tore them down here.
  nn->weight_quantized = NULL;
  nn->weight_scale = NULL;
  nn->bias_quantized = NULL;
  nn->bias_scale = NULL;
  fclose(file);
  nn_free(nn);
  return NULL;
  // If we failed partway through allocating/reading layer `layer` (or an
  // earlier layer had already been fully read): every layer 1..layer is
  // either NULL or a single valid flat allocation, so a plain free() over
  // each (safe as a no-op on NULL) is enough -- no per-row unwinding needed.
cleanup_quant_per_layer:
  for (int L = 1; L <= layer && L < nn->depth; L++) {
    free(nn->weight_quantized[L]);
    free(nn->weight_scale[L]);
    free(nn->bias_quantized[L]);
  }
  goto cleanup_quant_top;
}

// Minimal reader abstraction so the binary model format only needs to be
// parsed once, whether the bytes come from an open FILE* (nn_load_model_binary(),
// reading from disk) or directly from a caller-supplied buffer
// (nn_load_model_memory(), e.g. a model baked into flash on a microcontroller
// with no filesystem).
typedef struct {
  FILE *file;           // Non-NULL when reading from a file
  const uint8_t *buf;   // Non-NULL when reading from a memory buffer
  size_t buf_len;
  size_t buf_pos;
} nn_reader_t;

static bool nn_reader_read(nn_reader_t *r, void *dst, size_t n)
{
  if (r->file)
    return fread(dst, 1, n, r->file) == n;
  if (n > r->buf_len - r->buf_pos)
    return false;
  memcpy(dst, r->buf + r->buf_pos, n);
  r->buf_pos += n;
  return true;
}

// Advances a reader past `n` bytes without copying them. Used only for
// skipping the "inplace" format's alignment padding (see nn_load_model_inplace()),
// where the skipped bytes themselves are never meaningful.
static bool nn_reader_skip(nn_reader_t *r, size_t n)
{
  if (r->file)
    return fseek(r->file, (long)n, SEEK_CUR) == 0;
  if (n > r->buf_len - r->buf_pos)
    return false;
  r->buf_pos += n;
  return true;
}

// Reads `n` bytes into `dst` (copying, like nn_reader_read()), then skips
// forward to the next 4-byte boundary -- the padding nn_save_model_inplace()
// writes after every field so the fields that follow (in particular, the
// blocks nn_reader_alias_padded() below hands back as raw pointers) always
// start 4-byte aligned relative to the start of the buffer.
static bool nn_reader_read_padded(nn_reader_t *r, void *dst, size_t n)
{
  if (!nn_reader_read(r, dst, n))
    return false;
  size_t pad = (4 - (n % 4)) % 4;
  return pad == 0 || nn_reader_skip(r, pad);
}

// Hands back a pointer to the next `n` bytes of a memory-backed reader
// without copying them, advancing the read position past them; false if
// fewer than n bytes remain. Only valid for memory-backed readers (r->file
// must be NULL) -- a FILE*-backed stream has no stable address to alias.
// This is what gives nn_load_model_inplace() its zero-copy weight/bias
// arrays: the returned pointer aliases directly into the caller's buffer.
static bool nn_reader_alias(nn_reader_t *r, const void **out, size_t n)
{
  if (r->file || n > r->buf_len - r->buf_pos)
    return false;
  *out = r->buf + r->buf_pos;
  r->buf_pos += n;
  return true;
}

// Same as nn_reader_alias(), but also skips the trailing alignment padding
// (see nn_reader_read_padded() above).
static bool nn_reader_alias_padded(nn_reader_t *r, const void **out, size_t n)
{
  if (!nn_reader_alias(r, out, n))
    return false;
  size_t pad = (4 - (n % 4)) % 4;
  return pad == 0 || nn_reader_skip(r, pad);
}

// Obtains `n` bytes (plus alignment padding) from a memory-backed reader
// either by aliasing them directly (copy == false, used by
// nn_load_model_inplace() for zero-copy loading) or by allocating a fresh
// buffer and copying them into it (copy == true, used by
// nn_load_model_inplace_copy() to produce a normal, fully owned/mutable
// model instead). `*out` is set to the resulting pointer (aliased or
// owned) on success; the caller owns it in the copy case and must free()
// it eventually (nn_free() already does, via the same fields it would free
// for any other owned model).
static bool nn_reader_obtain(nn_reader_t *r, void **out, size_t n, bool copy)
{
  if (!copy) {
    const void *block;
    if (!nn_reader_alias_padded(r, &block, n))
      return false;
    *out = (void *)block; // see nn->immutable's doc comment in nn.h: the
                           // caller (nn_load_model_inplace) never writes
                           // through this despite the field's mutable type
    return true;
  }
  void *buf = n ? malloc(n) : NULL;
  if (n && !buf)
    return false;
  if (!nn_reader_read_padded(r, buf, n)) {
    free(buf);
    return false;
  }
  *out = buf;
  return true;
}

// Parses the binary model format described at the top of nn_save_model_binary()
// from `r`. Common to both nn_load_model_binary() and nn_load_model_memory().
static nn_t *nn_load_model_binary_impl(nn_reader_t *r)
{
  // Magic number
  uint8_t magic[NN_BINARY_MAGIC_LEN];
  if (!nn_reader_read(r, magic, NN_BINARY_MAGIC_LEN) ||
      memcmp(magic, NN_BINARY_MAGIC, NN_BINARY_MAGIC_LEN) != 0)
    return NULL;
  nn_t *nn = nn_init();
  if (!nn)
    return NULL;
  // Quantized flag
  uint8_t qflag;
  if (!nn_reader_read(r, &qflag, sizeof(qflag)))
    goto fail;
  nn->quantized = (qflag != 0);
  // Model version
  if (!nn_reader_read(r, &nn->version_major, sizeof(nn->version_major)))
    goto fail;
  if (!nn_reader_read(r, &nn->version_minor, sizeof(nn->version_minor)))
    goto fail;
  if (!nn_reader_read(r, &nn->version_patch, sizeof(nn->version_patch)))
    goto fail;
  if (!nn_reader_read(r, &nn->version_build, sizeof(nn->version_build)))
    goto fail;
  // Depth
  uint32_t depth;
  if (!nn_reader_read(r, &depth, sizeof(depth)))
    goto fail;
  // Read each layer's width, layer type, and activation and call nn_add_layer()
  for (uint32_t i = 0; i < depth; i++) {
    uint8_t layer_type;
    uint32_t w;
    uint8_t a;
    if (!nn_reader_read(r, &layer_type, sizeof(layer_type)))
      goto fail;
    if (!nn_reader_read(r, &w, sizeof(w)))
      goto fail;
    if (!nn_reader_read(r, &a, sizeof(a)))
      goto fail;
    cnn_t ctmp;
    pool_t ptmp;
    dropout_t dtmp;
    void *cptr = NULL;
    if (layer_type == LAYER_TYPE_CNN) {
      if (!nn_reader_read(r, &ctmp, sizeof(ctmp)))
        goto fail;
      cptr = &ctmp; w = 0;
    } else if (layer_type == LAYER_TYPE_POOL) {
      if (!nn_reader_read(r, &ptmp, sizeof(ptmp)))
        goto fail;
      cptr = &ptmp; w = 0;
    } else if (layer_type == LAYER_TYPE_DROPOUT) {
      if (!nn_reader_read(r, &dtmp, sizeof(dtmp)))
        goto fail;
      cptr = &dtmp; w = 0;
    }
    if (nn_add_layer(nn, layer_type, (int)w, (int)a, cptr) != 0)
     goto fail;
  }
  // Note: neuron/loss/preact for every layer >= 1 were already allocated by
  // nn_add_layer() inside the layer-construction loop above; re-allocating
  // them here would just leak those allocations.
  // Read weights & biases
  if (!nn->quantized) {
    // Float-mode: read dummy scales + real floats
    for (int L = 1; L < (int)depth; L++) {
      float dummy;
      // bias_scale placeholder
      if (!nn_reader_read(r, &dummy, sizeof(dummy)))
        goto fail;
      if (nn->layer_type[L] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[L];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        for (int k = 0; k < kernels; ++k) {
          // per-kernel weight-scale placeholder
          if (!nn_reader_read(r, &dummy, sizeof(dummy)))
            goto fail;
          if (!nn_reader_read(r, nn->weight[L] + k * k_elems, sizeof(float) * (size_t)k_elems))
            goto fail;
        }
        // One bias per output channel
        if (!nn_reader_read(r, nn->bias[L], sizeof(float) * (size_t)c->out_channels))
          goto fail;
      } else if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
        // Pooling and dropout both have no weights/bias beyond the placeholder read above
      } else {
        // FC / output / RNN -- row_len (named `prev` historically) comes
        // from quantized_layer_shape() so an RNN layer's wider row
        // (input-to-hidden + recurrent columns) is read the same way as
        // FC/OUTPUT's plain row; see that function's comment.
        int rows, prev, bias_count;
        quantized_layer_shape(nn, (int)L, &rows, &prev, &bias_count);
        // `rows` (not width[L]) is the right loop bound -- see the ascii
        // loader's identical comment for why (LSTM's rows == 4*width[L]).
        uint32_t curr = (uint32_t)rows;
        for (uint32_t i = 0; i < curr; i++) {
          // weight_scale placeholder
          if (!nn_reader_read(r, &dummy, sizeof(dummy)))
            goto fail;
          // weights
          if (!nn_reader_read(r, nn->weight[L] + i * prev, sizeof(float) * (size_t)prev))
            goto fail;
          // bias
          if (!nn_reader_read(r, &nn->bias[L][i], sizeof(float)))
            goto fail;
        }
      }
    }
  } else {
    // Quantized-mode: free float-side, allocate quantized arrays
    for (int L = 1; L < (int)depth; L++) {
      // weight/weight_adj are each one flat buffer per layer (or NULL for
      // POOL); free(NULL) is a no-op.
      free(nn->weight[L]);
      free(nn->weight_adj[L]);
      free(nn->bias[L]);
    }
    free(nn->weight);
    nn->weight = NULL;
    free(nn->weight_adj);
    nn->weight_adj = NULL;
    free(nn->bias);
    nn->bias = NULL;
    for (int L = 1; L < (int)depth; L++)
      free(nn->weight_scale[L]);
    free(nn->weight_scale);
    nn->weight_scale = NULL;
    free(nn->bias_scale);
    nn->bias_scale = NULL;
    // Top-level quant arrays
    nn->weight_quantized = malloc(depth * sizeof(int8_t *));
    nn->weight_scale = malloc(depth * sizeof(float *));
    nn->bias_quantized = malloc(depth * sizeof(int8_t *));
    nn->bias_scale = malloc(depth * sizeof(float));
    if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized ||
        !nn->bias_scale)
      goto fail;
    // Zero every layer's slot up front so that if a `goto fail` below
    // fires partway through the per-layer read loop, nn_free() can safely
    // free every layer -- including ones not reached yet -- instead of
    // indexing uninitialized garbage left over from this malloc.
    memset(nn->weight_quantized, 0, depth * sizeof(int8_t *));
    memset(nn->weight_scale, 0, depth * sizeof(float *));
    memset(nn->bias_quantized, 0, depth * sizeof(int8_t *));
    nn->bias_scale[0] = 0.0f;
    // Read per-layer quant data
    for (int L = 1; L < (int)depth; L++) {
      if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
        // Pooling and dropout both have no weights/bias to read (already NULL from the memsets above)
        nn->bias_scale[L] = 0.0f;
        continue;
      }
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
      // Allocate one flat, row-major buffer per layer
      nn->weight_quantized[L] = malloc((size_t)rows * row_len * sizeof(int8_t));
      nn->weight_scale[L] = malloc(rows * sizeof(float));
      nn->bias_quantized[L] = malloc(bias_count * sizeof(int8_t));
      if (!nn->weight_quantized[L] || !nn->weight_scale[L] ||
          !nn->bias_quantized[L])
        goto fail;
      // Read each row's (neuron or kernel) weight_scale and weights
      for (int i = 0; i < rows; i++) {
        if (!nn_reader_read(r, &nn->weight_scale[L][i], sizeof(float)))
          goto fail;
        if (!nn_reader_read(r, nn->weight_quantized[L] + i * row_len, sizeof(int8_t) * (size_t)row_len))
          goto fail;
      }
      // Read bias_scale[L]
      if (!nn_reader_read(r, &nn->bias_scale[L], sizeof(float)))
        goto fail;
      // Read quantized biases (one per output channel for CNN, one per neuron for FC/OUTPUT)
      if (!nn_reader_read(r, nn->bias_quantized[L], sizeof(int8_t) * (size_t)bias_count))
        goto fail;
    }
  }
  return nn;
fail:
  nn_free(nn);
  return NULL;
}

// Loads a neural-net model from a raw binary file.
nn_t *nn_load_model_binary(const char *path)
{
  FILE *file = fopen(path, "rb");
  if (!file)
    return NULL;
  nn_reader_t r = {.file = file};
  nn_t *nn = nn_load_model_binary_impl(&r);
  fclose(file);
  return nn;
}

// Loads a neural-net model from a binary-format buffer already resident in
// memory (e.g. a model baked into flash as a byte array on a microcontroller
// with no filesystem) instead of from a file. `data` must hold `size` bytes
// in the same format nn_save_model_binary() writes, and must stay valid only
// for the duration of this call -- the model is copied into its own
// allocations, so the buffer may be freed or reused immediately afterward.
// Intended for the inference path: load a model produced by nn_save_model_binary()
// (float or quantized) and use it with nn_predict(); the returned nn_t is
// otherwise identical to one loaded from a file and must still be released
// with nn_free().
nn_t *nn_load_model_memory(const uint8_t *data, size_t size)
{
  if (!data)
    return NULL;
  nn_reader_t r = {.buf = data, .buf_len = size};
  return nn_load_model_binary_impl(&r);
}

// Writes `n` bytes from `data`, then zero-pads to the next 4-byte boundary --
// the write-side counterpart of nn_reader_read_padded()/nn_reader_alias_padded(),
// used by nn_save_model_inplace() to keep every field in the "inplace"
// format aligned.
static bool nn_write_padded(FILE *file, const void *data, size_t n)
{
  if (n && fwrite(data, 1, n, file) != n)
    return false;
  size_t pad = (4 - (n % 4)) % 4;
  if (pad) {
    static const uint8_t zeros[4] = {0, 0, 0, 0};
    if (fwrite(zeros, 1, pad, file) != pad)
      return false;
  }
  return true;
}

// Writes a neural-net model in the "inplace" format (magic "NNP1"): a
// variant of the binary format tailored for nn_load_model_inplace()'s
// zero-copy loading. Every field is written at an offset that is a multiple
// of 4 bytes from the start of the file (fixed-size fields are themselves
// always a multiple of 4 bytes; the variable-length int8 arrays in a
// quantized model are zero-padded up to the next 4-byte boundary
// immediately after being written), and each layer's weight/bias data (or
// weight_scale/weight_quantized/bias_scale/bias_quantized data, for a
// quantized model) is written out contiguously, exactly matching the
// in-memory layout nn->weight[layer]/nn->bias[layer] (etc.) already use --
// see the comment on those fields in nn.h. That lets nn_load_model_inplace()
// simply alias a pointer into the middle of its input buffer for each of
// those arrays instead of parsing/copying them.
//
// Layout:
//   magic[4]                "NNP1"
//   uint32_t quantized_flag  0 or 1
//   uint32_t version         (major<<24)|(minor<<16)|(patch<<8)|build
//   uint32_t depth
//   -- per layer i in [0, depth): --
//     uint32_t layer_type
//     uint32_t width
//     uint32_t activation
//     cnn_t config            (only if layer_type == LAYER_TYPE_CNN)
//     pool_t config           (only if layer_type == LAYER_TYPE_POOL)
//     dropout_t config        (only if layer_type == LAYER_TYPE_DROPOUT)
//   -- per layer L in [1, depth), skipped entirely for LAYER_TYPE_POOL and
//      LAYER_TYPE_DROPOUT (neither has weights/bias): --
//     if quantized:
//       float weight_scale[rows]
//       int8_t weight_quantized[rows * row_len]
//       float bias_scale                          (one scale for the whole layer)
//       int8_t bias_quantized[bias_count]
//     else:
//       float weight[rows * row_len]
//       float bias[bias_count]
// (every field above is individually 4-byte-padded; rows/row_len/bias_count
// per layer are exactly quantized_layer_shape()'s output)
nn_error_t nn_save_model_inplace(nn_t *nn, const char *path)
{
  if (!nn)
    return NN_ERROR_INVALID_ARGUMENT;
  FILE *file = fopen(path, "wb");
  if (!file)
    return NN_ERROR_FILE_WRITE;
  uint32_t qflag = nn->quantized ? 1 : 0;
  uint32_t version = ((uint32_t)nn->version_major << 24) | ((uint32_t)nn->version_minor << 16) |
                      ((uint32_t)nn->version_patch << 8) | (uint32_t)nn->version_build;
  uint32_t depth = nn->depth;
  if (!nn_write_padded(file, NN_INPLACE_MAGIC, NN_INPLACE_MAGIC_LEN) ||
      !nn_write_padded(file, &qflag, sizeof(qflag)) ||
      !nn_write_padded(file, &version, sizeof(version)) ||
      !nn_write_padded(file, &depth, sizeof(depth)))
    goto fail;
  for (uint32_t i = 0; i < depth; i++) {
    uint32_t layer_type = nn->layer_type[i];
    uint32_t width = nn->width[i];
    uint32_t activation = nn->activation[i];
    if (!nn_write_padded(file, &layer_type, sizeof(layer_type)) ||
        !nn_write_padded(file, &width, sizeof(width)) ||
        !nn_write_padded(file, &activation, sizeof(activation)))
      goto fail;
    if (layer_type == LAYER_TYPE_CNN) {
      if (!nn_write_padded(file, nn->config[i], sizeof(cnn_t)))
        goto fail;
    } else if (layer_type == LAYER_TYPE_POOL) {
      if (!nn_write_padded(file, nn->config[i], sizeof(pool_t)))
        goto fail;
    } else if (layer_type == LAYER_TYPE_DROPOUT) {
      if (!nn_write_padded(file, nn->config[i], sizeof(dropout_t)))
        goto fail;
    }
  }
  for (uint32_t L = 1; L < depth; L++) {
    if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT)
      continue;
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, (int)L, &rows, &row_len, &bias_count);
    if (nn->quantized) {
      if (!nn_write_padded(file, nn->weight_scale[L], sizeof(float) * (size_t)rows) ||
          !nn_write_padded(file, nn->weight_quantized[L], sizeof(int8_t) * (size_t)rows * row_len) ||
          !nn_write_padded(file, &nn->bias_scale[L], sizeof(float)) ||
          !nn_write_padded(file, nn->bias_quantized[L], sizeof(int8_t) * (size_t)bias_count))
        goto fail;
    } else {
      if (!nn_write_padded(file, nn->weight[L], sizeof(float) * (size_t)rows * row_len) ||
          !nn_write_padded(file, nn->bias[L], sizeof(float) * (size_t)bias_count))
        goto fail;
    }
  }
  fclose(file);
  return NN_ERROR_NONE;
fail:
  fclose(file);
  return NN_ERROR_FILE_WRITE;
}

// Shared by nn_load_model_inplace() (copy == false: zero-copy, aliased,
// immutable) and nn_load_model_inplace_copy() (copy == true: everything
// copied into freshly owned, mutable allocations -- same relationship as
// nn_load_model_binary() vs. nn_load_model_memory(), just for the inplace
// format instead of the regular binary one). See the comments on those two
// public functions for what each mode is for.
static nn_t *nn_load_model_inplace_impl(const uint8_t *data, size_t size, bool copy)
{
  if (!data)
    return NULL;
  nn_reader_t r = {.buf = data, .buf_len = size};
  uint8_t magic[NN_INPLACE_MAGIC_LEN];
  if (!nn_reader_read_padded(&r, magic, NN_INPLACE_MAGIC_LEN) ||
      memcmp(magic, NN_INPLACE_MAGIC, NN_INPLACE_MAGIC_LEN) != 0)
    return NULL;
  uint32_t qflag, version, depth;
  if (!nn_reader_read_padded(&r, &qflag, sizeof(qflag)) ||
      !nn_reader_read_padded(&r, &version, sizeof(version)) ||
      !nn_reader_read_padded(&r, &depth, sizeof(depth)))
    return NULL;

  nn_t *nn = (nn_t *)malloc(sizeof(nn_t));
  if (!nn)
    return NULL;
  nn->quantized = (qflag != 0);
  nn->version_major = (uint8_t)(version >> 24);
  nn->version_minor = (uint8_t)(version >> 16);
  nn->version_patch = (uint8_t)(version >> 8);
  nn->version_build = (uint8_t)version;
  nn->immutable = !copy;
  nn->depth = 0; // Only set to `depth` once every array below is allocated (see comment there)
  nn->layer_type = NULL; nn->width = NULL; nn->activation = NULL; nn->config = NULL;
  nn->neuron = NULL; nn->loss = NULL; nn->preact = NULL;
  nn->weight = NULL; nn->weight_adj = NULL; nn->bias = NULL;
  nn->weight_quantized = NULL; nn->weight_scale = NULL;
  nn->bias_quantized = NULL; nn->bias_scale = NULL;
  nn->pool_argmax = NULL;
  nn->dropout_scale = NULL;
  nn->dropout_mask_valid = NULL;
  nn->rnn_hidden_prev = NULL;
  nn->lstm_cell = NULL;
  nn->lstm_cache = NULL;
  nn->lstm_gate_grad = NULL;
  nn->gru_cache = NULL;
  nn->gru_gate_grad = NULL;
  nn->optimizer = NN_OPTIMIZER_SGD;
  nn->optimizer_momentum = 0.0f;
  nn->optimizer_beta2 = 0.0f;
  nn->optimizer_epsilon = 0.0f;
  nn->adam_step = 0;
  nn->weight_moment1 = NULL;
  nn->weight_moment2 = NULL;
  nn->bias_moment1 = NULL;
  nn->bias_moment2 = NULL;
  nn->activation_buffer = NULL;

  // Allocate every top-level array up front, all sized to `depth` and
  // zeroed, before touching nn->depth: nn_free() indexes every layer <
  // nn->depth into each of these, so if any allocation below fails, the
  // nn_free(nn) call sees a fully consistent "empty" (depth still 0) model
  // instead of a depth that outruns arrays it hasn't allocated yet.
  //
  // weight/weight_adj/bias and weight_quantized/bias_quantized are each
  // allocated on ONE side only, matching nn->quantized -- exactly the
  // invariant nn_free() (and nn_quantize()/nn_dequantize()) already rely on
  // elsewhere in this file: a float model never has weight_quantized/
  // bias_quantized allocated, and a quantized model never has weight/
  // weight_adj/bias allocated. Allocating both unconditionally would leave
  // whichever side nn_free()'s matching branch doesn't look at leaked.
  nn->layer_type = (uint8_t *)calloc(depth, sizeof(*nn->layer_type));
  nn->width = (uint32_t *)calloc(depth, sizeof(*nn->width));
  nn->activation = (uint8_t *)calloc(depth, sizeof(*nn->activation));
  nn->config = (void **)calloc(depth, sizeof(*nn->config));
  nn->neuron = (float **)calloc(depth, sizeof(*nn->neuron));
  nn->loss = (float **)calloc(depth, sizeof(*nn->loss));
  nn->preact = (float **)calloc(depth, sizeof(*nn->preact));
  nn->weight_scale = (float **)calloc(depth, sizeof(*nn->weight_scale));
  nn->bias_scale = (float *)calloc(depth, sizeof(*nn->bias_scale));
  nn->pool_argmax = (int **)calloc(depth, sizeof(*nn->pool_argmax));
  // dropout_scale is training-only scratch (see its field comment in nn.h)
  // that an inplace-loaded (inference-only) model never populates or reads;
  // still allocated (all-NULL) so nn_free() can safely iterate it uniformly.
  nn->dropout_scale = (float **)calloc(depth, sizeof(*nn->dropout_scale));
  // Same reasoning as dropout_scale above -- training-only, unused by an
  // inplace-loaded model -- and calloc's all-zero (false) is exactly the
  // "no mask sampled yet" starting state nn_add_layer() also gives it.
  nn->dropout_mask_valid = (bool *)calloc(depth, sizeof(*nn->dropout_mask_valid));
  // Per RNN layer, holds the previous-hidden-state cache (see its field
  // comment in nn.h) -- unlike dropout_scale, this IS populated by every
  // forward pass (including nn_predict()/nn_error() on an immutable model),
  // so it's allocated per-layer below regardless of `copy`, not just when
  // copy == true.
  nn->rnn_hidden_prev = (float **)calloc(depth, sizeof(*nn->rnn_hidden_prev));
  // Per LSTM layer: cell state (lstm_cell, persistent -- like neuron[] but
  // for c) and its two backward-pass caches (lstm_cache, lstm_gate_grad).
  // All three are populated by every forward pass, same reasoning as
  // rnn_hidden_prev above -- allocated per-layer below regardless of `copy`.
  nn->lstm_cell = (float **)calloc(depth, sizeof(*nn->lstm_cell));
  nn->lstm_cache = (float **)calloc(depth, sizeof(*nn->lstm_cache));
  nn->lstm_gate_grad = (float **)calloc(depth, sizeof(*nn->lstm_gate_grad));
  // Per GRU layer: its one backward-pass cache (gru_cache) and gate-grad
  // buffer (gru_gate_grad) -- no separate persistent-state array, unlike
  // LSTM's lstm_cell, since GRU's one state lives in neuron[] like RNN's.
  nn->gru_cache = (float **)calloc(depth, sizeof(*nn->gru_cache));
  nn->gru_gate_grad = (float **)calloc(depth, sizeof(*nn->gru_gate_grad));
  bool top_level_ok = nn->layer_type && nn->width && nn->activation && nn->config &&
      nn->neuron && nn->loss && nn->preact && nn->weight_scale && nn->bias_scale &&
      nn->pool_argmax && nn->dropout_scale && nn->dropout_mask_valid && nn->rnn_hidden_prev &&
      nn->lstm_cell && nn->lstm_cache && nn->lstm_gate_grad &&
      nn->gru_cache && nn->gru_gate_grad;
  if (nn->quantized) {
    nn->weight_quantized = (int8_t **)calloc(depth, sizeof(*nn->weight_quantized));
    nn->bias_quantized = (int8_t **)calloc(depth, sizeof(*nn->bias_quantized));
    top_level_ok = top_level_ok && nn->weight_quantized && nn->bias_quantized;
  } else {
    nn->weight = (float **)calloc(depth, sizeof(*nn->weight));
    nn->weight_adj = (float **)calloc(depth, sizeof(*nn->weight_adj));
    nn->bias = (float **)calloc(depth, sizeof(*nn->bias));
    // Optimizer moment buffers, same top-level-only allocation as
    // weight_adj: every entry starts NULL (nn->optimizer is NN_OPTIMIZER_SGD
    // by default, set below), and stays that way unless/until a caller
    // later calls nn_set_optimizer() on this (necessarily mutable, copy ==
    // true) model.
    nn->weight_moment1 = (float **)calloc(depth, sizeof(*nn->weight_moment1));
    nn->weight_moment2 = (float **)calloc(depth, sizeof(*nn->weight_moment2));
    nn->bias_moment1 = (float **)calloc(depth, sizeof(*nn->bias_moment1));
    nn->bias_moment2 = (float **)calloc(depth, sizeof(*nn->bias_moment2));
    top_level_ok = top_level_ok && nn->weight && nn->weight_adj && nn->bias &&
      nn->weight_moment1 && nn->weight_moment2 && nn->bias_moment1 && nn->bias_moment2;
  }
  if (!top_level_ok) {
    nn_free(nn);
    return NULL;
  }
  nn->depth = depth;

  // Layer descriptors: layer_type/width/activation/config for every layer,
  // including layer 0 (INPUT), which -- like the rest of this format's
  // header -- is small and simply copied rather than aliased.
  for (uint32_t i = 0; i < depth; i++) {
    uint32_t layer_type, width, activation;
    if (!nn_reader_read_padded(&r, &layer_type, sizeof(layer_type)) ||
        !nn_reader_read_padded(&r, &width, sizeof(width)) ||
        !nn_reader_read_padded(&r, &activation, sizeof(activation)))
      goto fail;
    nn->layer_type[i] = (uint8_t)layer_type;
    nn->width[i] = width;
    nn->activation[i] = (uint8_t)activation;
    if (layer_type == LAYER_TYPE_CNN) {
      cnn_t *c = (cnn_t *)malloc(sizeof(cnn_t));
      if (!c || !nn_reader_read_padded(&r, c, sizeof(cnn_t))) {
        free(c);
        goto fail;
      }
      nn->config[i] = c;
    } else if (layer_type == LAYER_TYPE_POOL) {
      pool_t *p = (pool_t *)malloc(sizeof(pool_t));
      if (!p || !nn_reader_read_padded(&r, p, sizeof(pool_t))) {
        free(p);
        goto fail;
      }
      nn->config[i] = p;
    } else if (layer_type == LAYER_TYPE_DROPOUT) {
      dropout_t *d = (dropout_t *)malloc(sizeof(dropout_t));
      if (!d || !nn_reader_read_padded(&r, d, sizeof(dropout_t))) {
        free(d);
        goto fail;
      }
      nn->config[i] = d;
    }
  }

  // An immutable model only ever runs forward, and a forward pass reads
  // nothing but the layer just below the one it's computing, so every
  // non-recurrent layer's activations can live in one of two buffers
  // shared by the whole network, alternating layer by layer, instead of in
  // two buffers of its own (neuron[] and preact[]). preact[] shares
  // neuron[]'s storage: every layer type writes each element's preact and
  // then its activation from it, element by element, so the activation
  // simply overwrites the value it was computed from. That makes the RAM
  // two full-width buffers of the widest layers instead of two per layer --
  // on a CNN, where the first convolution's output dominates, about a third
  // of what it was. Recurrent layers keep their own neuron[]/preact[]
  // (their neuron[] is persistent state), allocated in the loop below.
  // A mutable (copy == true) model may be trained, and backprop needs
  // every layer's activations at once, so it keeps per-layer buffers too.
  size_t shared_width[2] = {0, 0};
  if (!copy) {
    int side = 0;
    for (uint32_t L = 1; L < depth; L++) {
      if (nn_is_recurrent_layer_type(nn->layer_type[L]))
        continue;
      if (nn->width[L] > shared_width[side])
        shared_width[side] = nn->width[L];
      side = 1 - side;
    }
    if (shared_width[0] + shared_width[1] > 0) {
      nn->activation_buffer = (float *)malloc((shared_width[0] + shared_width[1]) * sizeof(float));
      if (!nn->activation_buffer)
        goto fail;
    }
  }

  // Per-layer activation buffers (owned, small) and weight/bias data (for
  // everything but LAYER_TYPE_POOL/LAYER_TYPE_DROPOUT, which have neither):
  // aliased directly into `data` when copy == false, or allocated fresh and
  // copied out of it when copy == true (see nn_reader_obtain()) -- either
  // way, this loop never calls malloc() for weight/weight_quantized/
  // weight_scale/bias/bias_quantized itself.
  int side = 0;
  for (uint32_t L = 1; L < depth; L++) {
    if (nn_owns_activations(nn, (int)L)) {
      nn->neuron[L] = (float *)malloc((size_t)nn->width[L] * sizeof(float));
      nn->preact[L] = (float *)malloc((size_t)nn->width[L] * sizeof(float));
      if (!nn->neuron[L] || !nn->preact[L])
        goto fail;
    } else {
      // Same alternation as the sizing pass above.
      nn->neuron[L] = nn->activation_buffer + (side ? shared_width[0] : 0);
      nn->preact[L] = nn->neuron[L];
      side = 1 - side;
    }
    if (nn->layer_type[L] == LAYER_TYPE_RNN) {
      // neuron[L] doubles as this layer's hidden state (see LAYER_TYPE_RNN's
      // comment in nn.h) and must start at all-zeros, matching
      // nn_add_layer(); malloc() above left it uninitialized.
      memset(nn->neuron[L], 0, (size_t)nn->width[L] * sizeof(float));
      // Populated by every forward pass (including nn_predict()/nn_error()
      // on this immutable model, not just training) -- see rnn_hidden_prev's
      // field comment in nn.h -- so this is allocated unconditionally, not
      // gated on `copy` like weight_adj/pool_argmax/dropout_scale below.
      nn->rnn_hidden_prev[L] = (float *)malloc((size_t)nn->width[L] * sizeof(float));
      if (!nn->rnn_hidden_prev[L])
        goto fail;
    }
    if (nn->layer_type[L] == LAYER_TYPE_LSTM) {
      const int hidden = (int)nn->width[L];
      // neuron[L] doubles as this layer's hidden state h (same as RNN, and
      // for the same reason) and must start at all-zeros.
      memset(nn->neuron[L], 0, (size_t)hidden * sizeof(float));
      // The cell state c -- also persistent, also must start at all-zeros,
      // also populated by every forward pass (including on this immutable
      // model), so also allocated unconditionally here.
      nn->lstm_cell[L] = (float *)malloc((size_t)hidden * sizeof(float));
      if (!nn->lstm_cell[L])
        goto fail;
      memset(nn->lstm_cell[L], 0, (size_t)hidden * sizeof(float));
      // Cache of this timestep's previous cell/hidden state and every
      // gate's activation -- like lstm_cell, written by every forward pass
      // regardless of training, so also unconditional.
      nn->lstm_cache[L] = (float *)malloc(7 * (size_t)hidden * sizeof(float));
      if (!nn->lstm_cache[L])
        goto fail;
      if (copy) {
        // Unlike lstm_cell/lstm_cache above, this is only ever written by
        // nn_train()'s backward pass (nn_lstm_backward()), never by a
        // forward-only call -- so, like weight_adj below, it only needs to
        // exist for a mutable model.
        nn->lstm_gate_grad[L] = (float *)malloc(4 * (size_t)hidden * sizeof(float));
        if (!nn->lstm_gate_grad[L])
          goto fail;
      }
    }
    if (nn->layer_type[L] == LAYER_TYPE_GRU) {
      const int hidden = (int)nn->width[L];
      // neuron[L] doubles as this layer's hidden state h (same as RNN/LSTM,
      // and for the same reason) and must start at all-zeros. Unlike LSTM,
      // GRU has no second persistent-state array.
      memset(nn->neuron[L], 0, (size_t)hidden * sizeof(float));
      // Cache of this timestep's previous hidden state and every gate's
      // activation -- written by every forward pass regardless of
      // training, so unconditional (same reasoning as lstm_cache above).
      nn->gru_cache[L] = (float *)malloc(5 * (size_t)hidden * sizeof(float));
      if (!nn->gru_cache[L])
        goto fail;
      if (copy) {
        // Only ever written by nn_train()'s backward pass
        // (nn_gru_backward()), never by a forward-only call -- so, like
        // lstm_gate_grad above, only needs to exist for a mutable model.
        nn->gru_gate_grad[L] = (float *)malloc(3 * (size_t)hidden * sizeof(float));
        if (!nn->gru_gate_grad[L])
          goto fail;
      }
    }
    if (copy) {
      // A mutable model needs loss[L] the moment it's ever trained at all
      // (its own contents don't need pre-initializing -- nn_train()'s
      // backward pass always overwrites every element before reading it --
      // but the array itself must exist): the output layer is written
      // directly, and every hidden layer is written by the backprop loop,
      // on every single nn_train() call. nn_add_layer() always allocates
      // this for every layer too.
      nn->loss[L] = (float *)malloc((size_t)nn->width[L] * sizeof(float));
      if (!nn->loss[L])
        goto fail;
    }
    if (nn->layer_type[L] == LAYER_TYPE_POOL) {
      // Only MAX/MIN pooling's backward pass (nn_pool_backward()) reads
      // pool_argmax, and it does so unconditionally (unlike the forward
      // write side, nn_pool_forward(), which is NULL-guarded) -- match
      // nn_add_layer()'s allocation exactly (only for MAX/MIN, never for
      // AVG), or a mutable copy with MAX/MIN pooling would crash the
      // first time it's ever trained.
      if (copy) {
        pool_t *p = nn->config[L];
        if (p->pooling_type == POOLING_TYPE_MAX || p->pooling_type == POOLING_TYPE_MIN) {
          nn->pool_argmax[L] = (int *)malloc((size_t)nn->width[L] * sizeof(int));
          if (!nn->pool_argmax[L])
            goto fail;
        }
      }
      continue;
    }
    if (nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
      // Unlike weight_adj/pool_argmax above, nn_dropout_forward() writes
      // into dropout_scale[L] unconditionally the moment a mutable model
      // is ever trained at all (not lazily on first actual use) -- so this
      // has to be allocated up front too, matching nn_add_layer().
      if (copy) {
        nn->dropout_scale[L] = (float *)malloc((size_t)nn->width[L] * sizeof(float));
        if (!nn->dropout_scale[L])
          goto fail;
      }
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, (int)L, &rows, &row_len, &bias_count);
    void *block;
    if (nn->quantized) {
      if (!nn_reader_obtain(&r, &block, sizeof(float) * (size_t)rows, copy))
        goto fail;
      nn->weight_scale[L] = (float *)block;
      if (!nn_reader_obtain(&r, &block, sizeof(int8_t) * (size_t)rows * row_len, copy))
        goto fail;
      nn->weight_quantized[L] = (int8_t *)block;
      float bias_scale;
      if (!nn_reader_read_padded(&r, &bias_scale, sizeof(bias_scale)))
        goto fail;
      nn->bias_scale[L] = bias_scale;
      if (!nn_reader_obtain(&r, &block, sizeof(int8_t) * (size_t)bias_count, copy))
        goto fail;
      nn->bias_quantized[L] = (int8_t *)block;
    } else {
      if (!nn_reader_obtain(&r, &block, sizeof(float) * (size_t)rows * row_len, copy))
        goto fail;
      nn->weight[L] = (float *)block;
      if (!nn_reader_obtain(&r, &block, sizeof(float) * (size_t)bias_count, copy))
        goto fail;
      nn->bias[L] = (float *)block;
      if (copy) {
        // A mutable model needs a real (if initially zeroed) weight_adj to
        // train, quantize, or dequantize -- nn_add_layer() always allocates
        // it too; the immutable/aliased mode above leaves it NULL forever
        // since nothing is allowed to write to it there.
        nn->weight_adj[L] = (float *)calloc((size_t)rows * row_len, sizeof(float));
        if (!nn->weight_adj[L])
          goto fail;
      }
    }
  }
  return nn;
fail:
  nn_free(nn);
  return NULL;
}

// Loads a neural-net model from an "inplace"-format buffer (magic "NNP1",
// written by nn_save_model_inplace()) with zero-copy weight/bias aliasing:
// nn->weight/nn->bias (float models) or nn->weight_quantized/nn->weight_scale/
// nn->bias_quantized (quantized models) point directly into `data` instead
// of being copied into freshly malloc'd RAM. This is the entry point meant
// for microcontroller targets where the model lives in flash and RAM is
// tight: the dominant cost -- the weight matrices -- never gets duplicated
// into RAM at all. Activations, the RAM that remains, take two buffers
// shared by the whole network (nn->activation_buffer) rather than two per
// layer, since an inference-only forward pass never needs more than the
// layer it's computing and the one below it; see the comment where
// nn_load_model_inplace_impl() sizes them. RNN/LSTM/GRU layers keep their
// own, since their outputs are state carried to the next call.
//
// Requirements on `data`:
//  - It must stay valid and UNCHANGED for as long as the returned nn_t is
//    used -- typically forever, since it is normally a `static const
//    uint8_t[]` baked into flash. This is unlike nn_load_model_memory(),
//    which copies everything and only needs `data` valid for the call.
//  - It should be at least 4-byte aligned (true of any ordinary `const
//    uint8_t[]` in practice). Every field in this format sits at a
//    4-byte-aligned offset from the start of `data` (see the layout comment
//    above nn_save_model_inplace()), so a 4-byte-aligned `data` keeps every
//    aliased float/int32 access aligned too -- required on some
//    microcontroller cores (e.g. Cortex-M0), which fault on unaligned word
//    accesses.
//
// The returned model is read-only: nn_train(), nn_quantize(), nn_dequantize(),
// nn_remove_neuron(), and nn_prune_lightest_neuron() all refuse to run
// against it (nn->immutable is set to true), since each would need to
// write through the aliased pointers above. Use nn_predict()/nn_error() for
// inference. Release it with nn_free() as usual -- nn_free() checks
// immutable to know it must not free those aliased pointers, only
// the small bookkeeping this function allocates itself (layer_type/width/
// activation/config, the top-level pointer arrays, and the activation
// buffers).
nn_t *nn_load_model_inplace(const uint8_t *data, size_t size)
{
  return nn_load_model_inplace_impl(data, size, false);
}

// Loads a neural-net model from an "inplace"-format buffer the same way
// nn_load_model_inplace() does, except every weight/bias array is copied
// into a freshly owned allocation instead of aliased -- a normal, fully
// mutable model (nn->immutable is false), usable with nn_train(),
// nn_quantize(), nn_dequantize(), nn_remove_neuron(), and
// nn_prune_lightest_neuron(), at the cost of the RAM copy
// nn_load_model_inplace() exists to avoid. Intended for tools that need to
// modify an inplace-format model (e.g. quantize/dequantize converting one
// in place) rather than just run inference against it. Unlike
// nn_load_model_inplace(), `data` only needs to stay valid for the
// duration of this call, same as nn_load_model_memory().
nn_t *nn_load_model_inplace_copy(const uint8_t *data, size_t size)
{
  return nn_load_model_inplace_impl(data, size, true);
}

// Saves a neural net model to a file.
nn_error_t nn_save_model_ascii(nn_t *nn, const char *path)
{
  FILE *file = fopen(path, "w");
  if (file == NULL)
    return NN_ERROR_FILE_WRITE;
  // Write quantized flag: 0 = floating, 1 = fixed point
  fprintf(file, "%d\n", nn->quantized ? 1 : 0);
  // Write model version (major, minor, patch, build)
  fprintf(file, "%hhu %hhu %hhu %hhu\n", nn->version_major, nn->version_minor, nn->version_patch, nn->version_build);
  // Write depth
  fprintf(file, "%" PRId32 "\n", nn->depth);
  // Write each layer's type, width, and activation
  for (int i = 0; i < (int)nn->depth; i++) {
    // fprintf(file, "%d %" PRId32 " %d\n", nn->layer_type[i], nn->width[i], nn->activation[i]);
    fprintf(file, "%d %" PRId32 " %d", nn->layer_type[i], nn->width[i], nn->activation[i]);
    if (nn->layer_type[i] == LAYER_TYPE_CNN) {
      cnn_t *c = nn->config[i];
      fprintf(file, " %d %d %d %d %d %d %d %d", c->in_h, c->in_w, c->in_channels, c->out_channels, c->kernel_size, c->stride, c->padding, c->dilation);
    } else if (nn->layer_type[i] == LAYER_TYPE_POOL) {
      pool_t *p = nn->config[i];
      fprintf(file, " %d %d %d %d %d %d", p->in_h, p->in_w, p->channels, p->pool_size, p->stride, (int)p->pooling_type);
    } else if (nn->layer_type[i] == LAYER_TYPE_DROPOUT) {
      dropout_t *d = nn->config[i];
      fprintf(file, " %g", (double)d->rate);
    }
    fputc('\n', file);
  }
  // Write weights & biases
  if (!nn->quantized) {
    // Float mode: write weight_scale (0), weights, and bias
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      // bias_scale placeholder
      fprintf(file, "0\n");
      if (nn->layer_type[layer] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[layer];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        // One scale + k² weights per kernel
        // Bias scale placeholder
        fprintf(file, "0\n");
        for (int k = 0; k < kernels; ++k) {
          // Weight scale
          fprintf(file, "0\n");
          for (int e = 0; e < k_elems; ++e)
            fprintf(file, "%f\n", nn->weight[layer][k * k_elems + e]);
        }
        for (int oc = 0; oc < c->out_channels; ++oc)
          fprintf(file, "%f\n", nn->bias[layer][oc]);
      } else if (nn->layer_type[layer] == LAYER_TYPE_POOL || nn->layer_type[layer] == LAYER_TYPE_DROPOUT) {
        // Pooling and dropout both have no weights/bias beyond the placeholder line already written above
      } else {
        // FC / output / RNN -- see the matching read side in
        // nn_load_model_ascii() for why row_len comes from
        // quantized_layer_shape() here instead of a hardcoded width[layer-1].
        int rows, row_len, bias_count;
        quantized_layer_shape(nn, layer, &rows, &row_len, &bias_count);
        // `rows` (not width[layer]) is the right loop bound -- see the
        // matching read side's comment in nn_load_model_ascii().
        for (int i = 0; i < rows; i++) {
          // weight_scale placeholder
          fprintf(file, "0\n");
          for (int j = 0; j < row_len; j++) {
            fprintf(file, "%f\n", nn->weight[layer][i * row_len + j]);
          }
          fprintf(file, "%f\n", nn->bias[layer][i]);
        }
      }
    }
  } else {
    // Quantized mode: write weight_scale, quantized weights, bias_scale, quantized bias
    for (int layer = 1; layer < (int)nn->depth; layer++) {
      if (nn->layer_type[layer] == LAYER_TYPE_POOL || nn->layer_type[layer] == LAYER_TYPE_DROPOUT) {
        // Pooling and dropout both have no weights/bias to write
        continue;
      }
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, layer, &rows, &row_len, &bias_count);
      for (int row = 0; row < rows; row++) {
        fprintf(file, "%f\n", nn->weight_scale[layer][row]);
        for (int w = 0; w < row_len; w++) {
          fprintf(file, "%d\n", (int)nn->weight_quantized[layer][row * row_len + w]);
        }
      }
      fprintf(file, "%f\n", nn->bias_scale[layer]);
      for (int b = 0; b < bias_count; b++) {
        fprintf(file, "%d\n", (int)nn->bias_quantized[layer][b]);
      }
    }
  }
  fclose(file);
  return NN_ERROR_NONE;
}

// Exports a neural net model as raw binary
nn_error_t nn_save_model_binary(nn_t *nn, const char *path)
{
  FILE *file = fopen(path, "wb");
  if (!file)
    return NN_ERROR_FILE_WRITE;
  // Magic number
  fwrite(NN_BINARY_MAGIC, 1, NN_BINARY_MAGIC_LEN, file);
  // Quantized flag
  uint8_t qflag = nn->quantized ? 1 : 0;
  fwrite(&qflag, sizeof(qflag), 1, file);
  // Model version (major, minor, patch, build)
  fwrite(&nn->version_major, sizeof(nn->version_major), 1, file);
  fwrite(&nn->version_minor, sizeof(nn->version_minor), 1, file);
  fwrite(&nn->version_patch, sizeof(nn->version_patch), 1, file);
  fwrite(&nn->version_build, sizeof(nn->version_build), 1, file);
  // Depth
  uint32_t depth = nn->depth;
  fwrite(&depth, sizeof(depth), 1, file);
  // layer_type, width, and activation per layer
  for (uint32_t i = 0; i < depth; i++) {
    uint8_t layer_type = nn->layer_type[i];
    uint32_t w = nn->width[i];
    uint8_t a = nn->activation[i];
    fwrite(&layer_type, sizeof(layer_type), 1, file);
    fwrite(&w, sizeof(w), 1, file);
    fwrite(&a, sizeof(a), 1, file);
    if (layer_type == LAYER_TYPE_CNN) {
      cnn_t *c = nn->config[i];
      // Struct is POD -> dump
      fwrite(c, sizeof(cnn_t), 1, file);
    } else if (layer_type == LAYER_TYPE_POOL) {
      pool_t *p = nn->config[i];
      fwrite(p, sizeof(pool_t), 1, file);
    } else if (layer_type == LAYER_TYPE_DROPOUT) {
      dropout_t *d = nn->config[i];
      fwrite(d, sizeof(dropout_t), 1, file);
    }
  }
  // Weights & biases
  if (!nn->quantized) {
    // Float mode: placeholders for scales and actual floats
    for (uint32_t L = 1; L < depth; L++) {
      float bias_scale = 0.0f;
      fwrite(&bias_scale, sizeof(bias_scale), 1, file);
      if (nn->layer_type[L] == LAYER_TYPE_CNN) {
        cnn_t *c = nn->config[L];
        int kernels = c->out_channels * c->in_channels;
        int k_elems = c->kernel_size * c->kernel_size;
        for (int k = 0; k < kernels; ++k) {
          float weight_scale = 0.0f;
          fwrite(&weight_scale, sizeof(weight_scale), 1, file);
          fwrite(nn->weight[L] + k * k_elems, sizeof(float), k_elems, file);
        }
        fwrite(nn->bias[L], sizeof(float), c->out_channels, file);
      } else if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
        // Pooling and dropout both have no weights/bias beyond the placeholder written above
      } else {
        // FC / output / RNN -- see nn_load_model_binary_impl()'s matching
        // read side for why row_len (`prev`) comes from
        // quantized_layer_shape() here instead of a hardcoded width[L-1].
        int rows, prev, bias_count;
        quantized_layer_shape(nn, (int)L, &rows, &prev, &bias_count);
        // `rows` (not width[L]) is the right loop bound -- see the ascii
        // saver's identical comment for why (LSTM's rows == 4*width[L]).
        uint32_t curr = (uint32_t)rows;
        for (uint32_t i = 0; i < curr; i++) {
          float weight_scale = 0.0f;
          fwrite(&weight_scale, sizeof(weight_scale), 1, file);
          fwrite(nn->weight[L] + i * prev, sizeof(float), prev, file);
          fwrite(&nn->bias[L][i], sizeof(float), 1, file);
        }
      }
    }
  } else {
    // Quantized mode: real scales and int8 quantized data
    for (uint32_t L = 1; L < depth; L++) {
      if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
        // Pooling and dropout both have no weights/bias to write
        continue;
      }
      int rows, row_len, bias_count;
      quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
      for (int i = 0; i < rows; i++) {
        // Per-row (neuron or kernel) weight scale
        fwrite(&nn->weight_scale[L][i], sizeof(float), 1, file);
        // Quantized weights
        fwrite(nn->weight_quantized[L] + i * row_len, sizeof(int8_t), row_len, file);
      }
      // Bias scale (one per layer)
      fwrite(&nn->bias_scale[L], sizeof(float), 1, file);
      // Quantized biases (one per output channel for CNN, one per neuron for FC/OUTPUT)
      fwrite(nn->bias_quantized[L], sizeof(int8_t), bias_count, file);
    }
  }
  fclose(file);
  return NN_ERROR_NONE;
}

// Peeks a model file's first few bytes to report which format it's in
// (see nn_model_format_t in nn.h), without loading the model.
nn_model_format_t nn_model_format(const char *path)
{
  FILE *file = fopen(path, "rb");
  if (!file)
    return NN_MODEL_FORMAT_UNKNOWN;
  uint8_t magic[4]; // NN_BINARY_MAGIC_LEN == NN_INPLACE_MAGIC_LEN == 4
  size_t n = fread(magic, 1, sizeof(magic), file);
  fclose(file);
  if (n == NN_BINARY_MAGIC_LEN && memcmp(magic, NN_BINARY_MAGIC, NN_BINARY_MAGIC_LEN) == 0)
    return NN_MODEL_FORMAT_BINARY;
  if (n == NN_INPLACE_MAGIC_LEN && memcmp(magic, NN_INPLACE_MAGIC, NN_INPLACE_MAGIC_LEN) == 0)
    return NN_MODEL_FORMAT_INPLACE;
  return NN_MODEL_FORMAT_ASCII;
}

// Loads a model file, auto-detecting ascii vs. binary by peeking for the
// binary format's magic number.
nn_t *nn_load_model(const char *path)
{
  FILE *file = fopen(path, "rb");
  if (!file)
    return NULL;
  uint8_t magic[NN_BINARY_MAGIC_LEN];
  bool is_binary = fread(magic, 1, NN_BINARY_MAGIC_LEN, file) == NN_BINARY_MAGIC_LEN &&
                    memcmp(magic, NN_BINARY_MAGIC, NN_BINARY_MAGIC_LEN) == 0;
  fclose(file);
  return is_binary ? nn_load_model_binary(path) : nn_load_model_ascii(path);
}

// Saves a model file, writing binary format if `path` ends in ".bin"
// (case-insensitive) and ascii format otherwise.
nn_error_t nn_save_model(nn_t *nn, const char *path)
{
  size_t path_len = strlen(path);
  static const char ext[] = ".bin";
  size_t ext_len = sizeof(ext) - 1;
  bool is_binary = path_len >= ext_len;
  for (size_t i = 0; is_binary && i < ext_len; i++)
    is_binary = tolower((unsigned char)path[path_len - ext_len + i]) == ext[i];
  return is_binary ? nn_save_model_binary(nn, path) : nn_save_model_ascii(nn, path);
}

nn_error_t nn_remove_neuron(nn_t *nn, int layer, int neuron_index)
{
  if (nn == NULL || layer <= 0 || layer >= (int)nn->depth || neuron_index < 0 || neuron_index >= (int)nn->width[layer]) {
    return NN_ERROR_INVALID_ARGUMENT;
  }
  if (nn->immutable) {
    // Removing a neuron reallocs/shifts weight (or weight_quantized) and
    // bias in place, which is not possible on buffers aliased into the
    // caller's (read-only) buffer for a model loaded with nn_load_model_inplace().
    return NN_ERROR_READ_ONLY_MODEL;
  }
  // A CNN/POOL layer's width is derived entirely from its cnn_t/pool_t
  // config (spatial dims x channels), not a flat list of independent
  // neurons, and its weight array (if any) isn't neuron-indexed -- there is
  // no well-defined way to "remove one neuron" from it without corrupting
  // the layer's structural computation. Likewise, if the NEXT layer is
  // CNN/POOL, its weight array has no per-input-neuron column to shrink.
  // DROPOUT is neuron-indexed (same width as its input, 1:1), so it doesn't
  // have that structural problem, but it introduces a different one: this
  // function only ever adjusts the ONE layer immediately after `layer`
  // (shrinking its input-column count to match). A DROPOUT layer has no
  // weight matrix of its own to shrink, so removing a neuron *through* it
  // would need to cascade the adjustment one hop further, to whatever comes
  // after the DROPOUT layer -- not implemented, so it's rejected the same
  // way CNN/POOL are, for a different reason.
  // RNN is rejected for yet another reason: removing neuron k from an RNN
  // layer would have to shrink both a ROW (like FC/OUTPUT) AND, because the
  // recurrent half of every row is indexed by this same layer's own
  // neurons, a COLUMN of every remaining row (the one at row_len_in + k) --
  // a self-referential shrink this function has no logic for. And if the
  // NEXT layer is RNN, its row layout is [width[layer] input columns |
  // width[layer+1] recurrent columns] rather than the plain
  // width[layer]-wide row the generic column-removal logic below assumes,
  // so that combination is rejected too. LSTM and GRU are rejected for the
  // same two reasons as RNN, just with four (LSTM) or three (GRU) gate rows
  // per hidden unit instead of one (see quantized_layer_shape()) -- the
  // self-referential shrink problem and the wider-than-width[layer]
  // next-layer row both still apply.
  if (nn->layer_type[layer] == LAYER_TYPE_CNN || nn->layer_type[layer] == LAYER_TYPE_POOL ||
      nn->layer_type[layer] == LAYER_TYPE_DROPOUT || nn->layer_type[layer] == LAYER_TYPE_RNN ||
      nn->layer_type[layer] == LAYER_TYPE_LSTM || nn->layer_type[layer] == LAYER_TYPE_GRU) {
    return NN_ERROR_UNSUPPORTED_LAYER;
  }
  if (layer + 1 < (int)nn->depth &&
      (nn->layer_type[layer + 1] == LAYER_TYPE_CNN || nn->layer_type[layer + 1] == LAYER_TYPE_POOL ||
       nn->layer_type[layer + 1] == LAYER_TYPE_DROPOUT || nn->layer_type[layer + 1] == LAYER_TYPE_RNN ||
       nn->layer_type[layer + 1] == LAYER_TYPE_LSTM || nn->layer_type[layer + 1] == LAYER_TYPE_GRU)) {
    return NN_ERROR_UNSUPPORTED_LAYER;
  }
  int old_width = nn->width[layer];
  // Both `layer` and (if present) `layer + 1` are guaranteed FC/OUTPUT by
  // the guards above, so their weight buffers are always the generic flat,
  // row-major [width[L] x width[L-1]] layout -- no CNN/POOL shape to worry
  // about here.
  int in_row_len = (int)nn->width[layer - 1]; // this layer's row length (unaffected by removing a row)
  // Shift out neuron / preact / loss in this layer
  memmove(&nn->neuron[layer][neuron_index], &nn->neuron[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  memmove(&nn->preact[layer][neuron_index], &nn->preact[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  memmove(&nn->loss[layer][neuron_index], &nn->loss[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
  // Remove row `neuron_index` (in_row_len elements) from this layer's own
  // flat weight buffer, then shrink the buffer to match.
  if (nn->quantized) {
    int8_t *wq = nn->weight_quantized[layer];
    memmove(wq + neuron_index * in_row_len, wq + (neuron_index + 1) * in_row_len,
            sizeof(int8_t) * (size_t)(old_width - neuron_index - 1) * in_row_len);
    nn->weight_quantized[layer] = (int8_t *)realloc(wq, sizeof(int8_t) * (size_t)(old_width - 1) * in_row_len);
    // Shift the single byte biases in bias_quantized[layer]
    memmove(&nn->bias_quantized[layer][neuron_index], &nn->bias_quantized[layer][neuron_index + 1], sizeof(int8_t) * (old_width - neuron_index - 1));
    nn->bias_quantized[layer] = (int8_t *)realloc(nn->bias_quantized[layer], sizeof(int8_t) * (old_width - 1));
    // Leave bias_scale[layer] alone (it's a single float per layer).
    // The float side arrays (weight[layer], weight_adj[layer], bias[layer]) are NULL here, so we must NOT touch them in quantized mode.
  } else {
    float *w = nn->weight[layer];
    memmove(w + neuron_index * in_row_len, w + (neuron_index + 1) * in_row_len,
            sizeof(float) * (size_t)(old_width - neuron_index - 1) * in_row_len);
    nn->weight[layer] = (float *)realloc(w, sizeof(float) * (size_t)(old_width - 1) * in_row_len);
    // weight_adj's contents are always fully overwritten before being read
    // again (see nn_train()'s weight-adjustment computation, which covers
    // every element every call), so it only needs to end up the right
    // *size* -- no need to shift its (disposable) contents first.
    nn->weight_adj[layer] = (float *)realloc(nn->weight_adj[layer], sizeof(float) * (size_t)(old_width - 1) * in_row_len);
    // Shift the float biases in bias[layer]
    memmove(&nn->bias[layer][neuron_index], &nn->bias[layer][neuron_index + 1], sizeof(float) * (old_width - neuron_index - 1));
    nn->bias[layer] = (float *)realloc(nn->bias[layer], sizeof(float) * (old_width - 1));
  }
  // Update next layer's weights to remove the input connection from this
  // neuron: every row of layer+1's weight matrix must have column
  // `neuron_index` removed, shrinking its stride from old_width to
  // old_width-1. Since the buffer is row-major and the new stride is
  // smaller, this is done in place, row by row in increasing order: first
  // splice the column out of the row (in place, at its old offset), then
  // slide the now-shorter row down to its new tighter offset. Processing in
  // increasing order guarantees the slide-down for earlier rows never
  // reaches into a later row's not-yet-read data (row r's old data starts
  // at r*old_width, while all earlier rows' slides finish by r*(old_width-1),
  // which is always <= r*old_width).
  if (layer + 1 < (int)nn->depth) {
    int old_prev_width = old_width;
    int new_prev_width = old_width - 1;
    int next_width = nn->width[layer + 1];
    if (nn->quantized) {
      int8_t *wq_next = nn->weight_quantized[layer + 1];
      for (int r = 0; r < next_width; r++) {
        int8_t *old_row = wq_next + (size_t)r * old_prev_width;
        memmove(old_row + neuron_index, old_row + neuron_index + 1, sizeof(int8_t) * (old_prev_width - neuron_index - 1));
        memmove(wq_next + (size_t)r * new_prev_width, old_row, sizeof(int8_t) * new_prev_width);
      }
      nn->weight_quantized[layer + 1] = (int8_t *)realloc(wq_next, sizeof(int8_t) * (size_t)next_width * new_prev_width);
      // Do NOT touch any biases in layer+1.
    } else {
      float *w_next = nn->weight[layer + 1];
      for (int r = 0; r < next_width; r++) {
        float *old_row = w_next + (size_t)r * old_prev_width;
        memmove(old_row + neuron_index, old_row + neuron_index + 1, sizeof(float) * (old_prev_width - neuron_index - 1));
        memmove(w_next + (size_t)r * new_prev_width, old_row, sizeof(float) * new_prev_width);
      }
      nn->weight[layer + 1] = (float *)realloc(w_next, sizeof(float) * (size_t)next_width * new_prev_width);
      // weight_adj's contents are disposable (see above) -- just resize it.
      nn->weight_adj[layer + 1] = (float *)realloc(nn->weight_adj[layer + 1], sizeof(float) * (size_t)next_width * new_prev_width);
      // Do NOT touch any biases in layer+1.
    }
  }
  // Decrement the width of this layer
  nn->width[layer] = old_width - 1;
  // Reset (not migrate) this layer's and layer+1's optimizer moment
  // buffers to match their new shape: unlike weight/bias themselves, this
  // state has no well-defined per-weight correspondence to preserve across
  // a reshape (see nn_optimizer_alloc_layer()'s comment) -- restarting a
  // few steps of momentum/Adam warmup right after a prune is a negligible,
  // one-time cost next to the alternative of migrating it element-by-
  // element. Only relevant for a mutable float model -- a quantized one has
  // no moment buffers to begin with (nn_quantize() already freed them).
  if (!nn->quantized) {
    if (!nn_optimizer_alloc_layer(nn, layer))
      return NN_ERROR_OUT_OF_MEMORY;
    if (layer + 1 < (int)nn->depth && !nn_optimizer_alloc_layer(nn, layer + 1))
      return NN_ERROR_OUT_OF_MEMORY;
  }
  return NN_ERROR_NONE;
}

// Returns the total weight associated with a given neuron, defined as the sum
// of the absolute values of both: Input weights (weights feeding into the
// neuron from the previous layer) Output weights (weights going out from the
// neuron to the next layer)
float nn_get_total_neuron_weight(nn_t *nn, int layer, int neuron_index)
{
  if (nn == NULL || layer <= 0 || layer >= (int)nn->depth || neuron_index < 0 ||
      neuron_index >= (int)nn->width[layer]) {
    return 0.0f;
  }
  // CNN/POOL layers aren't neuron-indexed the way this function assumes
  // (see nn_remove_neuron() for why), DROPOUT has no weight matrix at all,
  // and RNN's row mixes input and recurrent columns in a layout this
  // function doesn't account for (and is rejected by nn_remove_neuron()
  // regardless) -- there's no meaningful "neuron weight" to report for any
  // of them.
  if (nn->layer_type[layer] == LAYER_TYPE_CNN || nn->layer_type[layer] == LAYER_TYPE_POOL ||
      nn->layer_type[layer] == LAYER_TYPE_DROPOUT || nn->layer_type[layer] == LAYER_TYPE_RNN ||
      nn->layer_type[layer] == LAYER_TYPE_LSTM || nn->layer_type[layer] == LAYER_TYPE_GRU) {
    return 0.0f;
  }
  float total = 0.0f;
  // Sum absolute values of input weights (previous layer to this neuron)
  int in_row_len = (int)nn->width[layer - 1];
  for (int i = 0; i < in_row_len; i++) {
    if (nn->quantized) {
      // For quantized models, we use the quantized weights
      total += fabsf((float)nn->weight_quantized[layer][neuron_index * in_row_len + i] * nn->weight_scale[layer][neuron_index]);
    } else {
      // For float models, we use the float weights directly
      total += fabsf(nn->weight[layer][neuron_index * in_row_len + i]);
    }
  }
  // Sum absolute values of output weights (this neuron to next layer), only
  // when the next layer's weight array is itself neuron-indexed with a
  // plain width[layer]-wide row (FC/OUTPUT) -- an RNN next layer's row is
  // width[layer]+width[layer+1] wide (input columns followed by its own
  // recurrent columns), which next_row_len below does not account for.
  if (layer + 1 < (int)nn->depth &&
      nn->layer_type[layer + 1] != LAYER_TYPE_CNN && nn->layer_type[layer + 1] != LAYER_TYPE_POOL &&
      nn->layer_type[layer + 1] != LAYER_TYPE_DROPOUT && nn->layer_type[layer + 1] != LAYER_TYPE_RNN &&
      nn->layer_type[layer + 1] != LAYER_TYPE_LSTM && nn->layer_type[layer + 1] != LAYER_TYPE_GRU) {
    int next_row_len = (int)nn->width[layer]; // layer+1's row length == this layer's width
    for (int i = 0; i < (int)nn->width[layer + 1]; i++) {
      if (nn->quantized) {
        // For quantized models, we use the quantized weights
        total += fabsf((float)nn->weight_quantized[layer + 1][i * next_row_len + neuron_index] * nn->weight_scale[layer + 1][i]);
      } else {
        // For float models, we use the float weights directly
        total += fabsf(nn->weight[layer + 1][i * next_row_len + neuron_index]);
      }
    }
  }
  return total;
}

bool nn_prune_lightest_neuron(nn_t *nn)
{
  if (nn == NULL || nn->depth < 2) {
    // Invalid or uninitialized network
    return false;
  }
  if (nn->immutable) {
    // nn_remove_neuron() below would refuse anyway, but its return value is
    // ignored here -- check explicitly so this doesn't silently report
    // success on a read-only, flash-resident model.
    return false;
  }
  int lightest_layer = -1;
  int lightest_index = -1;
  float min_weight = FLT_MAX;
  // Search all hidden layers (1..depth-2), skipping
  // CNN/POOL/DROPOUT/RNN/LSTM/GRU layers -- none of them can be pruned this
  // way (see nn_remove_neuron()). This isn't just an optimization:
  // nn_get_total_neuron_weight() returns 0.0f for all six, which would
  // otherwise look like the "lightest" possible neuron and win the search
  // below, silently turning every prune attempt into a no-op
  // (nn_remove_neuron() would then reject it, but its return value here is
  // intentionally ignored the same way it is elsewhere in this function).
  for (int layer = 1; layer < (int)nn->depth - 1; layer++) {
    if (nn->layer_type[layer] == LAYER_TYPE_CNN || nn->layer_type[layer] == LAYER_TYPE_POOL ||
        nn->layer_type[layer] == LAYER_TYPE_DROPOUT || nn->layer_type[layer] == LAYER_TYPE_RNN ||
        nn->layer_type[layer] == LAYER_TYPE_LSTM || nn->layer_type[layer] == LAYER_TYPE_GRU) {
      continue;
    }
    for (int neuron = 0; neuron < (int)nn->width[layer]; neuron++) {
      float tot = nn_get_total_neuron_weight(nn, layer, neuron);
      if (tot < min_weight) {
        min_weight = tot;
        lightest_layer = layer;
        lightest_index = neuron;
      }
    }
  }
  if (lightest_layer < 0) {
    return false;
  }
  nn_remove_neuron(nn, lightest_layer, lightest_index);
  return true;
}

void nn_pool2d(char *src, char *dest, int filter_size, int stride, pooling_type_t pooling_type, int x_in, int y_in)
{
  uint32_t pool_value;
  uint32_t pool_value_temp;

  int x_out = ((x_in - filter_size) / stride) + 1;
  int y_out = ((y_in - filter_size) / stride) + 1;
  // Assume src and dest are RGBA (4 channels)
  for (int z = 0; z < 4; z++) {
    for (int y = 0; y < y_out; y++) {
      for (int x = 0; x < x_out; x++) {
        switch (pooling_type) {
        case POOLING_TYPE_MIN:
          pool_value = 255;
          for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
              pool_value_temp = (uint8_t)*(src + z + ((x + fx) * stride) * 4 + ((y + fy) * stride) * 4 * x_in);
              if (pool_value_temp < pool_value) {
                pool_value = pool_value_temp;
              }
            }
          }
          break;
        case POOLING_TYPE_MAX:
          pool_value = 0;
          for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
              pool_value_temp = (uint8_t)*(src + z + ((x + fx) * stride) * 4 + ((y + fy) * stride) * 4 * x_in);
              if (pool_value_temp > pool_value) {
                pool_value = pool_value_temp;
              }
            }
          }
          break;
        case POOLING_TYPE_AVG:
          pool_value = 0;
          for (int fy = 0; fy < filter_size; fy++) {
            for (int fx = 0; fx < filter_size; fx++) {
              pool_value += (uint8_t)*(src + z + ((x + fx) * stride) * 4 + ((y + fy) * stride) * 4 * x_in);
            }
          }
          pool_value /= (filter_size * filter_size);
          break;
        case POOLING_TYPE_NONE:
        default:
          pool_value = 0;
          break;
        }
        *(dest + z + (x * 4) + (y * x_out * 4)) = (char)pool_value;
      }
    }
  }
}

// 2D convolution
void nn_conv2d(nn_t *nn, int layer)
{
    // Derive channel counts from previous bookkeeping
    cnn_t *cnn = nn->config[layer];
    const int in_c = cnn->in_channels;
    const int x_out = cnn->out_w;
    const int y_out = cnn->out_h;
    const int plane_out = y_out * x_out;
    const int out_c = cnn->out_channels;
    const int row_len = cnn->kernel_size * cnn->kernel_size; // flat weight buffer stride per kernel
    // Sanity‑check shapes
    if ((in_c <= 0) || (out_c <= 0) ||
        ((uint32_t)out_c * plane_out != nn->width[layer])) {
        fprintf(stderr, "conv2d: inconsistent shape (in_c=%d, out_c=%d, plane_out=%d, width[%d]=%u)\n", in_c, out_c, plane_out, layer, nn->width[layer]);
        return;
    }
    // Perform the convolution. The quantized/float branch is hoisted out to
    // once per layer call (instead of once per output-channel/pixel/input-
    // channel), and in the quantized case the per-input-channel weight_scale
    // multiply happens once after accumulating that channel's raw kernel_size^2
    // int8*float products, instead of once per kernel tap.
    //
    // Padding is handled without ever allocating a padded copy of the input:
    // for a given output position, (in_y0, in_x0) is where the kernel window
    // would start in the *unpadded* input if padding were physically
    // prepended, which is negative/out-of-bounds by up to `padding` near the
    // borders. ky_start/ky_end (and kx_start/kx_end) clip the kernel-tap
    // loop to just the sub-range that lands on a real input pixel -- taps
    // outside that range multiply an implicit zero (the padding) and are
    // simply skipped rather than read. With padding == 0 this range is
    // always the full [0, kernel_size), so the loop bodies below are
    // identical to the pre-padding behavior in that case.
    if (nn->quantized) {
        for (int oc = 0; oc < out_c; ++oc) {
            const float bias = (float)nn->bias_quantized[layer][oc] * nn->bias_scale[layer];
            float *dst = nn->neuron[layer] + oc * plane_out;
            float *pre = nn->preact[layer] + oc * plane_out;
            for (int oy = 0; oy < y_out; ++oy) {
                const int in_y0 = oy * cnn->stride - cnn->padding;
                const int ky_start = in_y0 < 0 ? -in_y0 : 0;
                const int ky_end = (in_y0 + cnn->kernel_size > cnn->in_h) ? (cnn->in_h - in_y0) : cnn->kernel_size;
                for (int ox = 0; ox < x_out; ++ox) {
                    const int in_x0 = ox * cnn->stride - cnn->padding;
                    const int kx_start = in_x0 < 0 ? -in_x0 : 0;
                    const int kx_end = (in_x0 + cnn->kernel_size > cnn->in_w) ? (cnn->in_w - in_x0) : cnn->kernel_size;
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_c; ++ic) {
                        const float *base = nn->neuron[layer - 1] + ic * cnn->in_h * cnn->in_w;
                        const int8_t *kptr = nn->weight_quantized[layer] + (oc * in_c + ic) * row_len;
                        const float wsc = nn->weight_scale[layer][oc * in_c + ic];
                        float raw = 0.0f;
                        for (int ky = ky_start; ky < ky_end; ++ky) {
                            const float *sptr = base + (in_y0 + ky) * cnn->in_w + in_x0;
                            const int8_t *wrow = kptr + ky * cnn->kernel_size;
                            for (int kx = kx_start; kx < kx_end; ++kx)
                                raw += sptr[kx] * (float)wrow[kx];
                        }
                        sum += raw * wsc;
                    }
                    const int oidx = oy * x_out + ox;
                    sum += bias;
                    pre[oidx] = sum;
                    dst[oidx] = activation_function[nn->activation[layer]](sum, false);
                }
            }
        }
    } else {
        for (int oc = 0; oc < out_c; ++oc) {
            const float bias = nn->bias[layer][oc];
            float *dst = nn->neuron[layer] + oc * plane_out;
            float *pre = nn->preact[layer] + oc * plane_out;
            for (int oy = 0; oy < y_out; ++oy) {
                const int in_y0 = oy * cnn->stride - cnn->padding;
                const int ky_start = in_y0 < 0 ? -in_y0 : 0;
                const int ky_end = (in_y0 + cnn->kernel_size > cnn->in_h) ? (cnn->in_h - in_y0) : cnn->kernel_size;
                for (int ox = 0; ox < x_out; ++ox) {
                    const int in_x0 = ox * cnn->stride - cnn->padding;
                    const int kx_start = in_x0 < 0 ? -in_x0 : 0;
                    const int kx_end = (in_x0 + cnn->kernel_size > cnn->in_w) ? (cnn->in_w - in_x0) : cnn->kernel_size;
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_c; ++ic) {
                        const float *base = nn->neuron[layer - 1] + ic * cnn->in_h * cnn->in_w;
                        const float *kptr = nn->weight[layer] + (oc * in_c + ic) * row_len;
                        for (int ky = ky_start; ky < ky_end; ++ky) {
                            const float *sptr = base + (in_y0 + ky) * cnn->in_w + in_x0;
                            const float *wrow = kptr + ky * cnn->kernel_size;
                            for (int kx = kx_start; kx < kx_end; ++kx)
                                sum += sptr[kx] * wrow[kx];
                        }
                    }
                    const int oidx = oy * x_out + ox;
                    sum += bias;
                    pre[oidx] = sum;
                    dst[oidx] = activation_function[nn->activation[layer]](sum, false);
                }
            }
        }
    }
}

// In-place quantization of nn_t
nn_error_t nn_quantize(nn_t *nn)
{
  if (!nn || nn->quantized) {
    return NN_ERROR_INVALID_ARGUMENT;
  }
  if (nn->immutable) {
    // Quantizing rewrites weight_scale/bias_scale/weight_quantized/bias_quantized
    // in place, which would require freeing/reallocating buffers aliased
    // into the caller's (read-only) buffer for a model loaded with
    // nn_load_model_inplace().
    return NN_ERROR_READ_ONLY_MODEL;
  }
  const int depth = (int)nn->depth;
  // Free the float-mode weight_scale/bias_scale placeholders (allocated by
  // nn_add_layer) before replacing them with the real quantized-mode arrays
  // below -- otherwise they leak.
  for (int L = 1; L < depth; L++)
    free(nn->weight_scale[L]);
  free(nn->weight_scale);
  free(nn->bias_scale);
  // Mark the network as quantized
  nn->quantized = true;
  // Allocate quantization arrays in the union fields
  nn->weight_quantized = malloc(depth * sizeof(int8_t *));
  nn->weight_scale = malloc(depth * sizeof(float *));
  nn->bias_quantized = malloc(depth * sizeof(int8_t *));
  nn->bias_scale = malloc(depth * sizeof(float));
  if (!nn->weight_quantized || !nn->weight_scale || !nn->bias_quantized || !nn->bias_scale) {
    return NN_ERROR_OUT_OF_MEMORY;
  }
  // Initialize layer 0 entries
  nn->weight_quantized[0] = NULL;
  nn->weight_scale[0] = NULL;
  nn->bias_quantized[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  // Quantize each layer > 1
  for (int L = 1; L < depth; L++) {
    if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
      // Pooling and dropout both have no weights/bias to quantize
      nn->weight_quantized[L] = NULL;
      nn->weight_scale[L] = NULL;
      nn->bias_quantized[L] = NULL;
      nn->bias_scale[L] = 0.0f;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
    // Allocate per-layer arrays (one flat, row-major buffer for the weights)
    nn->weight_quantized[L] = malloc((size_t)rows * row_len * sizeof(int8_t));
    nn->weight_scale[L] = malloc(rows * sizeof(float));
    nn->bias_quantized[L] = malloc(bias_count * sizeof(int8_t));
    if (!nn->weight_quantized[L] || !nn->weight_scale[L] || !nn->bias_quantized[L]) {
      return NN_ERROR_OUT_OF_MEMORY;
    }
    // Compute one bias_scale for this layer
    float min_b = nn->bias[L][0], max_b = min_b;
    for (int i = 1; i < bias_count; i++) {
      float v = nn->bias[L][i];
      if (v < min_b)
        min_b = v;
      if (v > max_b)
        max_b = v;
    }
    float layer_bias_scale = fmaxf(fabsf(min_b), fabsf(max_b)) / 127.0f;
    if (layer_bias_scale == 0.0f)
      layer_bias_scale = 1e-8f;
    nn->bias_scale[L] = layer_bias_scale;
    // For each row (neuron or kernel) in this layer:
    for (int n = 0; n < rows; n++) {
      float *wrow = nn->weight[L] + n * row_len;
      int8_t *qrow = nn->weight_quantized[L] + n * row_len;
      // Compute per-row weight scale
      float min_w = wrow[0], max_w = min_w;
      for (int k = 1; k < row_len; k++) {
        if (wrow[k] < min_w)
          min_w = wrow[k];
        if (wrow[k] > max_w)
          max_w = wrow[k];
      }
      float row_scale = fmaxf(fabsf(min_w), fabsf(max_w)) / 127.0f;
      if (row_scale == 0.0f)
        row_scale = 1e-8f;
      nn->weight_scale[L][n] = row_scale;
      // Quantize each weight
      for (int k = 0; k < row_len; k++) {
        int8_t q = (int8_t)lroundf(wrow[k] / row_scale);
        qrow[k] = (q > 127 ? 127 : (q < -128 ? -128 : q));
      }
    }
    // Quantize the biases (bias_count of them, using the shared layer scale)
    for (int n = 0; n < bias_count; n++) {
      float borig = nn->bias[L][n];
      int8_t bq = (int8_t)lroundf(borig / layer_bias_scale);
      nn->bias_quantized[L][n] = (bq > 127 ? 127 : (bq < -128 ? -128 : bq));
    }
  }
  // Free all of the original float-side storage AFTER quantization
  for (int L = 1; L < depth; L++) {
    // weight/weight_adj are each one flat buffer per layer (or NULL for
    // POOL/DROPOUT, where they were never allocated); free(NULL) is a no-op.
    free(nn->weight[L]);
    free(nn->weight_adj[L]);
    free(nn->bias[L]);
    // Optimizer moment buffers, if any (see nn_t's comment): a quantized
    // model can't train (nn_train() dequantizes first), so there's nothing
    // left to hold this state for. nn_dequantize() reallocates fresh,
    // zeroed buffers if `optimizer` still calls for them.
    free(nn->weight_moment1[L]);
    free(nn->weight_moment2[L]);
    free(nn->bias_moment1[L]);
    free(nn->bias_moment2[L]);
  }
  free(nn->weight);
  free(nn->weight_adj);
  free(nn->bias);
  free(nn->weight_moment1);
  free(nn->weight_moment2);
  free(nn->bias_moment1);
  free(nn->bias_moment2);
  nn->weight = NULL;
  nn->weight_adj = NULL;
  nn->bias = NULL;
  nn->weight_moment1 = NULL;
  nn->weight_moment2 = NULL;
  nn->bias_moment1 = NULL;
  nn->bias_moment2 = NULL;
  return NN_ERROR_NONE;
}

// Dequantize in-place: rebuild float weights/biases from the fixed-point model.
// Returns 0 on success, -1 on error.
nn_error_t nn_dequantize(nn_t *nn)
{
  if (!nn || !nn->quantized) {
    return NN_ERROR_INVALID_ARGUMENT;
  }
  if (nn->immutable) {
    // As in nn_quantize(): dequantizing rewrites weight/weight_adj/bias and
    // frees the quantized-side buffers, which are aliased into the caller's
    // (read-only) buffer for a model loaded with nn_load_model_inplace().
    return NN_ERROR_READ_ONLY_MODEL;
  }
  const int depth = (int)nn->depth;
  // Allocate top-level float pointers
  nn->weight = malloc(depth * sizeof(*nn->weight));
  nn->weight_adj = malloc(depth * sizeof(*nn->weight_adj));
  nn->bias = malloc(depth * sizeof(*nn->bias));
  // Optimizer moment buffers: calloc (not malloc) so every slot starts NULL
  // -- nn_optimizer_alloc_layer() below then (re)allocates only the ones
  // `optimizer` actually calls for, same as a freshly-added layer.
  nn->weight_moment1 = calloc(depth, sizeof(*nn->weight_moment1));
  nn->weight_moment2 = calloc(depth, sizeof(*nn->weight_moment2));
  nn->bias_moment1 = calloc(depth, sizeof(*nn->bias_moment1));
  nn->bias_moment2 = calloc(depth, sizeof(*nn->bias_moment2));
  if (!nn->weight || !nn->weight_adj || !nn->bias ||
      !nn->weight_moment1 || !nn->weight_moment2 || !nn->bias_moment1 || !nn->bias_moment2) {
    return NN_ERROR_OUT_OF_MEMORY;
  }
  nn->weight[0] = NULL;
  nn->weight_adj[0] = NULL;
  nn->bias[0] = NULL;
  // For each layer >=1, rebuild float weight, weight_adj, bias
  for (int L = 1; L < depth; L++) {
    if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
      // Pooling and dropout both have no weights/bias; nothing was
      // quantized for either.
      nn->weight[L] = NULL;
      nn->weight_adj[L] = NULL;
      nn->bias[L] = NULL;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
    // Allocate one flat, row-major float buffer per layer
    nn->weight[L] = malloc((size_t)rows * row_len * sizeof(*nn->weight[L]));
    nn->weight_adj[L] = malloc((size_t)rows * row_len * sizeof(*nn->weight_adj[L]));
    nn->bias[L] = malloc(bias_count * sizeof(*nn->bias[L]));
    if (!nn->weight[L] || !nn->weight_adj[L] || !nn->bias[L]) {
      return NN_ERROR_OUT_OF_MEMORY;
    }
    // Dequantize each weight; weight_adj was never meaningful in quantized
    // mode, so just zero it.
    for (int r = 0; r < rows; r++) {
      float wscale = nn->weight_scale[L][r];
      for (int c = 0; c < row_len; c++) {
        int idx = r * row_len + c;
        nn->weight[L][idx] = nn->weight_quantized[L][idx] * wscale;
        nn->weight_adj[L][idx] = 0.0f;
      }
    }
    // Dequantize the biases (bias_count of them, using the shared layer scale)
    for (int i = 0; i < bias_count; i++) {
      nn->bias[L][i] = nn->bias_quantized[L][i] * nn->bias_scale[L];
    }
    // Give this layer fresh, zeroed optimizer moment buffers if `optimizer`
    // still calls for them (nn_quantize() always freed whatever this layer
    // had before) -- a no-op under NN_OPTIMIZER_SGD.
    if (!nn_optimizer_alloc_layer(nn, L))
      return NN_ERROR_OUT_OF_MEMORY;
    // Free per-layer quant arrays
    free(nn->weight_quantized[L]);
    free(nn->weight_scale[L]);
    free(nn->bias_quantized[L]);
  }
  // Free top-level quant pointers
  free(nn->weight_quantized);
  free(nn->weight_scale);
  free(nn->bias_quantized);
  free(nn->bias_scale);
  nn->weight_quantized = NULL;
  nn->bias_quantized = NULL;
  // Reallocate fresh float-mode weight_scale/bias_scale placeholders,
  // matching what nn_add_layer sets up when a layer is first constructed --
  // without this, nn->weight_scale/bias_scale would be left dangling
  // (freed above, never reassigned), which nn_free() would later try to
  // read/free again.
  nn->weight_scale = malloc(depth * sizeof(*nn->weight_scale));
  nn->bias_scale = malloc(depth * sizeof(*nn->bias_scale));
  if (!nn->weight_scale || !nn->bias_scale) {
    return NN_ERROR_OUT_OF_MEMORY;
  }
  nn->weight_scale[0] = NULL;
  nn->bias_scale[0] = 0.0f;
  for (int L = 1; L < depth; L++) {
    nn->bias_scale[L] = 0.0f;
    if (nn->layer_type[L] == LAYER_TYPE_POOL || nn->layer_type[L] == LAYER_TYPE_DROPOUT) {
      nn->weight_scale[L] = NULL;
      continue;
    }
    int rows, row_len, bias_count;
    quantized_layer_shape(nn, L, &rows, &row_len, &bias_count);
    nn->weight_scale[L] = malloc(rows * sizeof(*nn->weight_scale[L]));
    if (!nn->weight_scale[L]) {
      return NN_ERROR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < rows; i++)
      nn->weight_scale[L][i] = 0.0f;
  }
  // Mark as float mode
  nn->quantized = false;
  return NN_ERROR_NONE;
}
