#pragma once
// Conv1D/Conv2D/DW/PW definitions with fused pool and activation.
#include "Noodle2Tensor.h"
#include "Noodle2Params.h"

// Default fast kernel: four-by-four tiles of the final (possibly pooled) output.
// Override to 0 in build configuration for the direct reference path.
// Keeping detailed convolution tracing on the direct path preserves MAC logs.
#ifndef NOODLE2_DEFAULT_CONV_TILE
#ifdef NOODLE2_TRACE_CONV
#define NOODLE2_DEFAULT_CONV_TILE 0
#else
#define NOODLE2_DEFAULT_CONV_TILE 4
#endif
#endif
#if NOODLE2_DEFAULT_CONV_TILE < 0 || NOODLE2_DEFAULT_CONV_TILE > 255
#error "NOODLE2_DEFAULT_CONV_TILE must be in [0,255]"
#endif

namespace noodle2 {

// SAME is input-dependent: output dimensions are ceil(input / stride), with
// extra padding (when odd) on the bottom/right. VALID never pads. EXPLICIT
// retains the exact top/bottom/left/right values supplied by the programmer.
enum PaddingMode : uint8_t { EXPLICIT_PADDING = 0, SAME_PADDING = 1, VALID_PADDING = 2 };

struct Conv2D {
    uint8_t kh, kw, sh, sw, pt, pb, pl, pr;
    PaddingMode padding;

    // Backward-compatible constructor: explicit padding, all zeros by default.
    Conv2D(uint8_t kernel_h, uint8_t kernel_w,
           uint8_t stride_h = 1, uint8_t stride_w = 1,
           uint8_t pad_top = 0, uint8_t pad_bottom = 0,
           uint8_t pad_left = 0, uint8_t pad_right = 0)
        : kh(kernel_h), kw(kernel_w), sh(stride_h), sw(stride_w),
          pt(pad_top), pb(pad_bottom), pl(pad_left), pr(pad_right),
          padding(EXPLICIT_PADDING) {}

    static Conv2D same(uint8_t kh, uint8_t kw, uint8_t sh = 1, uint8_t sw = 1) {
        Conv2D k(kh, kw, sh, sw);
        k.padding = SAME_PADDING;
        return k;
    }
    static Conv2D valid(uint8_t kh, uint8_t kw, uint8_t sh = 1, uint8_t sw = 1) {
        Conv2D k(kh, kw, sh, sw);
        k.padding = VALID_PADDING;
        return k;
    }
    static Conv2D explicit_pad(uint8_t kh, uint8_t kw, uint8_t sh, uint8_t sw,
                               uint8_t pt, uint8_t pb, uint8_t pl, uint8_t pr) {
        return Conv2D(kh, kw, sh, sw, pt, pb, pl, pr);
    }
};

struct Conv1D {
    uint8_t k, stride, pad_left, pad_right;
    PaddingMode padding;
    Conv1D(uint8_t kernel, uint8_t s = 1, uint8_t l = 0, uint8_t r = 0)
        : k(kernel), stride(s), pad_left(l), pad_right(r), padding(EXPLICIT_PADDING) {}
    static Conv1D same(uint8_t k, uint8_t stride = 1) {
        Conv1D conv(k, stride);
        conv.padding = SAME_PADDING;
        return conv;
    }
    static Conv1D valid(uint8_t k, uint8_t stride = 1) {
        Conv1D conv(k, stride);
        conv.padding = VALID_PADDING;
        return conv;
    }
    static Conv1D explicit_pad(uint8_t k, uint8_t stride, uint8_t left, uint8_t right) {
        return Conv1D(k, stride, left, right);
    }
};

// The same geometry calculator used by the convolution kernel.
// A graph planner can infer output shapes without executing the operator.
Status conv2d_output_shape(uint16_t in_h, uint16_t in_w, Conv2D conv,
                           Pool pool, uint16_t& out_h, uint16_t& out_w);

// Optional internal execution strategy. Same output semantics as conv2d().
// Operates on pooled-output tiles and caches one output-channel kernel.
// Falls back to direct convolution if kernel exceeds scratch bound.
Status conv2d_tiled(const NoodleTensor& in, NoodleTensor& out,
                    const NoodleParams& params, Conv2D conv, Activation activation,
                    Pool pool, uint8_t tile_h, uint8_t tile_w);
// Explicit reference implementation, retained for regression and diagnostics.
Status conv2d_direct(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& params,
              Conv2D conv, Activation act = NONE, Pool pool = Pool());
// Production dispatch uses tiled CHW kernel when it fits local scratch, else
// falls back to the reference direct kernel. No topology/API change required.
Status conv2d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& params,
              Conv2D conv, Activation act = NONE, Pool pool = Pool());
Status conv1d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& params,
              Conv1D conv, Activation act = NONE, Pool pool = Pool());
Status dwconv2d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& params,
                Conv2D conv, Activation act = NONE, Pool pool = Pool());
Status pwconv2d(const NoodleTensor& in, NoodleTensor& out, const NoodleParams& params,
                uint8_t sh = 1, uint8_t sw = 1,
                Activation act = NONE, Pool pool = Pool());

} // namespace noodle2
