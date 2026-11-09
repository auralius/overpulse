#pragma once
// Core error codes, activation/pooling flags, and Q31 metadata.
#include <stdint.h>
#include <stddef.h>

namespace noodle2 {

enum Status : uint8_t {
    OK = 0, BAD_SHAPE, BAD_PARAMS, NO_MEMORY, READ_ONLY,
    OUT_OF_BOUNDS, UNSUPPORTED, OVERFLOW, ALIASED_TENSORS, IO_ERROR
};
const char* status_text(Status code);

// LOGITS and NONE both preserve raw (quantized) layer outputs.
// SOFTMAX requires an exponential LUT; use softmax(tensor, lut) explicitly.
// Dense/Conv reject SOFTMAX rather than silently returning logits.
enum Activation : uint8_t { NONE = 0, RELU = 1, SOFTMAX = 2, LOGITS = 3 };
enum PoolType : uint8_t { NO_POOL = 0, MAX_POOL = 1, AVG_POOL = 2 };

struct Pool {
    PoolType type;
    uint8_t kh, kw, sh, sw;
    Pool(PoolType t = NO_POOL, uint8_t kernel_h = 1, uint8_t kernel_w = 1,
         uint8_t stride_h = 1, uint8_t stride_w = 1)
        : type(t), kh(kernel_h), kw(kernel_w), sh(stride_h), sw(stride_w) {}
    static Pool max2() { return Pool(MAX_POOL, 2, 2, 2, 2); }
    static Pool avg2() { return Pool(AVG_POOL, 2, 2, 2, 2); }
};

// Exactly: round_away_from_zero(acc * multiplier * 2^(shift-31)), saturate INT8.
// multiplier is nonnegative Q31; shift range [-30, 30].
// All activations/weights are symmetric INT8 with zero_point=0.
struct Quant {
    int32_t multiplier;
    int8_t shift;
    Quant(int32_t m = 1073741824L, int8_t s = 1) : multiplier(m), shift(s) {}
};

} // namespace noodle2
