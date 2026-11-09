#pragma once
// Standalone ReLU, sigmoid, tanh, softmax and argmax.
#include "Noodle2Tensor.h"

namespace noodle2 {

// Standalone in-place activations for e.g. network outputs.
Status relu(NoodleTensor& inout);
// Sigmoid/tanh: a 256-byte INT8 lookup table in Flash or RAM.
// Index 0 means input -128; index 255 means input 127.
class NoodleLUT {
public:
    NoodleLUT(const NoodleBuffer& source, uint32_t offset, uint16_t count)
        : source_(source), offset_(offset), count_(count) {}
    Status get(uint16_t index, uint8_t& value) const;
    uint16_t count() const { return count_; }
private:
    const NoodleBuffer& source_;
    uint32_t offset_;
    uint16_t count_;
};
Status sigmoid(NoodleTensor& inout, const NoodleLUT& lut);
Status tanh(NoodleTensor& inout, const NoodleLUT& lut);
// Softmax: 256 little-endian uint16 exponential LUT values; index is
// max(logit)-logit, LUT[i] = round(65535 * exp(-i * input_scale)).
// Output quantization: scale 1/256, zero-point -128.
Status softmax(NoodleTensor& inout, const NoodleLUT& exp_lut);
Status argmax(const NoodleTensor& in, uint32_t& index);

} // namespace noodle2
