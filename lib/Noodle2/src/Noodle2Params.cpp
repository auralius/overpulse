#include "Noodle2.h"
#include <string.h>
#include <stdlib.h>

namespace noodle2 {

NoodleParams::NoodleParams(uint32_t wo, uint32_t bo, uint32_t wc,
                           uint16_t oc, Quant q)
    : source_(0), stream_source_(0), layer_base_(0), weight_offset_(wo), bias_offset_(bo),
      weight_count_(wc), output_channels_(oc), quant_(q) {}
uint32_t NoodleParams::section_bytes() const {
    const uint64_t a = uint64_t(weight_offset_) + weight_count_;
    const uint64_t b = uint64_t(bias_offset_) + uint64_t(output_channels_) * 4U;
    const uint64_t n = a > b ? a : b;
    return n > UINT32_MAX ? 0 : static_cast<uint32_t>(n);
}
Status relocate_layer(NoodleParams& layer, const NoodleBuffer& packed_source,
                      uint32_t source_base, NoodleBuffer& destination,
                      uint32_t destination_base) {
    const uint32_t n = layer.section_bytes();
    if (!n) return OVERFLOW;
    if (&packed_source == &destination) return UNSUPPORTED;
    Status st = copy_region(packed_source,source_base,destination,destination_base,n);
    if (st != OK) return st;
    uint8_t a[256], b[256];
    for (uint32_t i=0; i<n; i+=sizeof(a)) {
        const uint32_t part = (n-i) < sizeof(a) ? (n-i) : sizeof(a);
        if ((st=packed_source.read_block(source_base+i,a,part)) != OK) return st;
        if ((st=destination.read_block(destination_base+i,b,part)) != OK) return st;
        if (memcmp(a,b,part) != 0) return IO_ERROR;
    }
    return layer.bind(destination,destination_base);
}
Status NoodleParams::bind(const NoodleBuffer& source, uint32_t layer_base) {
    // A compact layer block can be relocated without changing its descriptor.
    // Verify the entire span BEFORE accepting a potentially different backend.
    const uint64_t wend = uint64_t(layer_base) + weight_offset_ + weight_count_;
    const uint64_t bend = uint64_t(layer_base) + bias_offset_ +
                          uint64_t(output_channels_) * 4U;
    if (!weight_count_ || !output_channels_) return BAD_PARAMS;
    if (wend > source.capacity() || bend > source.capacity()) return OUT_OF_BOUNDS;
    source_ = &source;
    stream_source_ = 0;
    layer_base_ = layer_base;
    return OK;
}
Status NoodleParams::bind_stream(NoodleParamSource& stream_source) {
    if (!weight_count_ || !output_channels_) return BAD_PARAMS;
    source_ = 0;
    layer_base_ = 0;
    stream_source_ = &stream_source;
    return OK;
}
Status NoodleParams::validate(uint32_t expected_w, uint16_t expected_o) const {
    if ((!source_ && !stream_source_) || !expected_o || weight_count_ != expected_w ||
        output_channels_ != expected_o) return BAD_PARAMS;
    if (quant_.multiplier < 0 || quant_.shift < -30 || quant_.shift > 30)
        return BAD_PARAMS;
    const uint64_t wend = uint64_t(layer_base_) + weight_offset_ + weight_count_;
    const uint64_t bend = uint64_t(layer_base_) + bias_offset_ +
                          uint64_t(output_channels_) * 4U;
    if (source_ && (wend > source_->capacity() || bend > source_->capacity())) return OUT_OF_BOUNDS;
    return OK;
}
Status NoodleParams::weight(uint32_t i, int8_t& out) const {
    if (!source_ || i >= weight_count_) return BAD_PARAMS;
    uint8_t v;
    Status st = source_->read(layer_base_ + weight_offset_ + i, v);
    if (st == OK) out = static_cast<int8_t>(v);
    return st;
}
Status NoodleParams::bias(uint16_t i, int32_t& out) const {
    if (!source_ || i >= output_channels_) return BAD_PARAMS;
    return source_->read_i32(layer_base_ + bias_offset_ + uint32_t(i)*4U, out);
}

} // namespace noodle2
