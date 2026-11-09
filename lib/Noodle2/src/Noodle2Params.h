#pragma once
// Read-only model parameter views and relocation.
#include "Noodle2Buffer.h"

namespace noodle2 {
class NoodleParamSource;

// A layer's parameter block is [INT8 weights][little-endian INT32 biases].
// Offsets in the descriptor are relative to that block; bind(source, base)
// selects where the block begins in any NoodleBuffer. bind(source) uses base=0,
// ideal for a layer relocated into its own RAM/PSRAM allocation.
// PyTorch contiguous layouts: Dense [O,I], Conv2D [O,I,KH,KW],
// DW [I,M,KH,KW] (PyTorch [I*M,1,KH,KW] has identical flat order),
// PW [O,I,1,1], Conv1D [O,I,K]. Activation tensors are CHW.
class NoodleParams {
public:
    NoodleParams(uint32_t weight_offset, uint32_t bias_offset,
                 uint32_t weight_count, uint16_t output_channels,
                 Quant quant = Quant());
    // Returns OUT_OF_BOUNDS if this backing store cannot hold the entire layer.
    // bind(store) expects a compact layer-local block at byte zero.
    // bind(store, offset) selects a block from a packed multi-layer model.
    Status bind(const NoodleBuffer& source, uint32_t layer_base = 0);
    // Streamed metadata has no random-access weights. Graph can validate shape
    // but the only valid execution is a forward-only parameter stream.
    Status bind_stream(NoodleParamSource& source);
    NoodleParamSource* stream_source() const { return stream_source_; }
    uint16_t outputs() const { return output_channels_; }
    uint32_t weights() const { return weight_count_; }
    Quant quant() const { return quant_; }
    // Bytes in this compact [weights][biases] parameter section.
    uint32_t section_bytes() const;
    Status weight(uint32_t index, int8_t& out) const;
    Status bias(uint16_t index, int32_t& out) const;
    Status validate(uint32_t expected_weights, uint16_t expected_outputs) const;
private:
    const NoodleBuffer* source_;
    NoodleParamSource* stream_source_;
    uint32_t layer_base_, weight_offset_, bias_offset_, weight_count_;
    uint16_t output_channels_;
    Quant quant_;
};

// Transfer + byte-verify a compact layer blob, and rebind only on success.
// The source may be a large packed model; target may be a compact RAM/file blob.
Status relocate_layer(NoodleParams& layer, const NoodleBuffer& packed_source,
                      uint32_t source_base, NoodleBuffer& destination,
                      uint32_t destination_base = 0);

} // namespace noodle2
