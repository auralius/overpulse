#pragma once
// NoodleGraph v0.1.17: hand-written topology with two-parent nodes and arena planning.
// One input; topological nodes: Conv2D, Dense, Flatten, Concat, Add.
// All tensor values are INT8 in CHW layout. No heap allocations per layer.
#include "Noodle2Convolution.h"
#include "Noodle2Dense.h"

#ifndef NOODLE2_GRAPH_MAX_NODES
#define NOODLE2_GRAPH_MAX_NODES 16
#endif
#if NOODLE2_GRAPH_MAX_NODES < 2 || NOODLE2_GRAPH_MAX_NODES > 254
#error "NOODLE2_GRAPH_MAX_NODES must be 2..254"
#endif

namespace noodle2 {
class NoodleParamStream;

// One placement decision for the entire lifetime-planned activation arena.
// No changes to operator topology, tensor layout, or allocation per layer.
enum ArenaMemory : uint8_t { ARENA_SRAM = 0, ARENA_PSRAM = 1 };

#ifndef NOODLE2_DEFAULT_ARENA_PSRAM
#define NOODLE2_DEFAULT_ARENA_PSRAM 0
#endif

class NoodleGraph {
public:
    typedef uint8_t Handle;
    static const Handle INVALID = 255;

    NoodleGraph();
    NoodleGraph(const NoodleGraph&) = delete;
    NoodleGraph& operator=(const NoodleGraph&) = delete;

    Handle input(uint16_t channels, uint16_t height, uint16_t width);
    Handle conv2d(Handle from, const NoodleParams& params, Conv2D conv,
                  Activation act = NONE, Pool pool = Pool());
    Handle dense(Handle from, const NoodleParams& params, Activation act = NONE);
    Handle flatten(Handle from);
    // Merge two topologically earlier CHW tensors. Input quantization must
    // already be compatible; Concat performs no requantization.
    Handle concat(Handle left, Handle right);
    // Saturating signed INT8 elementwise sum for tensors using the SAME
    // signed representation. Not a TFLite quantized ADD replacement.
    Handle add(Handle left, Handle right);
    // Select an earlier/intermediate node as the public output if needed.
    Status set_output(Handle h);

    // Optional storage policy: set before compile(). Default is a build option;
    // application topology and all operators remain unchanged.
    Status set_arena_memory(ArenaMemory backend);
    ArenaMemory arena_memory() const { return arena_memory_; }
    // Determine shapes and lifetimes, then allocate and bind one arena.
    Status compile();
    // Copies the CHW INT8 input bytes into the planned input slot.
    Status set_input(const int8_t* values);
    // Executes Noodle2 kernels; tiles Conv2D by default when supported.
    // Pass conv_tile=0 for the direct reference kernel; no arena changes.
    // Optional microsecond clock and per-node timing buffer (MAX_NODES entries).
    // Non-profiled callers use the same run() API and incur no clock calls.
    Status run(uint32_t* node_us = 0, uint32_t (*timer_us)() = 0, uint8_t conv_tile = NOODLE2_DEFAULT_CONV_TILE,
               uint32_t* node_read_us = 0);
    // Execute from a forward-only parameter stream, consuming each channel once.
    Status run_stream(NoodleParamStream& stream,
                      uint32_t* node_us = 0, uint32_t (*timer_us)() = 0,
                      uint32_t* node_read_us = 0, uint8_t conv_tile = NOODLE2_DEFAULT_CONV_TILE);

    const NoodleTensor& output() const { return output_tensor_; }
    Status status() const { return status_; }
    uint32_t arena_bytes() const { return arena_bytes_; }
    uint32_t naive_activation_bytes() const { return naive_bytes_; }
    uint8_t node_count() const { return count_; }
    // Inspect the planned location/length of a node's activation (debugging).
    uint32_t offset(Handle h) const;
    uint32_t bytes(Handle h) const;

private:
    enum Kind : uint8_t { NODE_INPUT, NODE_CONV2D, NODE_DENSE, NODE_FLATTEN, NODE_CONCAT, NODE_ADD };
    struct Node {
        Kind kind;
        Handle parent, second_parent, root, last_use;
        const NoodleParams* params;
        Conv2D conv;
        Pool pool;
        Activation activation;
        uint16_t channels, height, width;
        uint32_t length, offset;
        Node() : kind(NODE_INPUT), parent(INVALID), second_parent(INVALID), root(INVALID), last_use(0),
                 params(0), conv(1, 1), pool(), activation(NONE),
                 channels(0), height(0), width(0), length(0), offset(0) {}
    };
    Handle append(Kind kind, Handle from);
    bool valid(Handle h) const { return h != INVALID && h < count_; }
    Status describe();
    Status plan();
    Status bind_output();
    static bool intersects(uint32_t a, uint32_t an, uint32_t b, uint32_t bn);

    // One owned activation arena; only the public output keeps a persistent view.
    // In/out views during run() live on the stack, not once per graph node.
    NoodleBuffer arena_;
    NoodleBuffer output_slot_;
    NoodleTensor output_tensor_;
    Node nodes_[NOODLE2_GRAPH_MAX_NODES];
    uint8_t count_;
    Handle output_;
    Status status_;
    uint32_t arena_bytes_, naive_bytes_;
    bool compiled_, input_ready_;
    ArenaMemory arena_memory_;
};

}  // namespace noodle2
