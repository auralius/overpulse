#include "Noodle2Graph.h"
#include "Noodle2Stream.h"
#include "Noodle2File.h"
#include <limits.h>
#include <string.h>

namespace noodle2 {

NoodleGraph::NoodleGraph()
    : count_(0), output_(INVALID), status_(OK), arena_bytes_(0),
      naive_bytes_(0), compiled_(false), input_ready_(false),
      arena_memory_(NOODLE2_DEFAULT_ARENA_PSRAM ? ARENA_PSRAM : ARENA_SRAM) {}

NoodleGraph::Handle NoodleGraph::append(Kind kind, Handle from) {
    if (status_ != OK) return INVALID;
    if (compiled_) { status_ = BAD_PARAMS; return INVALID; }
    if (count_ >= NOODLE2_GRAPH_MAX_NODES) {
        status_ = NO_MEMORY; return INVALID;
    }
    if (kind != NODE_INPUT && !valid(from)) { status_ = BAD_PARAMS; return INVALID; }
    const Handle h = count_++;
    Node& n = nodes_[h];
    n.kind = kind;
    n.parent = from;
    n.root = kind == NODE_FLATTEN ? nodes_[from].root : h;
    n.last_use = h;
    output_ = h;
    return h;
}

NoodleGraph::Handle NoodleGraph::input(uint16_t c, uint16_t h, uint16_t w) {
    if (count_ != 0 || !c || !h || !w) {
        status_ = BAD_SHAPE; return INVALID;
    }
    const Handle id = append(NODE_INPUT, INVALID);
    if (id != INVALID) {
        nodes_[id].channels = c;
        nodes_[id].height = h;
        nodes_[id].width = w;
    }
    return id;
}

NoodleGraph::Handle NoodleGraph::conv2d(Handle from, const NoodleParams& p,
                                       Conv2D k, Activation act, Pool pool) {
    const Handle h = append(NODE_CONV2D, from);
    if (h != INVALID) {
        nodes_[h].params = &p;
        nodes_[h].conv = k;
        nodes_[h].activation = act;
        nodes_[h].pool = pool;
    }
    return h;
}

NoodleGraph::Handle NoodleGraph::dense(Handle from, const NoodleParams& p,
                                      Activation act) {
    const Handle h = append(NODE_DENSE, from);
    if (h != INVALID) {
        nodes_[h].params = &p;
        nodes_[h].activation = act;
    }
    return h;
}

NoodleGraph::Handle NoodleGraph::flatten(Handle from) {
    return append(NODE_FLATTEN, from);
}

NoodleGraph::Handle NoodleGraph::concat(Handle left, Handle right) {
    if (!valid(left) || !valid(right)) { status_ = BAD_PARAMS; return INVALID; }
    const Handle h = append(NODE_CONCAT, left);
    if (h != INVALID) nodes_[h].second_parent = right;
    return h;
}

NoodleGraph::Handle NoodleGraph::add(Handle left, Handle right) {
    if (!valid(left) || !valid(right)) { status_ = BAD_PARAMS; return INVALID; }
    const Handle h = append(NODE_ADD, left);
    if (h != INVALID) nodes_[h].second_parent = right;
    return h;
}

Status NoodleGraph::set_output(Handle h) {
    if (!valid(h) || compiled_) return BAD_PARAMS;
    output_ = h;
    return OK;
}

Status NoodleGraph::describe() {
    naive_bytes_ = 0;
    for (Handle i = 0; i < count_; ++i) {
        Node& n = nodes_[i];
        if (n.kind != NODE_INPUT) {
            if (n.activation == SOFTMAX) return UNSUPPORTED;
            const Node& parent = nodes_[n.parent];
            switch (n.kind) {
            case NODE_CONCAT: {
                if (!valid(n.second_parent) || n.second_parent >= i) return BAD_PARAMS;
                const Node& other = nodes_[n.second_parent];
                if (other.height != parent.height || other.width != parent.width)
                    return BAD_SHAPE;
                if (uint32_t(other.channels) + parent.channels > UINT16_MAX)
                    return OVERFLOW;
                n.channels = static_cast<uint16_t>(uint32_t(other.channels) + parent.channels);
                n.height = parent.height;
                n.width = parent.width;
                break;
            }
            case NODE_ADD: {
                if (!valid(n.second_parent) || n.second_parent >= i) return BAD_PARAMS;
                const Node& other = nodes_[n.second_parent];
                if (other.channels != parent.channels || other.height != parent.height ||
                    other.width != parent.width) return BAD_SHAPE;
                n.channels = parent.channels;
                n.height = parent.height;
                n.width = parent.width;
                break;
            }
            case NODE_FLATTEN:
                if (parent.length > UINT16_MAX) return BAD_SHAPE;
                n.channels = static_cast<uint16_t>(parent.length);
                n.height = n.width = 1;
                break;
            case NODE_DENSE: {
                if (!n.params) return BAD_PARAMS;
                n.channels = n.params->outputs();
                n.height = n.width = 1;
                if (uint64_t(parent.length)*n.channels > UINT32_MAX) return OVERFLOW;
                const Status ps = n.params->validate(parent.length*n.channels, n.channels);
                if (ps != OK) return ps;
                break;
            }
            case NODE_CONV2D: {
                if (!n.params) return BAD_PARAMS;
                uint16_t out_h, out_w;
                const Status s = conv2d_output_shape(parent.height, parent.width,
                                                     n.conv, n.pool, out_h, out_w);
                if (s != OK) return s;
                n.channels = n.params->outputs();
                n.height = out_h;
                n.width = out_w;
                const uint64_t weights = uint64_t(parent.channels)*n.channels*
                    n.conv.kh*n.conv.kw;
                if (weights > UINT32_MAX) return OVERFLOW;
                const Status ps = n.params->validate(static_cast<uint32_t>(weights), n.channels);
                if (ps != OK) return ps;
                break;
            }
            default: return UNSUPPORTED;
            }
        }
        const uint64_t length = uint64_t(n.channels)*n.height*n.width;
        if (!length || length > UINT32_MAX) return OVERFLOW;
        n.length = static_cast<uint32_t>(length);
        if (n.kind != NODE_FLATTEN) {
            if (UINT32_MAX - naive_bytes_ < n.length) return OVERFLOW;
            naive_bytes_ += n.length;
        }
    }
    return OK;
}

bool NoodleGraph::intersects(uint32_t a, uint32_t an, uint32_t b, uint32_t bn) {
    return uint64_t(a) < uint64_t(b) + bn && uint64_t(b) < uint64_t(a) + an;
}

Status NoodleGraph::plan() {
    // Tensor and flatten aliases share the same lifetime and storage region.
    for (Handle i = 0; i < count_; ++i)
        nodes_[i].last_use = i;
    for (Handle i = 1; i < count_; ++i) {
        Node& prev = nodes_[nodes_[i].parent];
        Node& owner = nodes_[prev.root];
        if (owner.last_use < i) owner.last_use = i;
        if (nodes_[i].second_parent != INVALID) {
            Node& second = nodes_[nodes_[i].second_parent];
            Node& second_owner = nodes_[second.root];
            if (second_owner.last_use < i) second_owner.last_use = i;
        }
    }
    // Keep public output available after run(), even if not the last node.
    nodes_[nodes_[output_].root].last_use = count_;

    arena_bytes_ = 0;
    for (Handle i = 0; i < count_; ++i) {
        Node& n = nodes_[i];
        if (n.kind == NODE_FLATTEN) continue;
        uint32_t candidate = 0;
        // First-fit search across previously planned *live* regions.
        // The current node's input is live during this operator; never alias it.
        bool moved;
        do {
            moved = false;
            for (Handle j = 0; j < i; ++j) {
                const Node& older = nodes_[j];
                if (older.kind == NODE_FLATTEN || older.last_use < i) continue;
                if (intersects(candidate, n.length, older.offset, older.length)) {
                    const uint64_t next = uint64_t(older.offset) + older.length;
                    if (next > UINT32_MAX) return OVERFLOW;
                    candidate = static_cast<uint32_t>(next);
                    moved = true;
                    break;
                }
            }
        } while (moved);
        n.offset = candidate;
        const uint64_t end = uint64_t(candidate) + n.length;
        if (end > UINT32_MAX) return OVERFLOW;
        if (end > arena_bytes_) arena_bytes_ = static_cast<uint32_t>(end);
    }
    for (Handle i = 0; i < count_; ++i)
        if (nodes_[i].kind == NODE_FLATTEN)
            nodes_[i].offset = nodes_[nodes_[i].root].offset;
    return OK;
}

Status NoodleGraph::set_arena_memory(ArenaMemory backend) {
    if (compiled_ || (backend != ARENA_SRAM && backend != ARENA_PSRAM))
        return BAD_PARAMS;
    arena_memory_ = backend;
    return OK;
}

Status NoodleGraph::bind_output() {
    Status s = (arena_memory_ == ARENA_PSRAM) ? arena_.make_owned_psram()
                                               : arena_.make_owned();
    if (s != OK) return s;
    s = arena_.ensure(arena_bytes_);
    if (s != OK) return s;
    const Node& n = nodes_[output_];
    uint8_t* base = const_cast<uint8_t*>(arena_.address());
    if (!base) return NO_MEMORY;
    Status r = output_slot_.bind_ram(base + n.offset, n.length);
    if (r != OK) return r;
    output_tensor_.bind(output_slot_);
    return output_tensor_.resize(n.channels, n.height, n.width);
}

Status NoodleGraph::compile() {
    if (compiled_) return OK;
    if (status_ != OK) return status_;
    if (!count_ || output_ == INVALID) return (status_ = BAD_SHAPE);
    Status s = describe();
    if (s == OK) s = plan();
    if (s == OK) s = bind_output();
    if (s != OK) return (status_ = s);
    compiled_ = true;
    return OK;
}

Status NoodleGraph::set_input(const int8_t* values) {
    if (!compiled_ || !values) return BAD_PARAMS;
    const Node& n = nodes_[0];
    const Status s = arena_.write_block(n.offset, values, n.length);
    input_ready_ = s == OK;
    return s;
}

Status NoodleGraph::run(uint32_t* node_us, uint32_t (*timer_us)(), uint8_t conv_tile,
                        uint32_t* node_read_us) {
    if (!compiled_ || !input_ready_) return BAD_PARAMS;
    // A metadata-only graph selects its model's stream source automatically.
    // Reject mixed Flash/RAM and stream parameters, or multiple file sources:
    // they cannot satisfy one single-pass parameter schedule.
    NoodleParamSource* source = 0;
    bool saw_memory = false;
    for (Handle i=1; i<count_; ++i) {
        const NoodleParams* p = nodes_[i].params;
        if (!p) continue;
        NoodleParamSource* candidate = p->stream_source();
        if (candidate) {
            if (saw_memory || (source && source != candidate)) return BAD_PARAMS;
            source = candidate;
        } else {
            if (source) return BAD_PARAMS;
            saw_memory = true;
        }
    }
    if (source) {
        Status st = source->open();
        if (st != OK) return st;
        // CRC is diagnostic, not part of the hot inference path.
#if defined(NOODLE2_STREAM_VERIFY_CRC) && NOODLE2_STREAM_VERIFY_CRC
        NoodleParamStream stream(source, NoodleParamSource::read_adapter, true);
#else
        NoodleParamStream stream(source, NoodleParamSource::read_adapter, false);
#endif
        if (timer_us) stream.set_timer(timer_us);
#if defined(NOODLE2_STREAM_PREFETCH_BYTES) && NOODLE2_STREAM_PREFETCH_BYTES > 0
        uint8_t prefetch[NOODLE2_STREAM_PREFETCH_BYTES];
        st = stream.set_prefetch(prefetch, sizeof(prefetch));
#endif
        if (st == OK) st = stream.begin();
        if (st == OK) st = run_stream(stream, node_us, timer_us, node_read_us, conv_tile);
        source->close();
        return st;
    }
    input_ready_ = false;
    // Only two ephemeral tensor views, regardless of graph depth.
    // Each borrows a slice of the compiled arena without allocation.
    NoodleBuffer in_slot, out_slot;
    NoodleTensor in, out;
    uint8_t* base = const_cast<uint8_t*>(arena_.address());
    for (Handle i = 1; i < count_; ++i) {
        const Node& n = nodes_[i];
        if (n.kind == NODE_FLATTEN) continue; // Shape is already captured in Node.
        const Node& p = nodes_[n.parent];
        Status s = in_slot.bind_ram(base + p.offset, p.length);
        if (s == OK) s = out_slot.bind_ram(base + n.offset, n.length);
        if (s != OK) return s;
        in.bind(in_slot);
        out.bind(out_slot);
        s = in.resize(p.channels, p.height, p.width);
        if (s == OK) s = out.resize(n.channels, n.height, n.width);
        if (s != OK) return s;
        const uint32_t tick = (node_us && timer_us) ? timer_us() : 0;
        switch (n.kind) {
        case NODE_CONCAT: {
            const Node& other = nodes_[n.second_parent];
            // Planner keeps both input owners live, with no output overlap.
            memcpy(base + n.offset, base + p.offset, p.length);
            memcpy(base + n.offset + p.length, base + other.offset, other.length);
            s = OK;
            break;
        }
        case NODE_ADD: {
            const Node& other = nodes_[n.second_parent];
            for (uint32_t pos=0; pos<n.length; ++pos) {
                int16_t sum = static_cast<int8_t>(base[p.offset+pos]) +
                              static_cast<int8_t>(base[other.offset+pos]);
                if (sum > 127) sum = 127;
                if (sum < -128) sum = -128;
                base[n.offset+pos] = static_cast<uint8_t>(static_cast<int8_t>(sum));
            }
            s = OK;
            break;
        }
        case NODE_CONV2D:
            s = conv_tile ? noodle2::conv2d_tiled(in, out, *n.params, n.conv, n.activation, n.pool,conv_tile,conv_tile) :
                noodle2::conv2d_direct(in, out, *n.params, n.conv, n.activation, n.pool);
            break;
        case NODE_DENSE:
            s = noodle2::dense(in, out, *n.params, n.activation);
            break;
        default: return UNSUPPORTED;
        }
        if (node_us && timer_us) node_us[i] = timer_us() - tick;
        if (s != OK) return s;
    }
    return OK;
}

Status NoodleGraph::run_stream(NoodleParamStream& stream,
                               uint32_t* node_us, uint32_t (*timer_us)(),
                               uint32_t* node_read_us, uint8_t conv_tile) {
    if (!compiled_ || !input_ready_) return BAD_PARAMS;
    input_ready_ = false;
    NoodleBuffer in_slot, out_slot;
    NoodleTensor in, out;
    uint8_t* base=const_cast<uint8_t*>(arena_.address());
    for (Handle i=1; i<count_; ++i) {
        const Node& n=nodes_[i];
        if (n.kind == NODE_FLATTEN) continue;
        const Node& p=nodes_[n.parent];
        Status st=in_slot.bind_ram(base+p.offset,p.length);
        if (st == OK) st=out_slot.bind_ram(base+n.offset,n.length);
        if (st != OK) return st;
        in.bind(in_slot); out.bind(out_slot);
        st=in.resize(p.channels,p.height,p.width);
        if (st == OK) st=out.resize(n.channels,n.height,n.width);
        if (st != OK) return st;
        const uint32_t tick = (node_us && timer_us) ? timer_us() : 0;
        const uint32_t reads_before = stream.read_time_us();
        if (n.kind == NODE_CONCAT) {
            const Node& other = nodes_[n.second_parent];
            memcpy(base+n.offset,base+p.offset,p.length);
            memcpy(base+n.offset+p.length,base+other.offset,other.length);
            st=OK;
        } else if (n.kind == NODE_ADD) {
            const Node& other=nodes_[n.second_parent];
            for(uint32_t pos=0;pos<n.length;++pos) {
                int16_t sum=static_cast<int8_t>(base[p.offset+pos])+
                            static_cast<int8_t>(base[other.offset+pos]);
                if(sum>127)sum=127;
                if(sum< -128)sum=-128;
                base[n.offset+pos]=static_cast<uint8_t>(static_cast<int8_t>(sum));
            }
            st=OK;
        } else if (n.kind == NODE_CONV2D)
            st=noodle2::conv2d_stream(in,out,*n.params,n.conv,n.activation,n.pool,stream,conv_tile);
        else if (n.kind == NODE_DENSE)
            st=noodle2::dense_stream(in,out,*n.params,n.activation,stream);
        else return UNSUPPORTED;
        if (node_us && timer_us) node_us[i] = timer_us() - tick;
        if (node_read_us) node_read_us[i] = stream.read_time_us() - reads_before;
        if (st != OK) return st;
    }
    return stream.finish();
}

uint32_t NoodleGraph::offset(Handle h) const {
    return valid(h) ? nodes_[h].offset : UINT32_MAX;
}
uint32_t NoodleGraph::bytes(Handle h) const {
    return valid(h) ? nodes_[h].length : 0;
}

} // namespace noodle2
