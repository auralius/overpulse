#include "Noodle2Stream.h"
#include <string.h>
namespace noodle2 {
static uint32_t little32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1])<<8) | (uint32_t(p[2])<<16) | (uint32_t(p[3])<<24);
}
static uint32_t crc_step(uint32_t crc, uint8_t value) {
    crc ^= value;
    for (uint8_t bit=0; bit<8; ++bit)
        crc = (crc>>1) ^ ((crc&1u) ? 0xEDB88320u : 0u);
    return crc;
}
NoodleParamStream::NoodleParamStream(void* ctx, Reader reader, bool verify_crc)
    : context_(ctx), reader_(reader), length_(0), expected_crc_(0),
      crc_(0xFFFFFFFFu), consumed_(0), reads_(0), transferred_(0),
      started_(false), verify_crc_(verify_crc), timer_(0), read_time_us_(0),
      crc_time_us_(0), fifo_(0), fifo_capacity_(0), fifo_pos_(0), fifo_size_(0),
      fetched_(0), logical_calls_(0) {}
Status NoodleParamStream::set_prefetch(uint8_t* workspace, uint32_t capacity) {
    if (started_ || (capacity && !workspace)) return BAD_PARAMS;
    fifo_=capacity ? workspace : 0;
    fifo_capacity_=capacity;
    fifo_pos_=fifo_size_=fetched_=0;
    return OK;
}
Status NoodleParamStream::physical_read(uint8_t* dst, uint32_t n) {
    const uint32_t tick=timer_ ? timer_() : 0;
    const Status st=reader_(context_,dst,n);
    if (timer_) read_time_us_ += timer_()-tick;
    ++reads_;
    if (st == OK) transferred_+=n;
    return st;
}
Status NoodleParamStream::begin() {
    if (!context_ || !reader_ || started_) return BAD_PARAMS;
    uint8_t h[16];
    const Status s=physical_read(h,sizeof(h));
    if (s != OK) return s;
    if (memcmp(h, "N2STRM01", 8) != 0) return BAD_PARAMS;
    length_ = little32(h+8);
    expected_crc_ = little32(h+12);
    if (!length_) return BAD_PARAMS;
    started_ = true;
    return OK;
}
Status NoodleParamStream::next(void* dst, uint32_t n) {
    if (!started_ || !dst || n > length_ - consumed_) return BAD_PARAMS;
    if (!n) return OK;
    ++logical_calls_;
    uint8_t* out=static_cast<uint8_t*>(dst);
    if (!fifo_capacity_) {
        const Status s=physical_read(out,n);
        if (s != OK) return s;
    } else {
        uint32_t done=0;
        while (done<n) {
            if (fifo_pos_==fifo_size_) {
                // Fetch only remaining payload bytes, never beyond EOF/header.
                const uint32_t left=length_-fetched_;
                if (!left) return IO_ERROR;
                const uint32_t request=left<fifo_capacity_ ? left : fifo_capacity_;
                const Status s=physical_read(fifo_,request);
                if (s != OK) return s;
                fetched_+=request;
                fifo_size_=request;
                fifo_pos_=0;
            }
            const uint32_t available=fifo_size_-fifo_pos_;
            const uint32_t take=(n-done)<available ? n-done : available;
            memcpy(out+done,fifo_+fifo_pos_,take);
            fifo_pos_+=take;
            done+=take;
        }
    }
    const uint32_t crc_tick = timer_ ? timer_() : 0;
    if (verify_crc_) {
        const uint8_t* p=static_cast<const uint8_t*>(dst);
        for (uint32_t i=0; i<n; ++i) crc_=crc_step(crc_,p[i]);
    }
    if (timer_) crc_time_us_ += timer_() - crc_tick;
    consumed_ += n;
    return OK;
}
Status NoodleParamStream::finish() const {
    if (!started_ || consumed_ != length_ ||
        (verify_crc_ && (~crc_) != expected_crc_))
        return BAD_PARAMS;
    return OK;
}
// Reorder the one channel's sequential [bias][weights] into the existing
// NoodleParams compact [weights][bias] layout, in-place within 260 bytes.
static Status channel_block(NoodleParamStream& stream, uint8_t* scratch,
                            uint32_t weight_count) {
    if (weight_count > NOODLE2_STREAM_MAX_CHANNEL_WEIGHTS) return UNSUPPORTED;
    const Status st = stream.next(scratch, weight_count+4U);
    if (st != OK) return st;
    uint8_t bias[4];
    memcpy(bias, scratch, 4);
    memmove(scratch, scratch+4, weight_count);
    memcpy(scratch+weight_count, bias, 4);
    return OK;
}
Status conv2d_stream(const NoodleTensor& in, NoodleTensor& out,
                     const NoodleParams& meta, Conv2D conv, Activation act,
                     Pool pool, NoodleParamStream& stream, uint8_t tile) {
    const uint32_t wc=uint32_t(in.channels())*conv.kh*conv.kw;
    if (!wc || wc > NOODLE2_STREAM_MAX_CHANNEL_WEIGHTS) return UNSUPPORTED;
    if (meta.weights() != uint64_t(wc)*meta.outputs()) return BAD_PARAMS;
    uint16_t oh=0,ow=0;
    Status st=conv2d_output_shape(in.height(),in.width(),conv,pool,oh,ow);
    if (st != OK) return st;
    const uint32_t plane=uint32_t(oh)*ow;
    if (uint64_t(plane)*meta.outputs()!=out.size() || !out.buffer().address()) return BAD_SHAPE;
    uint8_t block[NOODLE2_STREAM_MAX_CHANNEL_WEIGHTS+4];
    NoodleBuffer params_buf, out_buf;
    NoodleTensor out_channel;
    NoodleParams params(0,wc,wc,1,meta.quant());
    for (uint16_t oc=0; oc<meta.outputs(); ++oc) {
        if ((st=channel_block(stream, block, wc))!=OK) return st;
        if ((st=params_buf.bind_ram(block,wc+4))!=OK) return st;
        if ((st=params.bind(params_buf))!=OK) return st;
        uint8_t* dst=const_cast<uint8_t*>(out.buffer().address())+uint32_t(oc)*plane;
        if ((st=out_buf.bind_ram(dst,plane))!=OK) return st;
        out_channel.bind(out_buf);
        if ((st=out_channel.resize(1,oh,ow))!=OK) return st;
        if (tile)
            st=noodle2::conv2d_tiled(in,out_channel,params,conv,act,pool,tile,tile);
        else
            st=noodle2::conv2d_direct(in,out_channel,params,conv,act,pool);
        if (st!=OK) return st;
    }
    return OK;
}
Status dense_stream(const NoodleTensor& in, NoodleTensor& out,
                    const NoodleParams& meta, Activation act,
                    NoodleParamStream& stream) {
    const uint32_t wc=in.size();
    if (!wc || wc > NOODLE2_STREAM_MAX_CHANNEL_WEIGHTS) return UNSUPPORTED;
    if (uint64_t(wc)*meta.outputs()!=meta.weights() || out.size()!=meta.outputs()
        || !out.buffer().address()) return BAD_SHAPE;
    uint8_t block[NOODLE2_STREAM_MAX_CHANNEL_WEIGHTS+4];
    NoodleBuffer params_buf, out_buf;
    NoodleTensor out_neuron;
    NoodleParams params(0,wc,wc,1,meta.quant());
    Status st;
    for (uint16_t oc=0; oc<meta.outputs(); ++oc) {
        if ((st=channel_block(stream,block,wc))!=OK) return st;
        if ((st=params_buf.bind_ram(block,wc+4))!=OK) return st;
        if ((st=params.bind(params_buf))!=OK) return st;
        uint8_t* dst=const_cast<uint8_t*>(out.buffer().address())+oc;
        if ((st=out_buf.bind_ram(dst,1))!=OK) return st;
        out_neuron.bind(out_buf);
        if ((st=out_neuron.resize(1,1,1))!=OK) return st;
        if ((st=noodle2::dense(in,out_neuron,params,act))!=OK) return st;
    }
    return OK;
}
} // namespace noodle2
