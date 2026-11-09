#pragma once
// Noodle 2.0 sequential parameter stream. No seek operation exists in this API.
#include "Noodle2Convolution.h"
#include "Noodle2Dense.h"
#ifndef NOODLE2_STREAM_MAX_CHANNEL_WEIGHTS
#define NOODLE2_STREAM_MAX_CHANNEL_WEIGHTS 256
#endif
namespace noodle2 {
class NoodleParamStream {
public:
    typedef Status (*Reader)(void* context, uint8_t* bytes, uint32_t length);
    typedef uint32_t (*Timer)();
    NoodleParamStream(void* context, Reader reader, bool verify_crc = false);
    Status begin();               // consumes the 16-byte stream header
    Status next(void* dst, uint32_t bytes);  // forward-only; no seek
    Status finish() const;       // check length; CRC32 only when enabled
    // Optional timing of physical reader callbacks only; excludes CRC/compute.
    void set_timer(Timer timer) { timer_ = timer; }
    // Optional caller-owned read-ahead, configured BEFORE begin().
    // Only the physical reader is buffered; next() remains forward-only.
    Status set_prefetch(uint8_t* workspace, uint32_t capacity);
    uint32_t prefetch_bytes() const { return fifo_capacity_; }
    uint32_t logical_calls() const { return logical_calls_; }
    uint32_t read_time_us() const { return read_time_us_; }
    uint32_t crc_time_us() const { return crc_time_us_; }
    bool crc_enabled() const { return verify_crc_; }
    uint32_t read_calls() const { return reads_; }
    uint32_t transferred_bytes() const { return transferred_; }
    uint32_t payload_bytes() const { return consumed_; }
private:
    void* context_;
    Reader reader_;
    uint32_t length_, expected_crc_, crc_, consumed_, reads_, transferred_;
    bool started_;
    bool verify_crc_;
    Timer timer_;
    uint32_t read_time_us_;
    uint32_t crc_time_us_;
    uint8_t* fifo_;
    uint32_t fifo_capacity_, fifo_pos_, fifo_size_, fetched_, logical_calls_;
    Status physical_read(uint8_t* dst, uint32_t n);
};

// Each output channel reads [bias_i32_le][contiguous weights] once.
// Existing, fused, bit-exact kernels are used with a one-channel parameter view.
Status conv2d_stream(const NoodleTensor& in, NoodleTensor& out,
                     const NoodleParams& metadata, Conv2D conv, Activation activation,
                     Pool pool, NoodleParamStream& stream, uint8_t tile = NOODLE2_DEFAULT_CONV_TILE);
Status dense_stream(const NoodleTensor& in, NoodleTensor& out,
                    const NoodleParams& metadata, Activation activation,
                    NoodleParamStream& stream);
} // namespace noodle2
