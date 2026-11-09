#pragma once
// Storage backends and byte-block transfers.
#include "Noodle2Types.h"

namespace noodle2 {

// The only storage owner. Default instances own growable heap RAM.
// bind_ram() binds borrowed, fixed-capacity RAM; bind_flash() is read-only.
// Flash reads use PROGMEM on AVR, normal memory reads on other platforms.
class NoodleBuffer {
public:
    // Callback storage is borrowed; the caller must keep its context alive.
    // Addresses are BYTE offsets in this backend. Return IO_ERROR on failures.
    typedef Status (*ReadBlock)(void* context, uint32_t offset, uint8_t* data, size_t length);
    typedef Status (*WriteBlock)(void* context, uint32_t offset, const uint8_t* data, size_t length);
    NoodleBuffer();
    ~NoodleBuffer();
    NoodleBuffer(const NoodleBuffer&) = delete;
    NoodleBuffer& operator=(const NoodleBuffer&) = delete;

    Status bind_ram(void* bytes, uint32_t capacity);
    Status bind_flash(const void* bytes, uint32_t capacity);
    Status bind_io(void* context, uint32_t capacity, ReadBlock reader,
                   WriteBlock writer = 0);
    Status make_owned();
    // Own a growable PSRAM allocation (ESP32 with external SPIRAM only).
    // Allocation remains exclusive to this buffer and is released in its destructor.
    // Other platforms return UNSUPPORTED; never silently fall back to SRAM.
    Status make_owned_psram();
    Status ensure(uint32_t capacity);
    Status read(uint32_t offset, uint8_t& value) const;
    Status read_block(uint32_t offset, void* dst, uint32_t length) const;
    Status write_block(uint32_t offset, const void* src, uint32_t length);
    Status write(uint32_t offset, uint8_t value);
    Status read_i32(uint32_t offset, int32_t& value) const;
    uint32_t capacity() const { return capacity_; }
    bool writable() const { return kind_ != FLASH && (kind_ != EXTERNAL_IO || write_fn_); }
    bool growable() const { return kind_ == OWNED_RAM || kind_ == OWNED_PSRAM; }
    const uint8_t* address() const { return bytes_; } // not a general backend API

private:
    enum Kind : uint8_t { OWNED_RAM, OWNED_PSRAM, BORROWED_RAM, FLASH, EXTERNAL_IO };
    uint8_t* bytes_;
    uint32_t capacity_;
    Kind kind_;
    void* context_;
    ReadBlock read_fn_;
    WriteBlock write_fn_;
    void release();
};

// Copies a contiguous region, using a small fixed scratch buffer (no heap).
// Bounds checked. Suitable for moving a layer's compact blob across backends.
Status copy_region(const NoodleBuffer& source, uint32_t source_offset,
                   NoodleBuffer& destination, uint32_t destination_offset,
                   uint32_t length);
uint32_t crc32_region(const NoodleBuffer& source, uint32_t offset,
                      uint32_t length, Status& result);

} // namespace noodle2
