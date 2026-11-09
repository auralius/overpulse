#include "Noodle2.h"
#include <string.h>
#include <stdlib.h>
#if defined(ARDUINO_ARCH_ESP32)
#include <esp_heap_caps.h>
#endif
#if defined(ARDUINO_ARCH_AVR) || defined(__AVR__)
#include <avr/pgmspace.h>
#endif
#if defined(ARDUINO_ARCH_ESP8266)
#include <pgmspace.h>
#endif

namespace noodle2 {

NoodleBuffer::NoodleBuffer() : bytes_(0), capacity_(0), kind_(OWNED_RAM), context_(0), read_fn_(0), write_fn_(0) {}
NoodleBuffer::~NoodleBuffer() { release(); }
void NoodleBuffer::release() {
    if ((kind_ == OWNED_RAM || kind_ == OWNED_PSRAM) && bytes_) free(bytes_);
    bytes_ = 0;
    capacity_ = 0;
    context_ = 0; read_fn_ = 0; write_fn_ = 0;
}
Status NoodleBuffer::make_owned() {
    release();
    kind_ = OWNED_RAM;
    return OK;
}
Status NoodleBuffer::make_owned_psram() {
#if defined(ARDUINO_ARCH_ESP32)
    release();
    kind_ = OWNED_PSRAM;
    return OK;
#else
    return UNSUPPORTED;
#endif
}
Status NoodleBuffer::bind_ram(void* ptr, uint32_t capacity) {
    if (capacity && !ptr) return BAD_PARAMS;
    release();
    bytes_ = static_cast<uint8_t*>(ptr);
    capacity_ = capacity;
    kind_ = BORROWED_RAM;
    return OK;
}
Status NoodleBuffer::bind_flash(const void* ptr, uint32_t capacity) {
    if (capacity && !ptr) return BAD_PARAMS;
    release();
    bytes_ = static_cast<uint8_t*>(const_cast<void*>(ptr));
    capacity_ = capacity;
    kind_ = FLASH;
    return OK;
}
Status NoodleBuffer::bind_io(void* context, uint32_t capacity,
                             ReadBlock reader, WriteBlock writer) {
    if (!context || !reader) return BAD_PARAMS;
    release();
    context_ = context;
    capacity_ = capacity;
    kind_ = EXTERNAL_IO;
    read_fn_ = reader;
    write_fn_ = writer;
    return OK;
}
Status NoodleBuffer::ensure(uint32_t capacity) {
    if (capacity <= capacity_) return OK;
    if (kind_ == FLASH) return READ_ONLY;
    if (kind_ == BORROWED_RAM || kind_ == EXTERNAL_IO) return NO_MEMORY;
#if SIZE_MAX < UINT32_MAX
    if (capacity > SIZE_MAX) return OVERFLOW;
#endif
    void* next = 0;
    if (kind_ == OWNED_PSRAM) {
#if defined(ARDUINO_ARCH_ESP32)
        next = heap_caps_realloc(bytes_, static_cast<size_t>(capacity),
                                 MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
        return UNSUPPORTED;
#endif
    } else {
        next = realloc(bytes_, static_cast<size_t>(capacity));
    }
    if (!next) return NO_MEMORY; // old pointer remains valid on failure
    bytes_ = static_cast<uint8_t*>(next);
    capacity_ = capacity;
    return OK;
}
Status NoodleBuffer::read_block(uint32_t offset, void* dst, uint32_t length) const {
    if (offset > capacity_ || length > capacity_ - offset) return OUT_OF_BOUNDS;
    if (length == 0) return OK;
    if (!dst) return BAD_PARAMS;
    if (kind_ == EXTERNAL_IO) {
        return (read_fn_ && context_) ? read_fn_(context_, offset,
            static_cast<uint8_t*>(dst), length) : IO_ERROR;
    }
    if (!bytes_) return BAD_PARAMS;
#if defined(ARDUINO_ARCH_AVR) || defined(__AVR__)
    if (kind_ == FLASH) {
        for (uint32_t i = 0; i < length; ++i)
            static_cast<uint8_t*>(dst)[i] = pgm_read_byte(bytes_ + offset + i);
        return OK;
    }
#endif
#if defined(ARDUINO_ARCH_ESP8266)
    if (kind_ == FLASH) {
        // ESP8266 PROGMEM must be read with flash-safe (unaligned) access.
        memcpy_P(dst, bytes_ + offset, length);
        return OK;
    }
#endif
    memcpy(dst, bytes_ + offset, length);
    return OK;
}
Status NoodleBuffer::write_block(uint32_t offset, const void* src, uint32_t length) {
    if (offset > capacity_ || length > capacity_ - offset) return OUT_OF_BOUNDS;
    if (length == 0) return OK;
    if (!src) return BAD_PARAMS;
    if (kind_ == FLASH) return READ_ONLY;
    if (kind_ == EXTERNAL_IO) {
        return (write_fn_ && context_) ? write_fn_(context_, offset,
            static_cast<const uint8_t*>(src), length) : READ_ONLY;
    }
    if (!bytes_) return BAD_PARAMS;
    memcpy(bytes_ + offset, src, length);
    return OK;
}
Status NoodleBuffer::read(uint32_t offset, uint8_t& v) const {
    return read_block(offset, &v, 1);
}
Status NoodleBuffer::write(uint32_t offset, uint8_t v) {
    return write_block(offset, &v, 1);
}

Status copy_region(const NoodleBuffer& from, uint32_t so,
                   NoodleBuffer& to, uint32_t doff, uint32_t n) {
    if (so > from.capacity() || n > from.capacity() - so ||
        doff > to.capacity() || n > to.capacity() - doff) return OUT_OF_BOUNDS;
    if (!to.writable()) return READ_ONLY;
    uint8_t scratch[256];
    while (n) {
        const uint32_t k = n < sizeof(scratch) ? n : sizeof(scratch);
        Status st = from.read_block(so, scratch, k);
        if (st != OK) return st;
        st = to.write_block(doff, scratch, k);
        if (st != OK) return st;
        n -= k; so += k; doff += k;
    }
    return OK;
}
uint32_t crc32_region(const NoodleBuffer& buf, uint32_t offset,
                      uint32_t length, Status& result) {
    uint8_t scratch[256];
    uint32_t crc = 0xFFFFFFFFu;
    result = OK;
    if (offset > buf.capacity() || length > buf.capacity() - offset) {
        result = OUT_OF_BOUNDS;
        return 0;
    }
    while (length) {
        const uint32_t n = length < sizeof(scratch) ? length : sizeof(scratch);
        result = buf.read_block(offset, scratch, n);
        if (result != OK) return 0;
        for (uint32_t i=0; i<n; ++i) {
            crc ^= scratch[i];
            for (uint8_t j=0; j<8; ++j)
                crc = (crc >> 1) ^ ((crc & 1u) ? 0xEDB88320u : 0u);
        }
        offset += n; length -= n;
    }
    return ~crc;
}

Status NoodleBuffer::read_i32(uint32_t offset, int32_t& v) const {
    if (offset > capacity_ || capacity_ - offset < 4) return OUT_OF_BOUNDS;
    uint8_t a, b, c, d;
    Status st;
    if ((st = read(offset, a)) != OK || (st = read(offset+1, b)) != OK ||
        (st = read(offset+2, c)) != OK || (st = read(offset+3, d)) != OK) return st;
    uint32_t bits = uint32_t(a) | (uint32_t(b) << 8) |
                    (uint32_t(c) << 16) | (uint32_t(d) << 24);
    // Round-trip through memcpy preserves the exact 2's-complement bit pattern.
    memcpy(&v, &bits, 4);
    return OK;
}

} // namespace noodle2
