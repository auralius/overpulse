#pragma once
// CHW INT8 tensor, with grow-only storage.
#include "Noodle2Buffer.h"

namespace noodle2 {

class NoodleTensor {
public:
    NoodleTensor();
    NoodleTensor(const NoodleTensor&) = delete;
    NoodleTensor& operator=(const NoodleTensor&) = delete;

    void bind(NoodleBuffer& buffer);    // storage stays owned by caller
    Status set(const int8_t* values, uint16_t c, uint16_t h, uint16_t w);
    Status resize(uint16_t c, uint16_t h, uint16_t w);
    Status flatten();                  // metadata only; CHW bytes unchanged
    Status get(uint32_t index, int8_t& value) const;
    Status put(uint32_t index, int8_t value);
    // First index of largest INT8 score; returns -1 if empty or unreadable.
    int32_t argmax() const;
    // Status-returning form for applications that need error diagnostics.
    Status argmax(uint32_t& index) const;
    uint32_t size() const { return size_; }
    uint32_t capacity() const { return storage_->capacity(); }
    uint16_t channels() const { return c_; }
    uint16_t height() const { return h_; }
    uint16_t width() const { return w_; }
    const NoodleBuffer& buffer() const { return *storage_; }
    NoodleBuffer& buffer() { return *storage_; }

private:
    NoodleBuffer internal_;
    NoodleBuffer* storage_;
    uint16_t c_, h_, w_;
    uint32_t size_;
};

} // namespace noodle2
