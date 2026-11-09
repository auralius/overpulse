#include "Noodle2.h"
#include <string.h>
#include <stdlib.h>

namespace noodle2 {

NoodleTensor::NoodleTensor()
    : storage_(&internal_), c_(0), h_(0), w_(0), size_(0) {}
void NoodleTensor::bind(NoodleBuffer& b) {
    if (&b != &internal_ && storage_ == &internal_)
        internal_.make_owned(); // release obsolete owned allocation before rebinding
    storage_ = &b;
    c_ = h_ = w_ = 0;
    size_ = 0;
}
Status NoodleTensor::resize(uint16_t c, uint16_t h, uint16_t w) {
    if (!c || !h || !w) return BAD_SHAPE;
    uint64_t total = uint64_t(c) * h * w;
    if (total > UINT32_MAX) return OVERFLOW;
    Status st = storage_->ensure(static_cast<uint32_t>(total));
    if (st != OK) return st;
    c_ = c; h_ = h; w_ = w; size_ = static_cast<uint32_t>(total);
    return OK;
}
Status NoodleTensor::set(const int8_t* values, uint16_t c, uint16_t h, uint16_t w) {
    if (!values) return BAD_PARAMS;
    Status st = resize(c, h, w);
    if (st != OK) return st;
    for (uint32_t i=0; i<size_; ++i) {
        st = put(i, values[i]);
        if (st != OK) return st;
    }
    return OK;
}
Status NoodleTensor::flatten() {
    if (!size_ || size_ > UINT16_MAX) return BAD_SHAPE;
    c_ = static_cast<uint16_t>(size_); h_ = 1; w_ = 1;
    return OK;
}
Status NoodleTensor::get(uint32_t i, int8_t& v) const {
    if (i >= size_) return OUT_OF_BOUNDS;
    uint8_t b;
    Status st = storage_->read(i, b);
    if (st == OK) v = static_cast<int8_t>(b);
    return st;
}
Status NoodleTensor::put(uint32_t i, int8_t v) {
    if (i >= size_) return OUT_OF_BOUNDS;
    return storage_->write(i, static_cast<uint8_t>(v));
}
Status NoodleTensor::argmax(uint32_t& index) const {
    return noodle2::argmax(*this, index);
}
int32_t NoodleTensor::argmax() const {
    uint32_t index = 0;
    if (argmax(index) != OK || index > INT32_MAX) return -1;
    return static_cast<int32_t>(index);
}

} // namespace noodle2
