#pragma once
// Private helpers shared by Dense and convolution, not part of public API.
#include "Noodle2Types.h"
#include <stdint.h>
#include <limits.h>
namespace noodle2 {
static bool supports_fused_activation(Activation act) {
    return act == NONE || act == RELU || act == LOGITS;
}
static int8_t apply_activation(int8_t v, Activation act) {
    return act == RELU && v < 0 ? 0 : v;
}
static bool size4_checked(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t& out) {
    uint64_t n = uint64_t(a)*b*c*d;
    if (n > UINT32_MAX) return false;
    out = static_cast<uint32_t>(n);
    return true;
}
static bool safe_accumulate(int32_t& acc, int32_t term) {
    if (term > 0 && acc > INT32_MAX-term) return false;
    if (term < 0 && acc < INT32_MIN-term) return false;
    acc += term;
    return true;
}

} // namespace noodle2
