#include "Noodle2.h"
#include <string.h>
#include <stdlib.h>

namespace noodle2 {

int8_t requantize(int32_t acc, Quant q) {
    if (q.multiplier <= 0) return 0;
    const uint8_t right = static_cast<uint8_t>(31 - q.shift); // 1..61
    int64_t prod = int64_t(acc) * q.multiplier;
    bool neg = prod < 0;
    uint64_t absval = neg ? uint64_t(-(prod + 1)) + 1U : uint64_t(prod);
    uint64_t rounded = (absval + (uint64_t(1) << (right - 1))) >> right;
    if (rounded >= (neg ? 129ULL : 128ULL)) return neg ? -128 : 127;
    int32_t value = neg ? -static_cast<int32_t>(rounded) : static_cast<int32_t>(rounded);
    return static_cast<int8_t>(value);
}

} // namespace noodle2
