#pragma once
// Exact fixed-point Q31 requantization.
#include "Noodle2Types.h"

namespace noodle2 {

int8_t requantize(int32_t accumulation, Quant quant);

} // namespace noodle2
