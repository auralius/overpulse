#pragma once
// Dense operator interface.
#include "Noodle2Tensor.h"
#include "Noodle2Params.h"

namespace noodle2 {

Status dense(const NoodleTensor& in, NoodleTensor& out,
             const NoodleParams& params, Activation act = NONE);

} // namespace noodle2
