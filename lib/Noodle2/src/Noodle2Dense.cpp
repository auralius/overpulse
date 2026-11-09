#include "Noodle2.h"
#include <string.h>
#include <stdlib.h>
#include "Noodle2KernelInternal.h"

namespace noodle2 {

Status dense(const NoodleTensor& in, NoodleTensor& out,
             const NoodleParams& params, Activation act) {
    if (!supports_fused_activation(act)) return UNSUPPORTED;
    if (&in == &out || &in.buffer() == &out.buffer() ||
        (in.buffer().address() && in.buffer().address() == out.buffer().address()))
        return ALIASED_TENSORS;
    if (!in.size()) return BAD_SHAPE;
    uint32_t expected;
    if (!size4_checked(in.size(), params.outputs(), 1, 1, expected)) return OVERFLOW;
    Status st = params.validate(expected, params.outputs());
    if (st != OK) return st;
    st = out.resize(params.outputs(), 1, 1);
    if (st != OK) return st;
    for (uint16_t o=0; o<params.outputs(); ++o) {
        int32_t acc;
        if ((st = params.bias(o, acc)) != OK) return st;
        for (uint32_t i=0; i<in.size(); ++i) {
            int8_t x, w;
            if ((st = in.get(i, x)) != OK ||
                (st = params.weight(uint32_t(o)*in.size()+i, w)) != OK) return st;
            if (!safe_accumulate(acc, int32_t(x)*int32_t(w))) return OVERFLOW;
        }
        if ((st = out.put(o, apply_activation(requantize(acc, params.quant()), act))) != OK)
            return st;
    }
    return OK;
}

} // namespace noodle2
