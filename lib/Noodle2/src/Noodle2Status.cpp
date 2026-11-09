#include "Noodle2.h"
#include <string.h>
#include <stdlib.h>

namespace noodle2 {

const char* status_text(Status s) {
    switch (s) {
        case OK: return "OK";
        case BAD_SHAPE: return "BAD_SHAPE";
        case BAD_PARAMS: return "BAD_PARAMS";
        case NO_MEMORY: return "NO_MEMORY";
        case READ_ONLY: return "READ_ONLY";
        case OUT_OF_BOUNDS: return "OUT_OF_BOUNDS";
        case UNSUPPORTED: return "UNSUPPORTED";
        case OVERFLOW: return "OVERFLOW";
        case ALIASED_TENSORS: return "ALIASED_TENSORS";
        case IO_ERROR: return "IO_ERROR";
    }
    return "UNKNOWN";
}

} // namespace noodle2
