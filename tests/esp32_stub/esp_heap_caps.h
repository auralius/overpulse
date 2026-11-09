#pragma once
#include <cstdlib>
#define MALLOC_CAP_SPIRAM (1u<<10)
#define MALLOC_CAP_8BIT (1u<<2)
inline void* heap_caps_realloc(void* p, size_t bytes, unsigned caps) {
    if ((caps & MALLOC_CAP_SPIRAM)==0 || (caps & MALLOC_CAP_8BIT)==0) return nullptr;
    return std::realloc(p,bytes);
}
