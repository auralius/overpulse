#pragma once
// Desktop approximation: real ESP8266 toolchain supplies flash-safe memcpy_P.
#include <string.h>
#define PROGMEM
inline void* memcpy_P(void* d,const void* s,size_t n) {return memcpy(d,s,n);}
