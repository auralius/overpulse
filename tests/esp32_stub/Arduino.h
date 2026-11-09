#pragma once
// Desktop-only compilation shim to exercise the Arduino ESP32 source path.
// Does NOT simulate PSRAM timing or ESP32 memory placement.
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdarg>
#define INPUT 0x01
struct FakeSerial {
    void begin(unsigned long) {}
    void flush() {}
    void println(const char* p) { std::puts(p); }
    void println(int x) { std::printf("%d\n",x); }
    void print(const char* p) { std::fputs(p, stdout); }
    void print(int x) { std::printf("%d",x); }
    void print(unsigned long x) { std::printf("%lu",x); }
    template<typename... T> void printf(const char* fmt,T... args) { std::printf(fmt,args...); }
};
static FakeSerial Serial __attribute__((unused));
struct FakeESP {
    uint32_t getPsramSize() const { return 8u*1024u*1024u; }
    uint32_t getFreeHeap() const {return 36000u;}
    uint32_t getFreePsram() const { return 7u*1024u*1024u; }
};
static FakeESP ESP __attribute__((unused));
inline bool psramFound() { return true; }
inline void delay(unsigned long) {}
inline unsigned long micros() {
    using namespace std::chrono;
    return static_cast<unsigned long>(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
}

// Desktop worker LED simulation: keep Arduino sketch signatures compilable.
#ifndef LED_BUILTIN
#define LED_BUILTIN 2
#endif
#ifndef LOW
#define LOW 0
#endif
#ifndef HIGH
#define HIGH 1
#endif
#ifndef OUTPUT
#define OUTPUT 1
inline void pinMode(int, int) {}
inline void digitalWrite(int, int) {}
#endif
