#include "Max30100Raw.h"
#include <Arduino.h>
#include <Wire.h>
// Register 0x09: upper nibble Red LED, lower nibble IR LED.
// 0x77 = 24 mA / 24 mA (hardware-tested baseline).
// 0x8F = Red 27.1 mA / IR 50 mA (tutorial comparison only).
#ifndef PPG_LED_CONFIG
#define PPG_LED_CONFIG 0x77
#endif
#if (PPG_LED_CONFIG < 0x00) || (PPG_LED_CONFIG > 0xFF)
#error "PPG_LED_CONFIG must be a one-byte MAX30100 LED register value"
#endif

namespace ppg {
namespace {
constexpr uint8_t ADDR = 0x57;
constexpr uint8_t REG_FIFO_WR = 0x02;
constexpr uint8_t REG_FIFO_OVER = 0x03;
constexpr uint8_t REG_FIFO_RD = 0x04;
constexpr uint8_t REG_FIFO_DATA = 0x05;
constexpr uint8_t REG_MODE = 0x06;
constexpr uint8_t REG_SPO2 = 0x07;
constexpr uint8_t REG_LED = 0x09;
constexpr uint8_t REG_PART = 0xFF;
}
Max30100Raw::Max30100Raw(TwoWire& bus) : wire_(bus), part_id_(0), errors_(0) {}
bool Max30100Raw::write_reg(uint8_t reg, uint8_t value) {
    wire_.beginTransmission(ADDR);
    wire_.write(reg);
    wire_.write(value);
    if (wire_.endTransmission(true) != 0) { ++errors_; return false; }
    return true;
}
bool Max30100Raw::read_bytes(uint8_t reg, uint8_t* dst, uint8_t len) {
    wire_.beginTransmission(ADDR);
    wire_.write(reg);
    if (wire_.endTransmission(false) != 0) { ++errors_; return false; }
    if (wire_.requestFrom(static_cast<uint8_t>(ADDR), len, static_cast<uint8_t>(true)) != len) {
        ++errors_; return false;
    }
    for (uint8_t i=0; i<len; ++i) {
        const int b=wire_.read();
        if (b<0) { ++errors_; return false; }
        dst[i]=static_cast<uint8_t>(b);
    }
    return true;
}
bool Max30100Raw::read_reg(uint8_t reg, uint8_t& value) { return read_bytes(reg,&value,1); }
bool Max30100Raw::clear_fifo() {
    return write_reg(REG_FIFO_WR,0) && write_reg(REG_FIFO_OVER,0) && write_reg(REG_FIFO_RD,0);
}
bool Max30100Raw::begin() {
    // A readable device at 0x57 is required. 0x11 is the expected MAX30100 ID.
    if (!read_reg(REG_PART,part_id_) || part_id_ != 0x11) return false;
    if (!write_reg(REG_MODE,0x40)) return false; // reset
    delay(30);
    if (!write_reg(0x01,0x00)) return false; // no interrupt pin needed
    if (!clear_fifo()) return false;
    // 100 samples/s, high-resolution ADC, 1600 us LED pulse.
    if (!write_reg(REG_SPO2,0x47)) return false;
    // Set both LED currents without changing the 100-Hz sensor time base.
    if (!write_reg(REG_LED,static_cast<uint8_t>(PPG_LED_CONFIG))) return false;
    if (!write_reg(REG_MODE,0x03)) return false; // SpO2 mode: both channels
    return true;
}
bool Max30100Raw::read(Sample* dst, uint8_t cap, uint8_t& n, bool& overflow) {
    n=0; overflow=false;
    if (!dst || !cap || cap>8) { ++errors_; return false; }
    uint8_t w=0,r=0,o=0;
    if (!read_reg(REG_FIFO_WR,w) || !read_reg(REG_FIFO_RD,r) || !read_reg(REG_FIFO_OVER,o)) return false;
    if ((o&0x0F)!=0) {
        overflow=true;
        // Discard invalid history and re-establish a synchronized FIFO.
        return clear_fifo();
    }
    const uint8_t available=static_cast<uint8_t>((w-r)&0x0F);
    if (!available) return true;
    n = (available<cap) ? available : cap;
    uint8_t bytes[8*4] = {};
    if (!read_bytes(REG_FIFO_DATA,bytes,static_cast<uint8_t>(n*4))) { n=0; return false; }
    for (uint8_t i=0;i<n;++i) {
        // MAX30100 FIFO format: IR MSB, IR LSB, RED MSB, RED LSB.
        dst[i].ir=uint16_t(uint16_t(bytes[4*i])<<8 | bytes[4*i+1]);
        dst[i].red=uint16_t(uint16_t(bytes[4*i+2])<<8 | bytes[4*i+3]);
    }
    return true;
}
} // namespace ppg
