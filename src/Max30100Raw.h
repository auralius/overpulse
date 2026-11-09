#pragma once
// Minimal MAX30100 FIFO driver: independent of Noodle2 and any display.
// Raw IR/RED data only; does not compute heart rate or SpO2.
#include <stdint.h>
#include <stddef.h>
#ifndef PPG_LED_CONFIG
#define PPG_LED_CONFIG 0x77
#endif
class TwoWire;

namespace ppg {
struct Sample { uint16_t ir, red; };
class Max30100Raw {
public:
    explicit Max30100Raw(TwoWire& bus);
    bool begin();
    // Remove up to max_samples available samples (max_samples <= 8).
    // n=0 is successful when FIFO empty. overflow is true if FIFO overran.
    bool read(Sample* dst, uint8_t max_samples, uint8_t& n, bool& overflow);
    uint8_t part_id() const { return part_id_; }
    uint32_t errors() const { return errors_; }
    bool clear_fifo();
private:
    bool write_reg(uint8_t reg, uint8_t value);
    bool read_reg(uint8_t reg, uint8_t& value);
    bool read_bytes(uint8_t reg, uint8_t* dst, uint8_t len);
    TwoWire& wire_;
    uint8_t part_id_;
    uint32_t errors_;
};
}  // namespace ppg
