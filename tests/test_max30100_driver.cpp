#include "Max30100Raw.h"
#include "Wire.h"
#include <cassert>
#include <cstdio>
int main() {
    TwoWire bus;
    bus.regs[0xFF]=0x11;
    ppg::Max30100Raw sensor(bus);
    assert(sensor.begin());
    assert(sensor.part_id()==0x11);
    assert(bus.regs[0x06]==0x03); // SPO2 mode
    assert(bus.regs[0x07]==0x47); // 100Hz / 1600us / high res
    assert(bus.regs[0x09]==PPG_LED_CONFIG);
    bus.append(0x1234,0xABCD);
    bus.append(0x5678,0xEF01);
    ppg::Sample buf[8]={}; uint8_t n=0;bool ovf=false;
    assert(sensor.read(buf,8,n,ovf) && n==2 && !ovf);
    assert(buf[0].ir==0x1234 && buf[0].red==0xABCD);
    assert(buf[1].ir==0x5678 && buf[1].red==0xEF01);
    assert(sensor.read(buf,8,n,ovf) && n==0 && !ovf);
    bus.regs[0x03]=1;
    assert(sensor.read(buf,8,n,ovf) && n==0 && ovf);
    assert(bus.regs[0x02]==0 && bus.regs[0x03]==0 && bus.regs[0x04]==0);
    std::puts("MAX30100_I2C_DRIVER_MOCK_PASS");
}
