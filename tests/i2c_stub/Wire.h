#pragma once
#include <stdint.h>
#include <deque>
#include <vector>
#include <algorithm>
class TwoWire {
public:
    uint8_t regs[256]={};
    uint8_t address_reg=0;
    std::vector<uint8_t> tx;
    std::deque<uint8_t> rx;
    std::deque<uint8_t> fifo;
    void begin(int,int) {}
    void setClock(uint32_t) {}
    void beginTransmission(uint8_t addr) { (void)addr; tx.clear(); }
    size_t write(uint8_t x) { tx.push_back(x); return 1; }
    uint8_t endTransmission(bool) {
        if(tx.empty()) return 1;
        address_reg=tx[0];
        if(tx.size()>1) for(size_t i=1;i<tx.size();++i) regs[uint8_t(address_reg+i-1)]=tx[i];
        return 0;
    }
    uint8_t requestFrom(uint8_t addr,uint8_t count,uint8_t stop) {
        (void)addr;(void)stop;
        rx.clear();
        if(address_reg==0x05) {
            if(fifo.size()<count) return 0;
            for(uint8_t i=0;i<count;++i) { rx.push_back(fifo.front()); fifo.pop_front(); }
            regs[0x04]=uint8_t((regs[0x04]+count/4)&0x0F);
        } else {
            for(uint8_t i=0;i<count;++i) rx.push_back(regs[uint8_t(address_reg+i)]);
        }
        return count;
    }
    int read() {if(rx.empty()) return -1;int x=rx.front();rx.pop_front();return x;}
    void append(uint16_t ir,uint16_t red) {
        fifo.push_back(uint8_t(ir>>8));fifo.push_back(uint8_t(ir));
        fifo.push_back(uint8_t(red>>8));fifo.push_back(uint8_t(red));
        regs[0x02]=uint8_t((regs[0x02]+1)&0x0F);
    }
};
