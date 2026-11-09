#include "Noodle2.h"
#include <string.h>
#include <stdlib.h>

namespace noodle2 {

Status relu(NoodleTensor& t) {
    for (uint32_t i=0; i<t.size(); ++i) {
        int8_t v; Status st=t.get(i,v);
        if (st != OK) return st;
        if (v<0 && (st=t.put(i,0)) != OK) return st;
    }
    return OK;
}
Status NoodleLUT::get(uint16_t i, uint8_t& v) const {
    if (i>=count_) return OUT_OF_BOUNDS;
    return source_.read(offset_+i,v);
}
static Status apply_lut(NoodleTensor& t, const NoodleLUT& lut) {
    if (lut.count()!=256) return BAD_PARAMS;
    for (uint32_t i=0; i<t.size(); ++i) {
        int8_t x; uint8_t y;
        Status st=t.get(i,x);
        if (st!=OK) return st;
        st=lut.get(uint16_t(int16_t(x)+128),y);
        if (st!=OK) return st;
        st=t.put(i,static_cast<int8_t>(y));
        if (st!=OK) return st;
    }
    return OK;
}
Status sigmoid(NoodleTensor& t,const NoodleLUT& lut) { return apply_lut(t,lut); }
Status tanh(NoodleTensor& t,const NoodleLUT& lut) { return apply_lut(t,lut); }
Status softmax(NoodleTensor& t, const NoodleLUT& exp_lut) {
    if (exp_lut.count()!=512 || !t.size()) return BAD_PARAMS;
    int8_t maximum=-128;
    for (uint32_t i=0; i<t.size(); ++i) {
        int8_t v; Status st=t.get(i,v);
        if (st!=OK) return st;
        if (v>maximum) maximum=v;
    }
    uint64_t sum=0;
    for (uint32_t i=0; i<t.size(); ++i) {
        int8_t v; Status st=t.get(i,v);
        if (st!=OK) return st;
        uint16_t diff=uint16_t(int16_t(maximum)-int16_t(v));
        uint8_t lo,hi;
        if ((st=exp_lut.get(uint16_t(2*diff),lo))!=OK ||
            (st=exp_lut.get(uint16_t(2*diff+1),hi))!=OK) return st;
        sum+=uint16_t(lo)|(uint16_t(hi)<<8);
    }
    if (!sum) return BAD_PARAMS;
    // No extra output tensor. Output uses scale=1/256 and zero point=-128.
    for (uint32_t i=0; i<t.size(); ++i) {
        int8_t v; Status st=t.get(i,v);
        if (st!=OK) return st;
        uint16_t diff=uint16_t(int16_t(maximum)-int16_t(v));
        uint8_t lo,hi;
        if ((st=exp_lut.get(uint16_t(2*diff),lo))!=OK ||
            (st=exp_lut.get(uint16_t(2*diff+1),hi))!=OK) return st;
        uint16_t weight=uint16_t(lo)|(uint16_t(hi)<<8);
        uint64_t q=(uint64_t(weight)*256U+sum/2)/sum;
        if (q>255) q=255;
        st=t.put(i,static_cast<int8_t>(int32_t(q)-128));
        if (st!=OK) return st;
    }
    return OK;
}
Status argmax(const NoodleTensor& t, uint32_t& index) {
    if (!t.size()) return BAD_SHAPE;
    int8_t top=-128;
    index=0;
    for (uint32_t i=0; i<t.size(); ++i) {
        int8_t v;
        Status st=t.get(i,v);
        if (st!=OK) return st;
        if (i==0 || v>top) { index=i; top=v; }
    }
    return OK;
}

} // namespace noodle2
