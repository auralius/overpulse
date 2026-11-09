#pragma once
// Exact port of generate_ppg_dataset.raw_to_int8 from the paired QAT export.
// QAT Input: symmetric signed INT8, q*1/32, IR first then RED in CHW.
// Do not filter input here: this pre-trained model was trained on UNFILTERED raw samples.
#include <stdint.h>
namespace ppg {
static const unsigned PPG_WINDOW=256u;
inline int8_t normalise_one(int32_t diff, int32_t denominator) {
    const int32_t numerator=diff*32;
    const int32_t mag=numerator<0 ? -numerator : numerator;
    const int32_t rounded=(2*mag+denominator)/(2*denominator);
    const int32_t signed_v=numerator<0 ? -rounded : rounded;
    return static_cast<int8_t>(signed_v>127 ? 127 : (signed_v< -128 ? -128 : signed_v));
}
inline void raw_to_chw_i8(const uint16_t* ir, const uint16_t* red, int8_t* chw) {
    const uint16_t* chan[2]={ir,red};
    for(unsigned ch=0;ch<2;++ch) {
        int64_t sum=0;
        for(unsigned i=0;i<PPG_WINDOW;++i) sum+=chan[ch][i];
        const int32_t mean=static_cast<int32_t>(sum/PPG_WINDOW);
        int64_t sumabs=0;
        for(unsigned i=0;i<PPG_WINDOW;++i) {
            const int32_t delta=int32_t(chan[ch][i])-mean;
            sumabs+=delta<0 ? -delta : delta;
        }
        int32_t mad=static_cast<int32_t>(sumabs/PPG_WINDOW);
        if(mad<16) mad=16;
        for(unsigned i=0;i<PPG_WINDOW;++i)
            chw[ch*PPG_WINDOW+i]=normalise_one(int32_t(chan[ch][i])-mean,mad);
    }
}
} // namespace ppg
