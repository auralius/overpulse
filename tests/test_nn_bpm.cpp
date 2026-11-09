#include "NnBpmEstimator.h"
#include <cassert>
#include <cstdio>
#include <cstdint>

static void make_logits(int8_t* logits, unsigned window_end, unsigned beat_interval) {
    for (unsigned i=0;i<256;++i) {
        const unsigned t = window_end - 256 + i;
        // Pulses at sample 200, 200+period, ...; a small, deterministic shape.
        const unsigned phase = (t >= 200) ? (t - 200) % beat_interval : 999;
        if (phase == 0) logits[i]=90;
        else if(phase == 1 || phase == beat_interval-1) logits[i]=50;
        else logits[i]=-30;
    }
}

static void emit(ppg::NnBpmEstimator& nn, unsigned window_end,
                 unsigned interval, unsigned now=0) {
    int8_t logits[256];make_logits(logits,window_end,interval);
    nn.observe(window_end,logits,now ? now : window_end+90);
}

int main() {
    ppg::NnBpmEstimator nn;
    for(unsigned end=256;end<=1856;end+=64) emit(nn,end,100);
    auto r=nn.estimate(1856+90);
    std::printf("NN_60_BPM valid=%u bpm=%u peaks=%u intervals=%u\n",
                r.valid,r.bpm,r.peaks,r.intervals);
    assert(r.valid && r.bpm==60);
    assert(r.peaks>=5 && r.peaks<=13);

    // An identical frame or overlapping window must never double count a beat.
    auto before=r.peaks;
    for(unsigned i=0;i<3;++i) emit(nn,1792,100,1946);
    assert(nn.estimate(1946).peaks==before);

    // The reference clock is sensor SAMPLE INDEX, not packet reception time.
    // Feed out-of-order results with a common later receive sample.
    ppg::NnBpmEstimator reordered;
    for(unsigned end=1152; end>=256; end-=64) emit(reordered,end,100,1300);
    auto out_of_order=reordered.estimate(1300);
    std::printf("NN_OUT_OF_ORDER valid=%u bpm=%u peaks=%u\n",
                out_of_order.valid,out_of_order.bpm,out_of_order.peaks);
    assert(out_of_order.valid && out_of_order.bpm==60);

    ppg::NnBpmEstimator ninety;
    for(unsigned end=256;end<=1600;end+=64) emit(ninety,end,67);
    auto ninety_r=ninety.estimate(1690);
    std::printf("NN_90_BPM valid=%u bpm=%u peaks=%u\n",
                ninety_r.valid,ninety_r.bpm,ninety_r.peaks);
    assert(ninety_r.valid && ninety_r.bpm>=88 && ninety_r.bpm<=91);

    // Missing ONE 64-sample launch should not hide a beat: subsequent windows
    // overlap broadly and deduplication is based on source timestamps.
    ppg::NnBpmEstimator missing_window;
    for(unsigned end=256;end<=1408;end+=64)
        if(end!=704) emit(missing_window,end,100);
    auto with_gap=missing_window.estimate(1498);
    assert(with_gap.valid && with_gap.bpm==60);

    // Require enough real peaks, no stale beat history and robust no-signal.
    ppg::NnBpmEstimator blank;
    assert(!blank.estimate(2000).valid);
    for(unsigned end=256;end<=448;end+=64) emit(blank,end,100);
    assert(!blank.estimate(538).valid);
    assert(!nn.estimate(2501).valid); // last beat too old
    nn.reset();
    assert(!nn.estimate(1946).valid);

    // A plausible but IRREGULAR series must not be presented as a stable BPM.
    ppg::NnBpmEstimator irregular;
    int8_t logits[256];
    for(unsigned end=256;end<=1500;end+=64) {
        for(unsigned i=0;i<256;++i) {
            const unsigned t=end-256+i;
            const bool beat=(t==200 || t==300 || t==400 || t==510 ||
                             t==590 || t==710 || t==815 || t==900 || t==1015);
            logits[i]=beat ? 90 : -30;
        }
        irregular.observe(end,logits,end+90);
    }
    assert(!irregular.estimate(1590).valid);

    std::puts("NN_BPM_TEST_PASS");
}
