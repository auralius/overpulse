#include "BpmEstimator.h"
#include <cassert>
#include <cstdio>
#include <cmath>

static ppg::BpmEstimator::Result synthetic(unsigned bpm, float amplitude) {
    ppg::BpmEstimator bpm_estimator;
    for (unsigned n=0; n<1500; ++n) {
        const float radians = 6.283185307179586f * bpm * n / (60.f * 100.f);
        const float v = amplitude * sinf(radians) + amplitude*0.25f*sinf(2*radians);
        bpm_estimator.push(static_cast<int16_t>(roundf(v)));
    }
    int16_t ordered[ppg::BpmEstimator::N];
    assert(bpm_estimator.snapshot(ordered));
    return ppg::BpmEstimator::estimate(ordered,ppg::BpmEstimator::N);
}

int main() {
    auto sixty=synthetic(60,40.0f);
    std::printf("60 bpm: valid=%d rate=%u corr=%.3f lag=%u\n",
                sixty.valid,sixty.bpm,sixty.correlation,sixty.lag);
    assert(sixty.valid && sixty.bpm>=58 && sixty.bpm<=62);
    auto ninety=synthetic(90,40.0f);
    std::printf("90 bpm: valid=%d rate=%u corr=%.3f lag=%u\n",
                ninety.valid,ninety.bpm,ninety.correlation,ninety.lag);
    assert(ninety.valid && ninety.bpm>=87 && ninety.bpm<=93);
    ppg::BpmEstimator blank;
    int16_t ordered[ppg::BpmEstimator::N];
    assert(!blank.snapshot(ordered));
    for (unsigned i=0;i<ppg::BpmEstimator::N;++i) blank.push(0);
    assert(blank.snapshot(ordered));
    assert(!ppg::BpmEstimator::estimate(ordered,ppg::BpmEstimator::N).valid);
    blank.reset();
    assert(!blank.snapshot(ordered));
    std::puts("BPM_ESTIMATOR_TEST_PASS");
}
