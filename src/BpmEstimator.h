#pragma once
// Independent, exploratory heart-rate estimate from the continuously filtered
// IR optical waveform. NOT inferred from PeakNet's unvalidated peak markers.
//
// Sampling is fixed at 100 Hz; all filtering state is continuous across frames.
// The 10-second autocorrelation estimator reports a value only when it finds
// a sufficiently reproducible interior peak (not a drifting-signal edge).
// This is an engineering visualization, not a medical instrument.

#include <math.h>
#include <stdint.h>

namespace ppg {
class BpmEstimator {
public:
    static const unsigned HZ = 100;
    static const unsigned N = 1000; // 10 s of filtered IR, 2 KB storage

    struct Result {
        bool valid;
        uint16_t bpm;
        float correlation;
        unsigned lag;
    };

    BpmEstimator() { reset(); }

    void reset() {
        next_ = 0;
        samples_ = 0;
        primed_ = false;
        previous_ = 0;
        hp_ = 0;
        lp_ = 0;
    }

    // Input: output of display-only ~0.3-4 Hz filter, in IR counts.
    // A further ~0.7-2 Hz filter isolates likely heart-rate periodicity.
    // Both stages are causal and updated once for every acquired sample.
    void push(int16_t filtered_ir) {
        const float x = static_cast<float>(filtered_ir);
        if (!primed_) {
            primed_ = true;
            previous_ = x;
        }
        hp_ = 0.958f * (hp_ + x - previous_);
        previous_ = x;
        lp_ += 0.112f * (hp_ - lp_);
        float v = roundf(lp_);
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        ring_[next_] = static_cast<int16_t>(v);
        next_ = (next_ + 1) % N;
        if (samples_ < N) ++samples_;
    }

    bool full() const { return samples_ == N; }

    // Thread safety: call this with the same sample-ring lock held by sensor_task.
    // Output is chronological, with no dynamic allocation.
    bool snapshot(int16_t* dest) const {
        if (!dest || !full()) return false;
        for (unsigned i=0; i<N; ++i) dest[i] = ring_[(next_+i)%N];
        return true;
    }

    static Result estimate(const int16_t* waveform, unsigned count) {
        Result invalid = {false, 0, 0.0f, 0};
        if (!waveform || count != N) return invalid;

        float mean = 0.0f;
        for (unsigned i=0; i<N; ++i) mean += waveform[i];
        mean /= N;
        float e = 0.0f;
        for (unsigned i=0; i<N; ++i) {
            const float d = waveform[i] - mean;
            e += d*d;
        }
        // Guard against no signal; low optical AC is common on MAX30100.
        if (e < 1.5f * 1.5f * N) return invalid;

        // 45..120 BPM corresponds approximately to lags 50..133 at 100 Hz.
        // Keep 1 neighboring lag to require a genuine autocorrelation maximum:
        // a broad monotonic decay from contact drift must not become 120 BPM.
        const unsigned FIRST = 49;
        const unsigned LAST = 134;
        float corr[LAST+1] = {};
        for (unsigned lag=FIRST; lag<=LAST; ++lag) {
            float xy=0.0f, xx=0.0f, yy=0.0f;
            for (unsigned i=0; i<N-lag; ++i) {
                const float a = waveform[i] - mean;
                const float b = waveform[i+lag] - mean;
                xy += a*b;
                xx += a*a;
                yy += b*b;
            }
            const float den = sqrtf(xx*yy);
            corr[lag] = den>0.00001f ? xy/den : 0.0f;
        }

        unsigned best_lag = 0;
        float best_correlation = 0.34f; // conservative quality gate
        for (unsigned lag=FIRST+1; lag<LAST; ++lag) {
            if (corr[lag]>=corr[lag-1] && corr[lag]>corr[lag+1] &&
                corr[lag]>best_correlation) {
                best_lag = lag;
                best_correlation = corr[lag];
            }
        }
        if (!best_lag) return invalid;
        const float rate = (60.0f * HZ) / best_lag;
        if (rate<45.0f || rate>120.0f) return invalid;
        Result out = {true, static_cast<uint16_t>(roundf(rate)),
                      best_correlation, best_lag};
        return out;
    }

private:
    int16_t ring_[N];
    unsigned next_, samples_;
    bool primed_;
    float previous_, hp_, lp_;
};
} // namespace ppg
