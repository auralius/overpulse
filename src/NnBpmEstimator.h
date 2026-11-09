#pragma once
// Experimental neural heart-rate estimator for distributed PeakNet1D outputs.
// Uses ORIGINAL sensor sample indices; radio completion time is never used as
// the heartbeat timestamp. Does not depend on the DSP/autocorrelation BPM.
//
// A 256-sample inference window is normally produced every 64 sensor samples,
// but that stride can slip under radio load. Inspect all reliable interior
// logits [13, 243) so occasionally skipped launches do not create artificial
// 64-sample-only coverage gaps. Ignore 13 border samples on each side where
// convolution context is incomplete. Overlap is handled by absolute sensor
// sample indices and 500-ms peak de-duplication (not by packet arrival time).
// Results can return out of order. Conservative
// rhythm-quality checks return invalid instead of forcing a BPM number.
// NOT a clinical/medical device.

#include <stdint.h>

namespace ppg {
class NnBpmEstimator {
public:
    static const unsigned HZ = 100;
    static const unsigned WINDOW = 256;
    static const unsigned HOP = 64;
    static const unsigned EDGE = 13;
    static const unsigned OWN_FIRST = EDGE;                // 13
    static const unsigned OWN_LAST = WINDOW - EDGE;       // exclusive, 243
    static const unsigned REFRACTORY = 50;                // 500 ms
    static const int8_t THRESHOLD = 18;
    static const unsigned HISTORY = 1200;                // 12 seconds
    static const unsigned MAX_EVENTS = 32;
    static const unsigned MIN_PEAKS = 5;                 // >=4 intervals
    static const unsigned MAX_STALE = 300;                // 3 seconds

    struct Result {
        bool valid;
        uint16_t bpm;
        unsigned peaks;
        unsigned intervals;
    };

    NnBpmEstimator() { reset(); }

    void reset() { count_ = 0; }

    // window_end is the exclusive absolute sample index of this window.
    // now_sample is the current exclusive sensor sample index for expiration.
    // Returns number of candidate local maxima accepted or updated.
    unsigned observe(uint32_t window_end, const int8_t* logits, uint32_t now_sample) {
        if (!logits || window_end < WINDOW || window_end > now_sample) return 0;
        prune(now_sample);
        unsigned accepted = 0;
        for (unsigned i = OWN_FIRST; i < OWN_LAST; ++i) {
            const int8_t score = logits[i];
            if (score < THRESHOLD || score < logits[i-1] || score <= logits[i+1])
                continue;
            const uint32_t absolute = window_end - WINDOW + i;
            if (now_sample - absolute > HISTORY) continue;
            if (insert(absolute, score)) ++accepted;
        }
        return accepted;
    }

    Result estimate(uint32_t now_sample) const {
        Result invalid = {false, 0, 0, 0};
        // Stored events are always sorted by the sample index, not arrival time.
        uint32_t recent[MAX_EVENTS];
        unsigned n = 0;
        for (unsigned i = 0; i < count_; ++i) {
            const uint32_t t = events_[i].sample;
            if (t <= now_sample && now_sample - t <= HISTORY)
                recent[n++] = t;
        }
        invalid.peaks = n;
        if (n < MIN_PEAKS || now_sample - recent[n-1] > MAX_STALE)
            return invalid;

        unsigned intervals[MAX_EVENTS-1];
        for (unsigned i = 1; i < n; ++i) {
            const uint32_t dt = recent[i] - recent[i-1];
            // Physiology gate: 45-120 BPM. Missing peaks or extra detections
            // invalidate the estimate, rather than using the DSP BPM as a fill.
            if (dt < 50 || dt > 133) return invalid;
            intervals[i-1] = static_cast<unsigned>(dt);
        }
        const unsigned m = n-1;
        // Insertion sort: small, fixed-size arrays; no dynamic allocation.
        for (unsigned i = 1; i < m; ++i) {
            unsigned v = intervals[i], j = i;
            while (j > 0 && intervals[j-1] > v) {
                intervals[j] = intervals[j-1]; --j;
            }
            intervals[j] = v;
        }
        const unsigned median = intervals[m/2];
        // Reject markedly inconsistent pulse intervals (relative tolerance 25%).
        // This is deliberately conservative for an unvalidated NN detector.
        for (unsigned i = 0; i < m; ++i) {
            const unsigned d = intervals[i] > median ? intervals[i]-median : median-intervals[i];
            if (d*4u > median) return invalid;
        }
        Result out = {true, static_cast<uint16_t>((6000u + median/2u)/median), n, m};
        return out;
    }

private:
    struct Event { uint32_t sample; int8_t score; };
    Event events_[MAX_EVENTS];
    unsigned count_;

    void prune(uint32_t now_sample) {
        unsigned dst = 0;
        for (unsigned i=0; i<count_; ++i) {
            if (events_[i].sample <= now_sample &&
                now_sample - events_[i].sample <= HISTORY)
                events_[dst++] = events_[i];
        }
        count_ = dst;
    }

    bool insert(uint32_t sample, int8_t score) {
        // Check ALL competing nearby events before replacing any one. A new
        // candidate must win against each, so a failed replacement cannot
        // accidentally erase an earlier valid peak.
        for (unsigned i=0; i<count_; ++i) {
            const uint32_t t=events_[i].sample;
            const uint32_t distance = sample>t ? sample-t : t-sample;
            if (distance < REFRACTORY &&
                (score < events_[i].score ||
                 (score == events_[i].score && sample >= t))) return false;
        }
        unsigned dest=0;
        for (unsigned i=0; i<count_; ++i) {
            const uint32_t t=events_[i].sample;
            const uint32_t distance = sample>t ? sample-t : t-sample;
            if (distance >= REFRACTORY) events_[dest++]=events_[i];
        }
        count_=dest;
        if (count_ == MAX_EVENTS) {
            // Prefer keeping the newest events for ongoing pulse estimation.
            if (sample <= events_[0].sample) return false;
            for(unsigned i=1;i<count_;++i)events_[i-1]=events_[i];
            --count_;
        }
        unsigned j=count_;
        while(j>0 && events_[j-1].sample>sample) {
            events_[j]=events_[j-1]; --j;
        }
        events_[j].sample=sample;
        events_[j].score=score;
        ++count_;
        return true;
    }

};
} // namespace ppg
