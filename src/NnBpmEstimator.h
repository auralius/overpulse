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
// Results can return out of order. Evaluate the newest consistent run
// of intervals, allowing recovery from an older missed/extra detection.
// Three peaks yield a preliminary engineering estimate, not a clinical result.

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
    static const unsigned HISTORY = 600;                 // 6 seconds
    static const unsigned MAX_EVENTS = 32;
    static const unsigned MIN_PEAKS = 3;                 // >=2 consecutive intervals
    static const unsigned MAX_STALE = 300;                // 3 seconds
    static const unsigned EVAL_INTERVALS = 4;             // at most 5 recent peaks

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
        // Stored events are sorted by original sample index, not arrival time.
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

        // Extend a CONSISTENT run backward from the newest detected peak.
        // A bad old interval ends the run instead of invalidating all newer
        // good intervals. Bound the working set to 4 intervals (5 peaks).
        unsigned interval[EVAL_INTERVALS];
        unsigned m = 0;
        for (unsigned j = n - 1; j > 0 && m < EVAL_INTERVALS; --j) {
            const uint32_t dt = recent[j] - recent[j - 1];
            // A missing beat or an extra false beat may break the run.
            if (dt < 50 || dt > 133) break; // 45-120 BPM gate at 100 Hz
            interval[m++] = static_cast<unsigned>(dt);

            // Small insertion sort of a COPY, preserving time order above.
            unsigned sorted[EVAL_INTERVALS];
            for (unsigned k = 0; k < m; ++k) {
                unsigned v = interval[k], pos = k;
                while (pos && sorted[pos-1] > v) {
                    sorted[pos] = sorted[pos-1]; --pos;
                }
                sorted[pos] = v;
            }
            const unsigned median = sorted[m/2];
            bool consistent = true;
            for (unsigned k = 0; k < m; ++k) {
                const unsigned d = interval[k] > median
                    ? interval[k] - median : median - interval[k];
                if (d*4u > median) { consistent = false; break; }
            }
            if (!consistent) { --m; break; }
        }
        if (m + 1 < MIN_PEAKS) return invalid;

        unsigned sorted[EVAL_INTERVALS];
        for (unsigned k = 0; k < m; ++k) {
            unsigned v = interval[k], pos = k;
            while (pos && sorted[pos-1] > v) {
                sorted[pos] = sorted[pos-1]; --pos;
            }
            sorted[pos] = v;
        }
        const unsigned median = sorted[m/2];
        // peaks reports how many candidate events are held in history;
        // intervals reports how many consistent RECENT intervals were used.
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
