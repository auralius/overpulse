// Live MAX30100 -> signed INT8 CHW PeakNet1D -> PSRAM Noodle2 -> ST7789 screen.
// MODEL: already QAT-trained on unfiltered SYNTHETIC IR/Red; real beats are UNVERIFIED.
// Separate, identical filters are used for the IR/Red TFT waveforms;
// filtered IR also drives the independent DSP BPM estimator.
// NN BPM uses distributed PeakNet1D peak logits and source sample indices.
// PeakNet still receives original training-matched unfiltered raw-to-INT8 input.
#include <Arduino.h>
#include <Wire.h>
#include <TFT_eSPI.h>
#include <Noodle2.h>
#include <NoodleNow.h>
#include <PipelineTransfer.h>
#include <NoodleAdmission.h>
#include "Max30100Raw.h"
#include "PPGPreprocess.h"
#include "BpmEstimator.h"
#include "NnBpmEstimator.h"
#include "peaknet_ppg_model.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <math.h>
#include <string.h>

#ifndef PPG_SDA_PIN
#define PPG_SDA_PIN 17
#endif
#ifndef PPG_SCL_PIN
#define PPG_SCL_PIN 18
#endif
#ifndef PPG_I2C_HZ
#define PPG_I2C_HZ 400000
#endif
#ifndef PPG_DEBUG_CSV
#define PPG_DEBUG_CSV 0
#endif
#ifndef PPG_UI_REFRESH_MS
#define PPG_UI_REFRESH_MS 400 // ~2.5 live plot updates/s; avoid starving ESP-NOW
#endif

#ifndef NOODLE_MAX_OCCUPANCY
#define NOODLE_MAX_OCCUPANCY 3
#endif
#ifndef NOODLE_HOST_OCCUPANCY
#define NOODLE_HOST_OCCUPANCY NOODLE_MAX_OCCUPANCY
#endif
#ifndef NOODLE_RESERVED_WORKERS
#define NOODLE_RESERVED_WORKERS 1
#endif
// Frame slots are independent of worker count (a worker can process many frames).
// Keep the static frame array bounded to avoid excess embedded RAM usage.
static_assert(NOODLE_MAX_OCCUPANCY >= 1 && NOODLE_MAX_OCCUPANCY <= 32,
              "NOODLE_MAX_OCCUPANCY must be between 1 and 32");
static_assert(NOODLE_HOST_OCCUPANCY >= 1 && NOODLE_HOST_OCCUPANCY <= NOODLE_MAX_OCCUPANCY,
              "NOODLE_HOST_OCCUPANCY must be in [1, NOODLE_MAX_OCCUPANCY]");
static_assert(NOODLE_RESERVED_WORKERS >= 0 &&
              NOODLE_RESERVED_WORKERS <= NOODLE_MAX_WORKERS,
              "NOODLE_RESERVED_WORKERS must be in [0, NOODLE_MAX_WORKERS]");

namespace {
constexpr unsigned MAX_WORKERS=NOODLE_MAX_WORKERS;
constexpr unsigned MAX_OCCUPANCY=NOODLE_MAX_OCCUPANCY;
constexpr unsigned HOST_OCCUPANCY=NOODLE_HOST_OCCUPANCY;
constexpr unsigned RESERVED_WORKERS=NOODLE_RESERVED_WORKERS;
constexpr unsigned W=256;
constexpr unsigned STEP=64;          // run once per 0.64 seconds, approximately
// Two time-aligned, separately auto-scaled AC traces within 320x170 pixels.
constexpr unsigned DRAW_LEFT=35;  // allow a channel label gutter
constexpr unsigned DRAW_WIDTH=278;
constexpr int IR_WAVE_TOP=38;
constexpr int IR_WAVE_BOT=86;
constexpr int RED_WAVE_TOP=90;
constexpr int RED_WAVE_BOT=138;
constexpr int SCORE_TOP=143;
constexpr int SCORE_BOT=156;
constexpr int8_t PEAK_LOGIT_THRESHOLD=ppg::NnBpmEstimator::THRESHOLD;
constexpr unsigned REFRACTORY=ppg::NnBpmEstimator::REFRACTORY;
constexpr unsigned EDGE=ppg::NnBpmEstimator::EDGE;
// Threshold, refractory, and edge guard are shared by NN BPM and red dots.

portMUX_TYPE sample_lock=portMUX_INITIALIZER_UNLOCKED;
uint16_t ir_ring[W]={}, red_ring[W]={};
int16_t ir_wave_ring[W]={}, red_wave_ring[W]={};
volatile uint32_t samples_seen=0, continuous=0, fifo_overruns=0, i2c_errors=0;
volatile bool sensor_ready=false;
volatile uint32_t sensor_epoch=0; // discontinuities / contact loss invalidate NN peak history
uint32_t last_window=0;
uint16_t raw_ir[W],raw_red[W];
int16_t ac_ir_wave[W], ac_red_wave[W];
int8_t model_input[2*W],model_logits[W];
uint8_t marks[W]={};
// NN marker positions refer to an absolute source-sample window, not screen
// pixels. Never draw stale markers at the wrong location as the live plot moves.
bool have_nn_overlay=false;
uint32_t nn_overlay_end=0,nn_overlay_epoch=0;
unsigned nn_overlay_count=0;
ppg::BpmEstimator bpm_estimator; // independent IR-AC period estimate
ppg::NnBpmEstimator nn_bpm_estimator; // PeakNet1D-derived beats, not DSP
ppg::NnBpmEstimator::Result nn_bpm_result={false,0,0,0};
uint32_t nn_epoch=0;
int16_t bpm_window[ppg::BpmEstimator::N];
ppg::BpmEstimator::Result bpm_result={false,0,0.0f,0};
uint32_t last_bpm_eval=0;
uint32_t next_frame=1,completed_frames=0,aborted_frames=0;
uint32_t recent_latency_ms=0;
// Implemented by host registry below; the UI does not need radio traffic.
unsigned connected();
TFT_eSPI tft;
peaknet_ppg::Model model;
noodle2::NoodleGraph graph;

// Streaming display-only band-pass: HP ~0.3Hz, LP ~4Hz, constant state.
// Do not feed this to existing QAT model without retraining.
struct DisplayFilter {
    bool primed=false;
    float prev_x=0, hp=0, lp=0;
    int16_t push(uint16_t raw) {
        const float x=static_cast<float>(raw);
        if(!primed) { prev_x=x; primed=true; return 0; }
        const float next_hp=0.981325f*(hp+x-prev_x);
        hp=next_hp;
        prev_x=x;
        lp+=0.222232f*(hp-lp);
        const int out=static_cast<int>(roundf(lp));
        return static_cast<int16_t>(out>32767?32767:(out<-32768?-32768:out));
    }
} ir_display_filter, red_display_filter;

void sensor_task(void*) {
    ppg::Max30100Raw sensor(Wire);
    if (!sensor.begin()) {
        portENTER_CRITICAL(&sample_lock); ++i2c_errors; portEXIT_CRITICAL(&sample_lock);
        Serial.println("ERROR MAX30100_NOT_FOUND_OR_CONFIG_FAILED");
        vTaskDelete(nullptr);
        return;
    }
    portENTER_CRITICAL(&sample_lock); sensor_ready=true; portEXIT_CRITICAL(&sample_lock);
    Serial.printf("MAX30100_OK addr=0x57 100Hz LED=0x%02X\n", unsigned(PPG_LED_CONFIG));
    ppg::Sample batch[8];
    bool contact_present=false;
    for (;;) {
        uint8_t n=0;
        bool overflow=false;
        if(!sensor.read(batch,8,n,overflow)) {
            portENTER_CRITICAL(&sample_lock);
            ++i2c_errors;continuous=0;++sensor_epoch;
            bpm_estimator.reset();
            portEXIT_CRITICAL(&sample_lock);
        } else if (overflow) {
            portENTER_CRITICAL(&sample_lock);
            ++fifo_overruns;continuous=0;++sensor_epoch;
            bpm_estimator.reset();
            portEXIT_CRITICAL(&sample_lock);
        } else if(n) {
            // One critical section per batch; never I2C or SPI inside the lock.
            portENTER_CRITICAL(&sample_lock);
            for(unsigned i=0;i<n;++i) {
                const uint32_t index=samples_seen++;
                ir_ring[index%W]=batch[i].ir;
                red_ring[index%W]=batch[i].red;
                // Same filter coefficients and sample index; independent states.
                ir_wave_ring[index%W]=ir_display_filter.push(batch[i].ir);
                red_wave_ring[index%W]=red_display_filter.push(batch[i].red);
                // IR autocorrelation and NN sample history both require
                // uninterrupted optical contact (same threshold as admission).
                const bool contact=batch[i].ir>=1000 && batch[i].red>=1000;
                if(!contact && contact_present) {++sensor_epoch;continuous=0;}
                contact_present=contact;
                if(contact) {
                    bpm_estimator.push(ir_wave_ring[index%W]);
                    if(continuous<W) ++continuous;
                } else {
                    bpm_estimator.reset();
                    continuous=0;
                }
            }
            portEXIT_CRITICAL(&sample_lock);
#if PPG_DEBUG_CSV
            for(unsigned i=0;i<n;++i)
                Serial.printf("PPG,%u,%u\n",unsigned(batch[i].ir),unsigned(batch[i].red));
#endif
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void build_graph() {
    // Existing Noodle2 CHW graph: six Conv1D mapped to Conv2D height=1.
    auto x=graph.input(2,1,W);
    x=graph.conv2d(x,model.conv01,noodle2::Conv2D::same(1,9),noodle2::RELU);
    x=graph.conv2d(x,model.conv02,noodle2::Conv2D::same(1,7),noodle2::RELU);
    x=graph.conv2d(x,model.conv03,noodle2::Conv2D::same(1,7),noodle2::RELU);
    x=graph.conv2d(x,model.conv04,noodle2::Conv2D::same(1,7),noodle2::RELU);
    x=graph.conv2d(x,model.conv05,noodle2::Conv2D::same(1,1),noodle2::RELU);
    graph.conv2d(x,model.conv06,noodle2::Conv2D::same(1,1),noodle2::LOGITS);
}

// Draw without sprites; SPI belongs only to Arduino loop() on core 1.
void draw_header() {
    tft.fillScreen(TFT_BLACK);
    tft.fillRect(0,0,320,35,TFT_DARKGREY);
    tft.setTextColor(TFT_CYAN,TFT_DARKGREY);
    tft.setTextSize(1);
    tft.setCursor(8,4);tft.print("OVERPULSE / Noodle2");
    tft.setTextColor(TFT_WHITE,TFT_DARKGREY);
    tft.setCursor(8,19);tft.print("MAX30100 / ESP-NOW");
}
void draw_bpm() {
    // Independent values: NN from distributed logits, DSP from autocorrelation.
    // A missing/irregular NN beat sequence remains invalid; never copy DSP BPM.
    tft.fillRect(159,0,161,35,TFT_DARKGREY);
    tft.setTextSize(1);
    tft.setTextColor(TFT_CYAN,TFT_DARKGREY);
    tft.setCursor(168,3);tft.print("NN BPM");
    tft.setCursor(245,3);tft.print("DSP BPM");
    tft.setTextSize(2);
    tft.setTextColor(nn_bpm_result.valid ? TFT_GREEN : TFT_YELLOW,TFT_DARKGREY);
    tft.setCursor(167,16);
    if(nn_bpm_result.valid) tft.printf("%3u",unsigned(nn_bpm_result.bpm));
    else tft.print("---");
    tft.setTextColor(bpm_result.valid ? TFT_WHITE : TFT_YELLOW,TFT_DARKGREY);
    tft.setCursor(245,16);
    if(bpm_result.valid) tft.printf("%3u",unsigned(bpm_result.bpm));
    else tft.print("---");
    tft.setTextSize(1);
}
// Both traces use identical horizontal sample indices, with separate vertical
// auto-scaling (different LED intensities and optical DC offsets). Auto-scale
// reveals shape, not comparable optical amplitude or calibrated physiology.
int wave_x(unsigned i) {
    return int(DRAW_LEFT + i*(DRAW_WIDTH-1)/(W-1));
}
// Render the samples that actually exist. Before the first 256 samples arrive,
// the short trace grows from right to left without inventing older samples.
void draw_ac_trace(const int16_t* data, unsigned first, int top, int bottom,
                   uint32_t color) {
    tft.drawRect(6,top,308,bottom-top+1,TFT_DARKGREY);
    tft.drawFastHLine(DRAW_LEFT,(top+bottom)/2,DRAW_WIDTH,TFT_DARKGREY);
    if(first>=W) return;
    int minv=int(data[first]),maxv=int(data[first]);
    for(unsigned i=first+1;i<W;++i) {
        const int v=int(data[i]);
        if(v<minv) minv=v;
        if(v>maxv) maxv=v;
    }
    const int range=maxv-minv;
    auto plot_y=[&](unsigned i)->int {
        if(range==0)return (top+bottom)/2;
        return bottom-2-(int(data[i])-minv)*(bottom-top-4)/range;
    };
    for(unsigned i=first+1;i<W;++i)
        tft.drawLine(wave_x(i-1),plot_y(i-1),wave_x(i),plot_y(i),color);
}

// The display is a sensor consumer, NOT an inference-result consumer.
// Snapshots are taken under the sensor lock and SPI rendering runs after
// releasing it. No dependency on worker registration or FRAME_DONE.
void draw_live_frame() {
    uint32_t seen,epoch;
    bool ready;
    portENTER_CRITICAL(&sample_lock);
    seen=samples_seen;
    epoch=sensor_epoch;
    ready=sensor_ready;
    const unsigned count=seen<W?static_cast<unsigned>(seen):W;
    const unsigned first=W-count;
    for(unsigned i=0;i<count;++i) {
        const uint32_t sample_index=seen-count+i;
        ac_ir_wave[first+i]=ir_wave_ring[sample_index%W];
        ac_red_wave[first+i]=red_wave_ring[sample_index%W];
    }
    portEXIT_CRITICAL(&sample_lock);

    draw_bpm();
    tft.fillRect(0,35,320,135,TFT_BLACK);
    draw_ac_trace(ac_ir_wave,first,IR_WAVE_TOP,IR_WAVE_BOT,TFT_GREEN);
    draw_ac_trace(ac_red_wave,first,RED_WAVE_TOP,RED_WAVE_BOT,TFT_RED);
    tft.setTextSize(1);
    tft.setTextColor(TFT_GREEN,TFT_BLACK);
    tft.setCursor(9,IR_WAVE_TOP+6);tft.print("IR");
    tft.setTextColor(TFT_RED,TFT_BLACK);
    tft.setCursor(9,RED_WAVE_TOP+6);tft.print("RED");

    // Old NN detections are indexed by their *original* 100-Hz sample
    // numbers. Align with the current scrolling ring and discard off-screen
    // markers (or everything after a FIFO/contact continuity break).
    unsigned visible_marks=0;
    if(have_nn_overlay && nn_overlay_epoch==epoch && seen>=W) {
        int ir_min=int(ac_ir_wave[0]),ir_max=ir_min;
        for(unsigned i=1;i<W;++i) {
            const int v=int(ac_ir_wave[i]);
            if(v<ir_min)ir_min=v;
            if(v>ir_max)ir_max=v;
        }
        const int64_t offset=static_cast<int64_t>(nn_overlay_end)-seen;
        for(unsigned i=EDGE;i<W-EDGE;++i)if(marks[i]) {
            const int64_t screen_i=static_cast<int64_t>(i)+offset;
            if(screen_i<0 || screen_i>=static_cast<int64_t>(W))continue;
            const unsigned at=static_cast<unsigned>(screen_i);
            const int y=ir_max==ir_min ? (IR_WAVE_TOP+IR_WAVE_BOT)/2 :
              IR_WAVE_BOT-2-(int(ac_ir_wave[at])-ir_min)*(IR_WAVE_BOT-IR_WAVE_TOP-4)/(ir_max-ir_min);
            tft.fillCircle(wave_x(at),y,3,TFT_YELLOW);
            ++visible_marks;
        }
    }

    // Display logits only after a valid inference result; the strip remains
    // empty (rather than showing stale results) while the radio is offline.
    tft.drawRect(6,SCORE_TOP,308,SCORE_BOT-SCORE_TOP+1,TFT_DARKGREY);
    if(have_nn_overlay && nn_overlay_epoch==epoch &&
       seen>=nn_overlay_end && seen-nn_overlay_end<W) {
        for(unsigned i=0;i<W;i+=2) {
            const int64_t at=static_cast<int64_t>(i)+nn_overlay_end-seen;
            if(at<0 || at>=static_cast<int64_t>(W))continue;
            const int logit=int(model_logits[i]);
            int bar=(logit+64)/12;
            if(bar<0)bar=0;
            if(bar>10)bar=10;
            tft.drawFastVLine(wave_x(static_cast<unsigned>(at)),SCORE_BOT-bar,bar+1,TFT_CYAN);
        }
    }
    tft.setTextColor(TFT_YELLOW,TFT_BLACK);
    tft.setCursor(8,159);
    if(!ready)tft.print("WAITING FOR MAX30100");
    else if(count<W)tft.printf("COLLECTING %u/256",count);
    else if(!completed_frames) tft.printf("NN WAIT W=%u",connected());
    else tft.printf("NN dots %u",visible_marks);
    if(completed_frames) {
        tft.setTextColor(TFT_WHITE,TFT_BLACK);
        tft.setCursor(118,159);tft.printf("%lums",(unsigned long)recent_latency_ms);
    }
    tft.setTextColor(TFT_WHITE,TFT_BLACK);
    tft.setCursor(242,159);tft.print("IR+RED");
}

unsigned pick_candidates() {
    memset(marks,0,sizeof(marks));
    // Greedy non-maximum suppression for screen markers, not HR measurement.
    unsigned selected=0;
    for(unsigned n=0;n<6;++n) {
        int best=-1;
        int8_t score=PEAK_LOGIT_THRESHOLD-1;
        for(unsigned i=EDGE;i<W-EDGE;++i) {
            const int8_t s=model_logits[i];
            if(s<PEAK_LOGIT_THRESHOLD || marks[i]) continue;
            if(s < model_logits[i-1] || s < model_logits[i+1]) continue;
            bool near=false;
            for(unsigned k=EDGE;k<W-EDGE;++k)
                if(marks[k] && (int(i)>int(k) ? i-k : k-i)<REFRACTORY) {near=true;break;}
            if(!near && s>score) {best=int(i);score=s;}
        }
        if(best<0)break;
        marks[best]=1;
        ++selected;
    }
    return selected;
}

// Distributed compute state: bounded frames in flight, each stage scheduled to
// whichever ESP8266 worker is currently free. Workers exchange
// CHW INT8 tensors directly; the host receives only the final 256-byte output.
noodlepipe::NowLink radio;
noodlepipe::Sender host_tx(0);
noodlepipe::Receiver host_rx(0);
struct Node {
    bool online,busy;uint32_t last_seen,jobs;
    Node():online(false),busy(false),last_seen(0),jobs(0){}
} workers[MAX_WORKERS+1]; // slot 0 is the host; workers are 1..MAX_WORKERS
struct Frame {
    bool active,done,forward_pending;
    uint32_t id,start_ms,progress_ms,last_forward;
    uint32_t sample_end,epoch; // original exclusive 100-Hz source sample count
    uint8_t stage,worker,source_worker,forward_stage;
    bool local_handoff;
    int16_t ir_waveform[W],red_waveform[W];
    Frame():active(false),done(false),forward_pending(false),id(0),start_ms(0),
      progress_ms(0),last_forward(0),sample_end(0),epoch(0),
      stage(0),worker(0),source_worker(0),
      forward_stage(0),local_handoff(false),ir_waveform{0},red_waveform{0}{}
} frames[MAX_OCCUPANCY];
 uint32_t last_hello_ms=0,last_ui_ms=0,last_wave_ui_ms=0;
uint32_t radio_errors=0,skipped_windows=0;

Frame* frame_for(uint32_t id) {
    for(unsigned i=0;i<MAX_OCCUPANCY;++i)if(frames[i].active && frames[i].id==id)return &frames[i];
    return nullptr;
}
Frame* free_frame() {
    for(unsigned i=0;i<MAX_OCCUPANCY;++i)if(!frames[i].active)return &frames[i];
    return nullptr;
}
unsigned connected() {
    unsigned n=0;const uint32_t now=millis();
    for(unsigned i=1;i<=MAX_WORKERS;++i)if(workers[i].online && now-workers[i].last_seen<5000)++n;
    return n;
}
unsigned idle_workers() {
    unsigned n=0;const uint32_t now=millis();
    for(unsigned i=1;i<=MAX_WORKERS;++i)
        if(workers[i].online && !workers[i].busy && now-workers[i].last_seen<5000)++n;
    return n;
}
unsigned active_frames() {
    unsigned n=0;
    for(unsigned i=0;i<MAX_OCCUPANCY;++i)if(frames[i].active)++n;
    return n;
}
int select_worker(int avoid=-1) {
    int best=-1;
    for(int i=1;i<=static_cast<int>(MAX_WORKERS);++i) {
        if(i==avoid || !workers[i].online || workers[i].busy ||
           millis()-workers[i].last_seen>5000)continue;
        if(best<0 || workers[i].jobs<workers[best].jobs)best=i;
    }
    return best;
}
void abort_frame(Frame& f,const char* reason) {
    Serial.printf("FRAME_ABORT id=%lu stage=%u reason=%s\n",(unsigned long)f.id,
                  unsigned(f.stage),reason);
    ++aborted_frames;
    // A stalled worker remains reserved until it advertises FREE or is rebooted.
    // Do not silently reuse a possibly-busy worker and corrupt its activation.
    f.active=false;
}
void announce_host() {
    radio.send(noodlepipe::BROADCAST,noodlepipe::packet(noodlepipe::HELLO,0));
}
void forward_command(Frame& f,uint32_t now) {
    if(!f.forward_pending)return;
    if(f.last_forward && now-f.last_forward<450)return;
    noodlepipe::Packet p=noodlepipe::packet(f.local_handoff?noodlepipe::RUN_LOCAL:noodlepipe::FORWARD,
                                   0,f.id,f.forward_stage);
    // All devices derive the same MAC from this ID; no MAC in FORWARD payload.
    if(!f.local_handoff)
        p.payload[0]=f.forward_stage==noodlepipe::RESULT_STAGE
                     ? noodlepipe::HOST_ID : f.worker;
    if(workers[f.source_worker].online &&
       radio.send_to(f.source_worker,p,f.local_handoff?0:1))
        f.last_forward=now?now:1;
}
void sync_nn_epoch(uint32_t current_epoch) {
    if(nn_epoch != current_epoch) {
        nn_bpm_estimator.reset();
        nn_bpm_result={false,0,0,0};
        nn_epoch=current_epoch;
        have_nn_overlay=false;
        nn_overlay_count=0;
        memset(marks,0,sizeof(marks));
    }
}
void consume_result(Frame& f,const int8_t* logits) {
    uint32_t seen,epoch;
    portENTER_CRITICAL(&sample_lock);
    seen=samples_seen; epoch=sensor_epoch;
    portEXIT_CRITICAL(&sample_lock);
    sync_nn_epoch(epoch);
    if(f.epoch!=epoch) {abort_frame(f,"sensor continuity changed");return;}

    // The waveform is always live; the result supplies *only* the NN overlay.
    // Do not let a late, older frame replace a newer overlay.
    unsigned marks_found=0;
    if(!have_nn_overlay || f.sample_end>=nn_overlay_end) {
        memcpy(model_logits,logits,W);
        marks_found=pick_candidates();
        nn_overlay_count=marks_found;
        nn_overlay_end=f.sample_end;
        nn_overlay_epoch=f.epoch;
        have_nn_overlay=true;
    }
    const unsigned nn_updates=nn_bpm_estimator.observe(f.sample_end,logits,seen);
    nn_bpm_result=nn_bpm_estimator.estimate(seen);
    recent_latency_ms=millis()-f.start_ms;
    // No TFT access here: the periodic live display loop runs independently.
    Serial.printf("FRAME_DONE id=%lu t_ms=%lu latency_ms=%lu dots=%u nn_new=%u "
                  "nn_bpm=%u nn_valid=%u nn_peaks=%u dsp_bpm=%u dsp_valid=%u complete=%lu\n",
        (unsigned long)f.id,(unsigned long)millis(),(unsigned long)recent_latency_ms,
        marks_found,nn_updates,unsigned(nn_bpm_result.bpm),unsigned(nn_bpm_result.valid),
        nn_bpm_result.peaks,unsigned(bpm_result.bpm),unsigned(bpm_result.valid),
        (unsigned long)(completed_frames+1));
    f.active=false;++completed_frames;
}
void receive_radio(const noodlepipe::Message& m) {
    const noodlepipe::Packet& p=m.p;
    if(p.kind==noodlepipe::HELLO && noodlepipe::worker_id_valid(p.sender)) {
        // Validate identity even in host-only tests which bypass NowLink::receive.
        if(!noodlepipe::node_mac_matches(p.sender,m.mac))return;
        Node& node=workers[p.sender];
        if(!node.online) {
            Serial.printf("JOIN W%u %02X:%02X:%02X:%02X:%02X:%02X\n",p.sender,
                          m.mac[0],m.mac[1],m.mac[2],m.mac[3],m.mac[4],m.mac[5]);
        }
        node.online=true;node.last_seen=millis();
        // Only recover a busy slot if no active frame still refers to it.
        if(p.flags&1) {
            bool referenced=false;
            for(unsigned i=0;i<MAX_OCCUPANCY;++i)if(frames[i].active &&
               (frames[i].worker==p.sender ||
                (frames[i].forward_pending && frames[i].source_worker==p.sender)))
                referenced=true;
            if(!referenced)node.busy=false;
        }
        return;
    }
    if(host_tx.handle(m))return;
    if(p.kind==noodlepipe::PREPARE || p.kind==noodlepipe::DATA ||
       p.kind==noodlepipe::ACK_REQUEST) {
        // Only final logits come back to the host. No intermediate host relay.
        if(p.stage==noodlepipe::RESULT_STAGE)host_rx.handle(radio,m);
        return;
    }
    if(!noodlepipe::worker_id_valid(p.sender) || !workers[p.sender].online ||
       !noodlepipe::node_mac_matches(p.sender,m.mac))return;
    workers[p.sender].last_seen=millis();
    if(p.kind==noodlepipe::RELEASE) {
        workers[p.sender].busy=false;
        for(unsigned i=0;i<MAX_OCCUPANCY;++i)if(frames[i].active &&
           frames[i].forward_pending && frames[i].source_worker==p.sender) {
            frames[i].forward_pending=false;
        }
        return;
    }
    if(p.kind==noodlepipe::ERROR_PACKET) {
        ++radio_errors;
        Serial.printf("WORKER_ERROR w=%u id=%lu stage=%u code=%u\n",p.sender,
                      (unsigned long)p.frame,p.stage,p.flags);
        return;
    }
    if(p.kind!=noodlepipe::DONE)return;
    Frame* f=frame_for(p.frame);
    if(!f || !f->active || f->stage!=p.stage || f->worker!=p.sender || f->done)return;
    if(p.size!=noodlepipe::tensor_output(p.stage)) {
        abort_frame(*f,"unexpected output size");return;
    }
    const uint32_t stage_elapsed_ms=millis()-f->progress_ms;
    f->done=true;f->progress_ms=millis();
    // RUN_LOCAL does not produce a RELEASE; a new DONE also completes that handoff.
    if(f->forward_pending && f->local_handoff)f->forward_pending=false;
    Serial.printf("STAGE_DONE id=%lu stage=%u worker=%u size=%u crc=%08lx stage_elapsed_ms=%lu\n",
        (unsigned long)p.frame,p.stage,p.sender,p.size,(unsigned long)p.crc,
        (unsigned long)stage_elapsed_ms);
}
void drive_frames() {
    const uint32_t now=millis();
    for(unsigned i=0;i<MAX_OCCUPANCY;++i) {
        Frame& f=frames[i];if(!f.active)continue;
        if(now-f.progress_ms>30000) {abort_frame(f,"stage timeout");continue;}
        if(f.forward_pending)forward_command(f,now);
        if(!f.done)continue;
        const uint8_t next_stage=f.stage+1;
        int next=next_stage==noodlepipe::RESULT_STAGE ? 0 : select_worker(f.worker);
        if(next<0) {
            // Allows a single ESP8266 to execute the full model for bring-up.
            // Multiple available workers take precedence to encourage overlap.
            if(now-f.progress_ms<500)continue;
            next=f.worker;
        }
        Serial.printf("STAGE_ASSIGN id=%lu stage=%u from=%u to=%d wait_ms=%lu\n",
            (unsigned long)f.id,unsigned(next_stage),unsigned(f.worker),next,
            (unsigned long)(now-f.progress_ms));
        f.done=false;
        f.source_worker=f.worker;
        f.local_handoff=(next==f.worker);
        f.worker=next==0?f.source_worker:static_cast<uint8_t>(next);
        f.stage=next_stage;f.forward_stage=next_stage;
        f.forward_pending=true;f.last_forward=0;f.progress_ms=now;
        if(next>0 && !f.local_handoff) {
            workers[next].busy=true;++workers[next].jobs;
        }
        forward_command(f,now);
    }
}
void try_launch() {
    if(host_tx.state()==noodlepipe::Sender::COMPLETE) {
        Serial.printf("HOST_TX_V2 frame=%lu transfer_ms=%lu packets=%lu retries=%lu acks=%lu\n",
            (unsigned long)host_tx.frame(),(unsigned long)host_tx.transfer_ms(millis()),
            (unsigned long)host_tx.sent_packets(),(unsigned long)host_tx.retransmitted(),
            (unsigned long)host_tx.ack_count());
        host_tx.clear();
    }
    if(host_tx.state()==noodlepipe::Sender::FAILED) {
        Frame* f=frame_for(host_tx.frame());if(f)abort_frame(*f,"host TX failed");
        host_tx.clear();
    }
    if(host_tx.state()!=noodlepipe::Sender::IDLE)return;
    // Separate host admission credits from total frame slots. Reserve spare
    // workers for intermediate activation forwarding, not new input jobs.
    // With only one worker online, allow the established RUN_LOCAL fallback.
    if(!noodlepipe::host_can_admit(active_frames(),HOST_OCCUPANCY,
                                   connected(),idle_workers(),RESERVED_WORKERS))return;
    Frame* f=free_frame();const int worker=select_worker();
    if(!f || worker<0)return;
    uint32_t seen,valid,epoch;
    portENTER_CRITICAL(&sample_lock);
    seen=samples_seen;valid=continuous;epoch=sensor_epoch;
    if(valid>=W && seen-last_window>=STEP) {
        for(unsigned i=0;i<W;++i) {
            const unsigned ix=(seen-W+i)%W;
            raw_ir[i]=ir_ring[ix];raw_red[i]=red_ring[ix];
            f->ir_waveform[i]=ir_wave_ring[ix];
            f->red_waveform[i]=red_wave_ring[ix];
        }
    }
    portEXIT_CRITICAL(&sample_lock);
    if(valid<W || seen-last_window<STEP)return;
    if(last_window && seen-last_window>STEP)
        skipped_windows+=(seen-last_window)/STEP-1;
    last_window=seen;
    uint32_t ir_sum=0,red_sum=0;
    for(unsigned i=0;i<W;++i){ir_sum+=raw_ir[i];red_sum+=raw_red[i];}
    if(ir_sum/W<1000 || red_sum/W<1000)return;
    ppg::raw_to_chw_i8(raw_ir,raw_red,model_input);
    f->active=true;f->done=false;f->forward_pending=false;
    f->id=next_frame++;f->start_ms=millis();f->progress_ms=f->start_ms;
    f->sample_end=seen;f->epoch=epoch;
    f->stage=0;f->worker=static_cast<uint8_t>(worker);
    uint8_t worker_address[6];
    noodlepipe::node_mac(static_cast<uint8_t>(worker),worker_address);
    if(!host_tx.start(radio,worker_address,f->id,0,model_input,512)) {
        f->active=false;return;
    }
    workers[worker].busy=true;++workers[worker].jobs;
    Serial.printf("FRAME_START id=%lu worker=%d seq_end=%lu host_outstanding=%u/%u "
                  "pipeline_slots=%u idle_workers=%u reserve=%u\n",
                  (unsigned long)f->id,worker,(unsigned long)seen,
                  active_frames(),HOST_OCCUPANCY,MAX_OCCUPANCY,
                  idle_workers(),noodlepipe::effective_reserve(connected(),RESERVED_WORKERS));
}
void refresh_bpm(uint32_t seen) {
    if(seen-last_bpm_eval<200)return;
    bool have=false;
    portENTER_CRITICAL(&sample_lock);
    have=bpm_estimator.snapshot(bpm_window);
    portEXIT_CRITICAL(&sample_lock);
    bpm_result=have ? ppg::BpmEstimator::estimate(bpm_window,ppg::BpmEstimator::N)
                    : ppg::BpmEstimator::Result{false,0,0.0f,0};
    last_bpm_eval=seen;
}
void fatal(const char* message) {
    Serial.printf("ERROR %s\n",message);
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_RED,TFT_BLACK);
    tft.setTextSize(1);
    tft.setCursor(10,55); tft.print(message);
    for(;;)delay(1000);
}
}

void setup() {
    Serial.begin(115200);
    delay(250);
    Serial.println("BOOT: BEFORE TFT INIT");
    Serial.flush();
    pinMode(14,OUTPUT);
    digitalWrite(14,HIGH);
    tft.init();
    Serial.println("BOOT: TFT INIT OK");
    tft.setRotation(1); // 320 x 170 landscape on 170 x 320 ST7789
    draw_header();
    Serial.println("BOOT: DISPLAY HEADER OK");
    Serial.println("NOODLE2_PPG_ESPNOW_HOST: dynamic stage scheduling");
    if(!psramFound()) fatal("PSRAM NOT DETECTED");
    if(model.bind_flash()!=noodle2::OK) fatal("MODEL BIND FAILED");
    if(graph.set_arena_memory(noodle2::ARENA_PSRAM)!=noodle2::OK)
        fatal("PSRAM POLICY FAILED");
    build_graph();
    if(graph.compile()!=noodle2::OK) fatal("GRAPH COMPILE FAILED");
    if(graph.arena_memory()!=noodle2::ARENA_PSRAM) fatal("NOT IN PSRAM");
    Serial.printf("MODEL trained=1 channels=2 samples=256 arena=%lu PSRAM=1\n",
                  static_cast<unsigned long>(graph.arena_bytes()));
    Serial.printf("NOODLE_MAX_WORKERS=%u MAX_OCCUPANCY=%u HOST_OCCUPANCY=%u\n",
                  MAX_WORKERS, MAX_OCCUPANCY, HOST_OCCUPANCY);
    Serial.printf("NOODLE_TRANSPORT_CRC=%d (0=OFF, 1=ON)\n", NOODLE_TRANSPORT_CRC);
    Serial.printf("NOODLE_ACK_INTERVAL=%d (4=baseline, 18=whole tensor)\n",
                  NOODLE_ACK_INTERVAL);
    if(!radio.begin(0))fatal("ESP-NOW INITIALIZATION FAILED");
    Serial.printf("RADIO host=%02X:%02X:%02X:%02X:%02X:%02X channel=%u\n",
        radio.address()[0],radio.address()[1],radio.address()[2],radio.address()[3],
        radio.address()[4],radio.address()[5],noodlepipe::CHANNEL);
    Wire.begin(PPG_SDA_PIN,PPG_SCL_PIN);
    Wire.setClock(PPG_I2C_HZ);
    if(xTaskCreatePinnedToCore(sensor_task,"ppg_fifo",4096,nullptr,3,nullptr,0)!=pdPASS)
        fatal("SENSOR TASK FAILED");
    draw_live_frame();
}

void loop() {
    noodlepipe::Message m;
    while(radio.poll(m))receive_radio(m);
    const uint32_t now=millis();
    if(now-last_hello_ms>1000) {announce_host();last_hello_ms=now;}
    host_tx.tick(radio,now);
    if(host_rx.complete()) {
        Frame* f=frame_for(host_rx.frame());
        if(f && f->stage==noodlepipe::RESULT_STAGE && host_rx.size()==W)
            consume_result(*f,host_rx.data());
        host_rx.release();
    }
    drive_frames();
    try_launch();
    uint32_t seen,valid,over,err,epoch;
    portENTER_CRITICAL(&sample_lock);
    seen=samples_seen;valid=continuous;over=fifo_overruns;err=i2c_errors;
    epoch=sensor_epoch;
    portEXIT_CRITICAL(&sample_lock);
    sync_nn_epoch(epoch);
    refresh_bpm(seen);
    if(now-last_wave_ui_ms>=PPG_UI_REFRESH_MS) {
        last_wave_ui_ms=now;
        nn_bpm_result=nn_bpm_estimator.estimate(seen);
        draw_live_frame();
    }
    if(now-last_ui_ms>=2000) {
        last_ui_ms=now;
        const unsigned active=active_frames();
        const unsigned online=connected();
        const unsigned idle=idle_workers();
        Serial.printf("PIPE_STATUS workers=%u idle=%u in_flight=%u max_occupancy=%u "
                      "host_occupancy=%u reserved_workers=%u completed=%lu aborted=%lu "
                      "samples=%lu valid=%lu nn_bpm=%u nn_valid=%u nn_peaks=%u "
                      "dsp_bpm=%u dsp_valid=%u "
                      "overruns=%lu i2c_errors=%lu rx_dropped=%lu tx_skipped=%lu\n",
                      online,idle,active,MAX_OCCUPANCY,HOST_OCCUPANCY,
                      noodlepipe::effective_reserve(online,RESERVED_WORKERS),
                      (unsigned long)completed_frames,(unsigned long)aborted_frames,
                      (unsigned long)seen,(unsigned long)valid,
                      unsigned(nn_bpm_result.bpm),unsigned(nn_bpm_result.valid),
                      nn_bpm_result.peaks,
                      unsigned(bpm_result.bpm),unsigned(bpm_result.valid),
                      (unsigned long)over,(unsigned long)err,(unsigned long)radio.dropped(),
                      (unsigned long)skipped_windows);
    }
    delay(1);
}
