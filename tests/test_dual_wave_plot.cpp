// Exercise live host display using a recording TFT stub. No worker or
// completed inference frame must be required for Red/IR traces.
#include "../src/main_host.cpp"
#include <cassert>
#include <cstdio>
int main() {
    assert(completed_frames==0);
    assert(connected()==0);
    sensor_ready=true;
    samples_seen=W;
    continuous=W;
    for(unsigned i=0;i<W;++i) {
        const int saw=int(i%32)-16;
        ir_wave_ring[i]=static_cast<int16_t>(saw);
        red_wave_ring[i]=static_cast<int16_t>(saw*100);
    }
    draw_live_frame();
    assert(tft.green_lines==255);
    assert(tft.red_lines==255);
    assert(tft.green_first_x==tft.red_first_x);
    assert(tft.green_last_x==tft.red_last_x);
    assert(tft.green_last_x==312);
    assert(tft.yellow_circles==0); // no NN result yet
    assert(tft.green_min_y>=IR_WAVE_TOP && tft.green_max_y<=IR_WAVE_BOT);
    assert(tft.red_min_y>=RED_WAVE_TOP && tft.red_max_y<=RED_WAVE_BOT);
    assert(tft.green_max_y-tft.green_min_y>=40);
    assert(tft.red_max_y-tft.red_min_y>=40);
    // A recent inference result places overlay dots at their absolute
    // sample positions even while the live ring keeps advancing.
    have_nn_overlay=true;
    nn_overlay_epoch=sensor_epoch;
    nn_overlay_end=samples_seen;
    model_logits[100]=60;
    marks[100]=1;
    draw_live_frame();
    assert(tft.yellow_circles==1);
    const int first_dot_x=tft.yellow_last_x;
    assert(first_dot_x==wave_x(100));
    // Scroll forward by 10 samples. The marker must remain on the plot,
    // 10 pixels-worth of sample coordinates to the left.
    for(unsigned i=0;i<10;++i) {
        const uint32_t ix=samples_seen++;
        ir_wave_ring[ix%W]=0;
        red_wave_ring[ix%W]=0;
    }
    draw_live_frame();
    assert(tft.yellow_circles==2);
    assert(tft.yellow_last_x==wave_x(90));
    assert(tft.yellow_last_x<first_dot_x);
    // A signal discontinuity invalidates all old NN overlays.
    sync_nn_epoch(sensor_epoch+1);
    draw_live_frame();
    assert(tft.yellow_circles==2);
    // After reboot / reinitialization a partial waveform should still draw
    // (without fabricating a full 256-sample history).
    samples_seen=32;
    for(unsigned i=0;i<32;++i) {
        ir_wave_ring[i]=static_cast<int16_t>(i);
        red_wave_ring[i]=static_cast<int16_t>(2*i);
    }
    const unsigned green_before=tft.green_lines;
    const unsigned red_before=tft.red_lines;
    draw_live_frame();
    assert(tft.green_lines-green_before==31);
    assert(tft.red_lines-red_before==31);
    std::puts("LIVE_RED_IR_DISPLAY_WITHOUT_WORKERS_PASS");
}
