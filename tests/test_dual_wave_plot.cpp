// Exercise the actual host display functions with a recording TFT stub.
#include "../src/main_host.cpp"
#include <cassert>
#include <cstdio>
int main() {
    for(unsigned i=0;i<W;++i) {
        // Different AC amplitudes; both traces must be visible after auto-scale.
        const int saw=int(i%32)-16;
        ac_ir_wave[i]=static_cast<int16_t>(saw);
        ac_red_wave[i]=static_cast<int16_t>(saw*100);
        model_logits[i]=0;
        marks[i]=0;
    }
    marks[100]=1;
    draw_frame(true,1,850000);
    assert(tft.green_lines==255);
    assert(tft.red_lines==255);
    assert(tft.green_first_x==tft.red_first_x);
    assert(tft.green_last_x==tft.red_last_x);
    assert(tft.green_last_x==312);
    assert(tft.yellow_circles==1);
    assert(tft.green_min_y>=IR_WAVE_TOP && tft.green_max_y<=IR_WAVE_BOT);
    assert(tft.red_min_y>=RED_WAVE_TOP && tft.red_max_y<=RED_WAVE_BOT);
    assert(tft.green_max_y-tft.green_min_y>=40);
    assert(tft.red_max_y-tft.red_min_y>=40);
    std::puts("DUAL_RED_IR_WAVEFORM_DISPLAY_PASS");
}
