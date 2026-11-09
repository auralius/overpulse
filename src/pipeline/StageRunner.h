#pragma once
// Single-stage Noodle2 execution with one planned SRAM ping-pong arena.
// All workers bind the SAME complete six-layer QAT export from PROGMEM.
#include <Noodle2.h>
#include "peaknet_ppg_model.h"
#include <PipelineProtocol.h>
namespace noodlepipe {
class StageRunner {
public:
    StageRunner() : length_(0),stage_(0) {}
    bool begin() {
        if(model_.bind_flash()!=noodle2::OK)return false;
        if(a_.bind_ram(a_bytes_,sizeof(a_bytes_))!=noodle2::OK)return false;
        if(b_.bind_ram(b_bytes_,sizeof(b_bytes_))!=noodle2::OK)return false;
        return true;
    }
    bool run(uint8_t stage,const int8_t* data,uint16_t len) {
        if(stage>=NUM_STAGES || !data || len!=tensor_input(stage))return false;
        memcpy(a_bytes_,data,len);
        noodle2::NoodleTensor a,b;
        a.bind(a_);b.bind(b_);
        uint16_t c=stage==0?2:stage==1?8:16;
        if(a.resize(c,1,256)!=noodle2::OK)return false;
        uint16_t next=stage==0?8:16;
        if(b.resize(next,1,256)!=noodle2::OK)return false;
        noodle2::Status st=noodle2::BAD_PARAMS;
        switch(stage) {
        case 0:st=noodle2::conv2d(a,b,model_.conv01,noodle2::Conv2D::same(1,9),noodle2::RELU);break;
        case 1:st=noodle2::conv2d(a,b,model_.conv02,noodle2::Conv2D::same(1,7),noodle2::RELU);break;
        case 2:st=noodle2::conv2d(a,b,model_.conv03,noodle2::Conv2D::same(1,7),noodle2::RELU);break;
        case 3:st=noodle2::conv2d(a,b,model_.conv04,noodle2::Conv2D::same(1,7),noodle2::RELU);break;
        case 4:
            st=noodle2::conv2d(a,b,model_.conv05,noodle2::Conv2D::same(1,1),noodle2::RELU);
            if(st!=noodle2::OK)return false;
            // Last conv writes into the now-dead original input slot.
            if(a.resize(1,1,256)!=noodle2::OK)return false;
            st=noodle2::conv2d(b,a,model_.conv06,noodle2::Conv2D::same(1,1),noodle2::LOGITS);
            break;
        }
        if(st!=noodle2::OK)return false;
        length_=tensor_output(stage);stage_=stage;
        return true;
    }
    const int8_t* output()const {return stage_==4 ? a_bytes_:b_bytes_;}
    uint16_t output_bytes()const{return length_;}
    uint8_t current_stage()const{return stage_;}
private:
    peaknet_ppg::Model model_;
    noodle2::NoodleBuffer a_,b_;
    int8_t a_bytes_[MAX_TENSOR],b_bytes_[MAX_TENSOR];
    uint16_t length_;uint8_t stage_;
};
}
