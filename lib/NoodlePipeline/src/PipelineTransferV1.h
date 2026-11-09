#pragma once
// Transport-level stop-and-wait fragmentation with whole-tensor CRC32.
// Sender never treats ESP-NOW MAC callback as an application ACK.
#include <Arduino.h>
#include <NoodleNow.h>
#include "PipelineProtocol.h"
namespace noodlepipe {
class Sender {
public:
    enum State : uint8_t { IDLE, PREPARE_WAIT, DATA_WAIT, COMPLETE, FAILED };
    Sender(uint8_t my_id) : my_id_(my_id), state_(IDLE), frame_(0), stage_(0),len_(0),
        seq_(0),at_(0),last_(0),retries_(0),crc_(0) { memset(dest_,0,6); }
    bool start(NowLink& link,const uint8_t* dest,uint32_t frame,uint8_t stage,
               const int8_t* bytes,uint16_t n) {
        if(state_==PREPARE_WAIT || state_==DATA_WAIT || !bytes ||
           n!=tensor_input(stage) || !link.peer(dest))return false;
        memcpy(dest_,dest,6);memcpy(content_,bytes,n);
        frame_=frame;stage_=stage;len_=n;seq_=0;at_=0;last_=0;retries_=0;
        crc_=crc32(content_,len_);state_=PREPARE_WAIT;
        return true;
    }
    bool handle(const Message& m) {
        if(m.p.frame!=frame_ || m.p.stage!=stage_ || !equal_mac(m.mac,dest_))return false;
        if(state_==PREPARE_WAIT && m.p.kind==PREPARED) {
            state_=DATA_WAIT;seq_=0;at_=0;last_=0;retries_=0;return true;
        }
        if(state_==DATA_WAIT && m.p.kind==ACK && m.p.seq==seq_) {
            const uint16_t remain=static_cast<uint16_t>(len_-at_);
            at_=static_cast<uint16_t>(at_+(remain<PAYLOAD?remain:PAYLOAD));
            ++seq_;last_=0;retries_=0;
            if(at_==len_)state_=COMPLETE;
            return true;
        }
        if((state_==PREPARE_WAIT || state_==DATA_WAIT) && m.p.kind==ERROR_PACKET) {
            // Receiver is busy or CRC rejected; hold the previous sender output.
            // Full frame is dropped by the host watchdog rather than silent retry forever.
            state_=FAILED;return true;
        }
        return false;
    }
    void tick(NowLink& link,uint32_t ms) {
        if(state_!=PREPARE_WAIT && state_!=DATA_WAIT)return;
        if(last_ && ms-last_<140)return;
        if(last_ && ++retries_>15) {state_=FAILED;return;}
        Packet p=packet(state_==PREPARE_WAIT?PREPARE:DATA,my_id_,frame_,stage_);
        uint16_t payload=0;
        if(state_==PREPARE_WAIT) {p.size=len_;p.crc=crc_;}
        else {
            const uint16_t remain=static_cast<uint16_t>(len_-at_);
            payload=remain<PAYLOAD?remain:PAYLOAD;
            p.seq=seq_;p.size=payload;
            memcpy(p.payload,content_+at_,payload);
        }
        if(link.send(dest_,p,payload))last_=ms?ms:1;
    }
    State state()const{return state_;}
    uint32_t frame()const{return frame_;}
    uint8_t stage()const{return stage_;}
    uint16_t size()const{return len_;}
    void clear(){state_=IDLE;}
private:
    uint8_t my_id_,dest_[6]; State state_; uint32_t frame_;
    uint8_t stage_;uint16_t len_,seq_,at_;uint32_t last_;
    uint8_t retries_;uint32_t crc_;
    int8_t content_[MAX_TENSOR];
};
class Receiver {
public:
    Receiver(uint8_t id):my_id_(id),active_(false),complete_(false),frame_(0),
      stage_(0),total_(0),pos_(0),seq_(0),crc_(0),prior_frame_(0),prior_stage_(0),
      prior_seq_(0) {memset(src_,0,6);memset(prior_src_,0,6);}
    void handle(NowLink& link,const Message& m) {
        const Packet& p=m.p;
        if(p.kind==PREPARE) {
            if(!tensor_input(p.stage) || p.size!=tensor_input(p.stage))return;
            if(active_) {
                if(p.frame==frame_ && p.stage==stage_ && equal_mac(m.mac,src_)) {
                    link.send(m.mac,packet(PREPARED,my_id_,frame_,stage_));
                } else {
                    Packet e=packet(ERROR_PACKET,my_id_,p.frame,p.stage);e.flags=1;link.send(m.mac,e);
                }
                return;
            }
            active_=true;complete_=false;frame_=p.frame;stage_=p.stage;
            total_=p.size;pos_=0;seq_=0;crc_=p.crc;
            memcpy(src_,m.mac,6);
            link.send(src_,packet(PREPARED,my_id_,frame_,stage_));
            return;
        }
        if(p.kind!=DATA)return;
        if(!active_) {
            // ACK the previous final fragment if its ACK was lost.
            if(p.frame==prior_frame_ && p.stage==prior_stage_ && p.seq==prior_seq_ &&
                equal_mac(m.mac,prior_src_)) {
                Packet ack=packet(ACK,my_id_,p.frame,p.stage);ack.seq=p.seq;link.send(m.mac,ack);
            }
            return;
        }
        if(p.frame!=frame_ || p.stage!=stage_ || !equal_mac(m.mac,src_))return;
        if(seq_ && p.seq==seq_-1) {
            Packet ack=packet(ACK,my_id_,frame_,stage_);ack.seq=p.seq;link.send(src_,ack);return;
        }
        if(complete_ || p.seq!=seq_ || p.size>total_-pos_)return;
        memcpy(buffer_+pos_,p.payload,p.size);pos_=static_cast<uint16_t>(pos_+p.size);
        if(pos_==total_) {
            if(crc32(buffer_,total_)!=crc_) {
                Packet e=packet(ERROR_PACKET,my_id_,frame_,stage_);e.flags=2;link.send(src_,e);
                active_=false;return;
            }
            complete_=true;
        }
        Packet ack=packet(ACK,my_id_,frame_,stage_);ack.seq=seq_;
        link.send(src_,ack);
        ++seq_;
    }
    bool complete()const{return active_&&complete_;}
    uint32_t frame()const{return frame_;}
    uint8_t stage()const{return stage_;}
    uint16_t size()const{return total_;}
    const int8_t* data()const{return buffer_;}
    void release() {
        if(!complete())return;
        prior_frame_=frame_;prior_stage_=stage_;prior_seq_=seq_-1;memcpy(prior_src_,src_,6);
        active_=false;complete_=false;
    }
    void reset() {active_=false;complete_=false;}
    bool busy()const{return active_;}
private:
    uint8_t my_id_;bool active_,complete_;uint32_t frame_;uint8_t stage_;
    uint16_t total_,pos_,seq_;uint32_t crc_;
    uint8_t src_[6];int8_t buffer_[MAX_TENSOR];
    uint32_t prior_frame_;uint8_t prior_stage_;uint16_t prior_seq_;uint8_t prior_src_[6];
};
}
