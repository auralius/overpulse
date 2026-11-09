#pragma once
// NoodleTransport v2: bounded 4-fragment selective-repeat ARQ.
// Per-fragment ACKs are replaced with a group bitmap ACK.
// All model data stay CHW int8; operator code and Packet layout are unchanged.
// V2 is deliberately NOT wire-compatible with V1: reflash host and workers.
#include <Arduino.h>
#include <NoodleNow.h>
#include "PipelineProtocol.h"
namespace noodlepipe {
#ifndef NOODLE_ESPNOW_PACING_MS
#define NOODLE_ESPNOW_PACING_MS 4
#endif
#ifndef NOODLE_TRANSPORT_WINDOW
#define NOODLE_TRANSPORT_WINDOW 4
#endif
static_assert(NOODLE_TRANSPORT_WINDOW>=1 && NOODLE_TRANSPORT_WINDOW<=8,
              "Transport window must be in 1..8");
static const uint16_t MAX_FRAGMENTS=(MAX_TENSOR+PAYLOAD-1)/PAYLOAD;
static_assert(MAX_FRAGMENTS<=32,"Receiver bitmap needs <=32 fragments");
static const uint32_t TRANSPORT_TIMEOUT_MS=140;
static const uint8_t TRANSPORT_RETRIES=15;
inline uint16_t frag_count(uint16_t size) { return (size+PAYLOAD-1)/PAYLOAD; }
inline uint16_t group_size(uint16_t n,uint16_t base) {
    return (n-base)<NOODLE_TRANSPORT_WINDOW ? (n-base) : NOODLE_TRANSPORT_WINDOW;
}
inline uint16_t group_bits(uint16_t n,uint16_t base) {
    return static_cast<uint16_t>((1u<<group_size(n,base))-1u);
}

class Sender {
public:
    enum State : uint8_t { IDLE, PREPARE_WAIT, DATA_WAIT, COMPLETE, FAILED };
    explicit Sender(uint8_t my_id): my_id_(my_id),state_(IDLE),frame_(0),stage_(0),len_(0),
       base_(0),acked_(0),tried_(0),last_prepare_(0),last_send_(0),ack_wait_(0),
       retry_round_(0),crc_(0),sent_packets_(0),retransmitted_(0),acks_(0),
       started_ms_(0),ended_ms_(0),ever_sent_(0) {memset(dest_,0,6);}
    bool start(NowLink& link,const uint8_t* dest,uint32_t frame,uint8_t stage,
               const int8_t* bytes,uint16_t n) {
        if(state_==PREPARE_WAIT || state_==DATA_WAIT || !bytes ||
           !tensor_input(stage) || n!=tensor_input(stage) || !link.peer(dest))return false;
        memcpy(dest_,dest,6);memcpy(content_,bytes,n);
        frame_=frame;stage_=stage;len_=n;base_=0;acked_=tried_=0;
        last_prepare_=last_send_=ack_wait_=0;retry_round_=0;
        crc_=tensor_crc32(content_,len_);state_=PREPARE_WAIT;
        sent_packets_=retransmitted_=acks_=0;started_ms_=ended_ms_=0;ever_sent_=0;
        return true;
    }
    bool handle(const Message& m) {
        if(m.p.frame!=frame_ || m.p.stage!=stage_ || !equal_mac(m.mac,dest_))return false;
        if(state_==PREPARE_WAIT && m.p.kind==PREPARED) {
            state_=DATA_WAIT;base_=0;acked_=tried_=0;
            last_send_=ack_wait_=0;retry_round_=0;
            return true;
        }
        if(state_==DATA_WAIT && m.p.kind==ACK && m.p.seq==base_) {
            ++acks_;
            const uint16_t total=frag_count(len_);
            const uint16_t mask=group_bits(total,base_);
            const uint16_t new_acked=static_cast<uint16_t>(acked_|(m.p.flags & mask));
            if(new_acked==acked_)return true; // Duplicate/stale ACK.
            acked_=new_acked;
            if(acked_==mask) {
                base_=static_cast<uint16_t>(base_+group_size(total,base_));
                acked_=tried_=0;ack_wait_=0;retry_round_=0;
                if(base_==total)state_=COMPLETE;
            } else {
                // Selectively retransmit only missing fragments.
                tried_=0;ack_wait_=0;retry_round_=0;
            }
            return true;
        }
        if((state_==PREPARE_WAIT || state_==DATA_WAIT) && m.p.kind==ERROR_PACKET) {
            state_=FAILED;return true;
        }
        return false;
    }
    void tick(NowLink& link,uint32_t ms) {
        if(state_!=PREPARE_WAIT && state_!=DATA_WAIT)return;
        if(!started_ms_)started_ms_=ms?ms:1;
        if(state_==PREPARE_WAIT) {
            if(last_prepare_ && ms-last_prepare_<TRANSPORT_TIMEOUT_MS)return;
            if(last_prepare_ && ++retry_round_>TRANSPORT_RETRIES){state_=FAILED;return;}
            Packet p=packet(PREPARE,my_id_,frame_,stage_);
            p.size=len_;p.crc=crc_;
            p.flags=NOODLE_TRANSPORT_CRC; // explicit wire-level mode agreement
            if(link.send(dest_,p))last_prepare_=ms?ms:1;
            return;
        }
        if(last_send_ && ms-last_send_<NOODLE_ESPNOW_PACING_MS)return;
        const uint16_t total=frag_count(len_);
        const uint16_t mask=group_bits(total,base_);
        const uint16_t missing=static_cast<uint16_t>(mask & ~acked_);
        const uint16_t to_send=static_cast<uint16_t>(missing & ~tried_);
        if(to_send) {
            uint8_t k=0;
            while(!(to_send&(1u<<k)))++k;
            const uint16_t seq=static_cast<uint16_t>(base_+k);
            const uint16_t offset=static_cast<uint16_t>(seq*PAYLOAD);
            const uint16_t remain=static_cast<uint16_t>(len_-offset);
            const uint16_t chunk=remain<PAYLOAD?remain:PAYLOAD;
            Packet p=packet(DATA,my_id_,frame_,stage_);
            p.seq=seq;p.size=chunk;
            memcpy(p.payload,content_+offset,chunk);
            if(link.send(dest_,p,chunk)) {
                const uint32_t bit=(1ul<<seq);
                if(ever_sent_ & bit)++retransmitted_;
                ever_sent_|=bit;
                ++sent_packets_;
                tried_|=static_cast<uint16_t>(1u<<k);
                last_send_=ms?ms:1;
                if((tried_ & missing)==missing)ack_wait_=ms?ms:1;
            }
            return;
        }
        if(ack_wait_ && ms-ack_wait_>=TRANSPORT_TIMEOUT_MS) {
            if(++retry_round_>TRANSPORT_RETRIES){state_=FAILED;return;}
            tried_=0;ack_wait_=0; // resend fragments that remain unacknowledged
        }
    }
    State state()const{return state_;}
    uint32_t frame()const{return frame_;}
    uint8_t stage()const{return stage_;}
    uint16_t size()const{return len_;}
    uint32_t sent_packets()const{return sent_packets_;}
    uint32_t retransmitted()const{return retransmitted_;}
    uint32_t ack_count()const{return acks_;}
    uint32_t transfer_ms(uint32_t now)const{return started_ms_ ? now-started_ms_ : 0;}
    void clear(){state_=IDLE;}
private:
    uint8_t my_id_,dest_[6];State state_;uint32_t frame_;uint8_t stage_;uint16_t len_,base_,acked_,tried_;
    uint32_t last_prepare_,last_send_,ack_wait_;uint8_t retry_round_;
    uint32_t crc_,sent_packets_,retransmitted_,acks_,started_ms_,ended_ms_,ever_sent_;
    int8_t content_[MAX_TENSOR];
};

class Receiver {
public:
    explicit Receiver(uint8_t id):my_id_(id),active_(false),complete_(false),
     prior_valid_(false),frame_(0),stage_(0),total_(0),crc_(0),received_(0),
     prior_frame_(0),prior_stage_(0),prior_base_(0),prior_mask_(0),
     chunks_(0),duplicates_(0),ack_packets_(0) {
        memset(src_,0,6);memset(prior_src_,0,6);
    }
    void handle(NowLink& link,const Message& m) {
        const Packet& p=m.p;
        if(p.kind==PREPARE) {
            if(p.stage>RESULT_STAGE || p.size!=tensor_input(p.stage))return;
            // Reject peers built with the opposite checksum mode immediately.
            // Do not silently interpret a missing CRC as a valid checksum.
            if(p.flags!=NOODLE_TRANSPORT_CRC) {
                Packet e=packet(ERROR_PACKET,my_id_,p.frame,p.stage);
                e.flags=3; // transport CRC configuration mismatch
                link.send(m.mac,e);
                return;
            }
            if(active_) {
                if(p.frame==frame_ && p.stage==stage_ && equal_mac(m.mac,src_))
                    link.send(src_,packet(PREPARED,my_id_,frame_,stage_));
                else {
                    Packet e=packet(ERROR_PACKET,my_id_,p.frame,p.stage);
                    e.flags=1;link.send(m.mac,e);
                }
                return;
            }
            active_=true;complete_=false;frame_=p.frame;stage_=p.stage;
            total_=p.size;crc_=p.crc;received_=0;
            chunks_=duplicates_=ack_packets_=0;
            memcpy(src_,m.mac,6);
            link.send(src_,packet(PREPARED,my_id_,frame_,stage_));
            return;
        }
        if(p.kind!=DATA)return;
        if(!active_) {
            // A receiver may have released its tensor after the final ACK was lost.
            if(prior_valid_ && p.frame==prior_frame_ && p.stage==prior_stage_ &&
               equal_mac(m.mac,prior_src_) && p.seq>=prior_base_ &&
               p.seq<prior_base_+NOODLE_TRANSPORT_WINDOW) {
                Packet ack=packet(ACK,my_id_,p.frame,p.stage);
                ack.seq=prior_base_;ack.flags=prior_mask_;
                link.send(m.mac,ack);
            }
            return;
        }
        if(p.frame!=frame_ || p.stage!=stage_ || !equal_mac(m.mac,src_))return;
        const uint16_t total_frag=frag_count(total_);
        if(p.seq>=total_frag)return;
        const uint16_t offset=static_cast<uint16_t>(p.seq*PAYLOAD);
        const uint16_t expect=static_cast<uint16_t>(total_-offset)<PAYLOAD ?
            static_cast<uint16_t>(total_-offset) : PAYLOAD;
        if(p.size!=expect)return;
        const uint32_t bit=1ul<<p.seq;
        if(received_ & bit)++duplicates_;
        else {
            memcpy(buffer_+offset,p.payload,p.size);
            received_|=bit;++chunks_;
        }
        const uint16_t base=static_cast<uint16_t>(
            (p.seq/NOODLE_TRANSPORT_WINDOW)*NOODLE_TRANSPORT_WINDOW);
        const uint16_t mask=group_bits(total_frag,base);
        const uint16_t have=static_cast<uint16_t>((received_>>base)&mask);
        const bool all=(received_==((1ul<<total_frag)-1ul));
        if(all && !complete_) {
#if NOODLE_TRANSPORT_CRC
            if(crc32(buffer_,total_)!=crc_) {
                Packet e=packet(ERROR_PACKET,my_id_,frame_,stage_);
                e.flags=2;link.send(src_,e);active_=false;return;
            }
#endif
            complete_=true;
        }
        // ACK once group end arrives (possibly partial), or once all its pieces arrive.
        // Duplicates of complete-group fragments also re-ACK lost ACKs.
        if(p.seq==base+group_size(total_frag,base)-1 || have==mask) {
            Packet ack=packet(ACK,my_id_,frame_,stage_);
            ack.seq=base;ack.flags=have;
            if(link.send(src_,ack))++ack_packets_;
        }
    }
    bool complete()const{return active_ && complete_;}
    uint32_t frame()const{return frame_;}
    uint8_t stage()const{return stage_;}
    uint16_t size()const{return total_;}
    const int8_t* data()const{return buffer_;}
    uint32_t chunks()const{return chunks_;}
    uint32_t duplicates()const{return duplicates_;}
    uint32_t ack_packets()const{return ack_packets_;}
    void release() {
        if(!complete())return;
        const uint16_t n=frag_count(total_);
        prior_base_=static_cast<uint16_t>(((n-1)/NOODLE_TRANSPORT_WINDOW)*NOODLE_TRANSPORT_WINDOW);
        prior_mask_=group_bits(n,prior_base_);
        prior_frame_=frame_;prior_stage_=stage_;
        memcpy(prior_src_,src_,6);prior_valid_=true;
        active_=complete_=false;
    }
    void reset(){active_=complete_=false;}
    bool busy()const{return active_;}
private:
    uint8_t my_id_;bool active_,complete_,prior_valid_;uint32_t frame_;
    uint8_t stage_;uint16_t total_;uint32_t crc_,received_;
    uint8_t src_[6];int8_t buffer_[MAX_TENSOR];
    uint32_t prior_frame_;uint8_t prior_stage_;uint16_t prior_base_,prior_mask_;
    uint8_t prior_src_[6];uint32_t chunks_,duplicates_,ack_packets_;
};
} // namespace noodlepipe
