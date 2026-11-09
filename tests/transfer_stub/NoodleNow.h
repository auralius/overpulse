#pragma once
#include <deque>
#include <PipelineProtocol.h>
namespace noodlepipe {
struct Message {Packet p;uint16_t bytes;uint8_t mac[6];};
class NowLink {
public:
    explicit NowLink(uint8_t id):data_sent_(0),ack_sent_(0),data_dropped_(0),ack_dropped_(0),
        id_(id),peer_(nullptr),drop_data5_(false),drop_data7_(false),
        drop_prepared_(false),drop_ack4_(false),drop_ack16_(false) {
        memset(mac_,0,6);mac_[5]=id;
    }
    void connect(NowLink* p){peer_=p;}
    bool peer(const uint8_t*){return true;}
    const uint8_t* address()const{return mac_;}
    void inject(bool loss){drop_data5_=drop_data7_=drop_prepared_=drop_ack4_=drop_ack16_=loss;}
    bool send(const uint8_t* dst,const Packet& p,uint16_t n=0) {
        if(!peer_ || !equal_mac(dst,peer_->mac_))return false;
        if(p.kind==DATA)++data_sent_;
        if(p.kind==ACK)++ack_sent_;
        bool drop=false;
        if(p.kind==DATA && p.seq==5 && drop_data5_){drop_data5_=false;drop=true;}
        if(p.kind==DATA && p.seq==7 && drop_data7_){drop_data7_=false;drop=true;}
        if(p.kind==PREPARED && drop_prepared_){drop_prepared_=false;drop=true;}
        if(p.kind==ACK && p.seq==4 && drop_ack4_){drop_ack4_=false;drop=true;}
        if(p.kind==ACK && p.seq==16 && drop_ack16_){drop_ack16_=false;drop=true;}
        if(drop) {
            if(p.kind==DATA)++data_dropped_;
            if(p.kind==ACK)++ack_dropped_;
            return true;
        }
        Message m={};m.p=p;m.bytes=20+n;
        memcpy(m.mac,mac_,6);peer_->q_.push_back(m);return true;
    }
    bool poll(Message& m){if(q_.empty())return false;m=q_.front();q_.pop_front();return true;}
    uint32_t data_sent_,ack_sent_,data_dropped_,ack_dropped_;
private:
    uint8_t id_,mac_[6];NowLink* peer_;
    bool drop_data5_,drop_data7_,drop_prepared_,drop_ack4_,drop_ack16_;
    std::deque<Message> q_;
};
}
