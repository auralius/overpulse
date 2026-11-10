#pragma once
// Compile-only test shim; NOT real ESP-NOW behavior.
#include <PipelineProtocol.h>
namespace noodlepipe {
struct Message {Packet p;uint16_t bytes;uint8_t mac[6];};
class NowLink {
    uint8_t id_,mac_[6];
public:
    NowLink():id_(0),mac_{0}{}
    bool begin(uint8_t id){id_=id;return node_mac(id,mac_);}
    bool peer(const uint8_t[6]){return true;}
    bool send(const uint8_t[6],const Packet&,uint16_t=0){return true;}
    bool send_to(uint8_t id,const Packet& p,uint16_t n=0) {
        uint8_t mac[6];return node_mac(id,mac) && send(mac,p,n);
    }
    bool poll(Message&){return false;}
    uint32_t dropped()const{return 0;}
    const uint8_t* address()const{return mac_;}
};
}
