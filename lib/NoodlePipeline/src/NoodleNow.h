#pragma once
// Hardware-specific ESP-NOW adapter. Never used inside Noodle2 operator code.
#include <Arduino.h>
#include <PipelineProtocol.h>
namespace noodlepipe {
struct Message {
    Packet p;
    uint16_t bytes;
    uint8_t mac[6];
};
class NowLink {
public:
    NowLink();
    bool begin(uint8_t node_id);
    bool peer(const uint8_t mac[6]);
    bool send(const uint8_t mac[6], const Packet& p, uint16_t body_bytes=0);
    bool poll(Message& out);
    uint32_t dropped() const { return dropped_; }
    const uint8_t* address() const { return local_mac_; }
    uint8_t node_id() const { return node_id_; }
    void receive(const uint8_t* mac,const uint8_t* data,int length);
private:
    static const uint8_t QCAP=10;
    Message queue_[QCAP];
    volatile uint8_t head_, tail_;
    volatile uint32_t dropped_;
    uint8_t local_mac_[6];
    uint8_t node_id_;
};
}
