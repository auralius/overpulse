#pragma once
// Network protocol v1: fixed-size little-endian headers; ESP-NOW v1 250-byte limit.
// No hidden activation requantization: all payloads are raw CHW int8 bytes.
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// Total worker capacity (IDs 1..NOODLE_MAX_WORKERS), unrelated to graph nodes.
// Each ESP-NOW device may need the broadcast peer and all other devices.
// Default preserves the original five-worker layout; scale up in platformio.ini.
#ifndef NOODLE_MAX_WORKERS
#define NOODLE_MAX_WORKERS 5
#endif
#if NOODLE_MAX_WORKERS < 1 || NOODLE_MAX_WORKERS > 19
#error "NOODLE_MAX_WORKERS must be 1..19 (ESP-NOW peer budget)"
#endif

// Whole-tensor software CRC32, independent of Wi-Fi hardware FCS.
// The host and every worker MUST use the same value (0 or 1).
#ifndef NOODLE_TRANSPORT_CRC
#define NOODLE_TRANSPORT_CRC 1
#endif
static_assert(NOODLE_TRANSPORT_CRC==0 || NOODLE_TRANSPORT_CRC==1,
              "NOODLE_TRANSPORT_CRC must be 0 or 1");
namespace noodlepipe {
static const uint16_t MAGIC = 0x324E;
// Version 3: deterministic STA MAC and one-byte destination worker ID in FORWARD.
// Deliberately not wire-compatible with the dynamic-MAC version 2.
static const uint8_t VERSION = 3;
static const uint8_t HOST_ID = 0;
static const uint8_t MAX_WORKERS = NOODLE_MAX_WORKERS;
inline bool worker_id_valid(uint8_t id) { return id>=1 && id<=MAX_WORKERS; }
static const uint8_t RESULT_STAGE = 5;
static const uint8_t NUM_STAGES = 5;
static const uint16_t MAX_TENSOR = 4096;
static const uint16_t MAX_PACKET = 250;
static const uint16_t PAYLOAD = 230;
static const uint8_t CHANNEL = 6;
static const uint8_t BROADCAST[6] = {255,255,255,255,255,255};
// Private, locally administered unicast prefix ('NDL'), unique per node ID.
// Host ID 0 => 02:4E:44:4C:00:00; worker W9 => 02:4E:44:4C:00:09.
// These are assigned to the actual Wi-Fi station interface by NowLink::begin().
inline bool node_mac(uint8_t id, uint8_t out[6]) {
    if (!out || (id != HOST_ID && !worker_id_valid(id))) return false;
    out[0]=0x02; out[1]=0x4E; out[2]=0x44;
    out[3]=0x4C; out[4]=0x00; out[5]=id;
    return true;
}
inline bool node_mac_matches(uint8_t id, const uint8_t received[6]) {
    uint8_t expected[6];
    return received && node_mac(id,expected) && memcmp(received,expected,6)==0;
}
enum Kind : uint8_t { HELLO=1, PREPARE=2, PREPARED=3, DATA=4, ACK=5,
                     DONE=6, FORWARD=7, RELEASE=8, ERROR_PACKET=9, RUN_LOCAL=10 };
#pragma pack(push,1)
struct Packet {
    uint16_t magic; uint8_t version; uint8_t kind;
    uint32_t frame; uint16_t seq; uint16_t size;
    uint8_t stage; uint8_t sender; uint16_t flags; uint32_t crc;
    uint8_t payload[PAYLOAD];
};
#pragma pack(pop)
static_assert(offsetof(Packet,payload)==20,"packet header is 20 bytes");
static_assert(sizeof(Packet)==MAX_PACKET,"ESP-NOW v1 payload bound");
inline uint16_t tensor_input(uint8_t stage) {
    static const uint16_t sizes[6]={512,2048,4096,4096,4096,256};
    return stage<=RESULT_STAGE ? sizes[stage] : 0;
}
inline uint16_t tensor_output(uint8_t stage) {
    static const uint16_t sizes[5]={2048,4096,4096,4096,256};
    return stage<NUM_STAGES ? sizes[stage] : 0;
}
inline uint32_t crc32(const void* data, uint16_t n) {
    const uint8_t* p=static_cast<const uint8_t*>(data);
    uint32_t crc=0xFFFFFFFFu;
    for(uint16_t i=0;i<n;++i) {
        crc^=p[i];
        for(uint8_t bit=0;bit<8;++bit) crc=(crc>>1) ^ ((crc&1u) ? 0xEDB88320u : 0u);
    }
    return ~crc;
}
// No checksum calculation is compiled in for normal tensor transfers when OFF.
inline uint32_t tensor_crc32(const void* data,uint16_t n) {
#if NOODLE_TRANSPORT_CRC
    return crc32(data,n);
#else
    (void)data;
    (void)n;
    return 0;
#endif
}
inline Packet packet(uint8_t kind,uint8_t sender,uint32_t frame=0,uint8_t stage=0) {
    Packet p={}; p.magic=MAGIC;p.version=VERSION;p.kind=kind;p.sender=sender;
    p.frame=frame;p.stage=stage;return p;
}
inline bool valid(const Packet& p,uint16_t length) {
    return length>=20 && length<=MAX_PACKET && p.magic==MAGIC && p.version==VERSION &&
           p.size<=MAX_TENSOR && (p.kind!=DATA || (length==20+p.size && p.size>0 && p.size<=PAYLOAD));
}
inline bool equal_mac(const uint8_t* a,const uint8_t* b) {
    return memcmp(a,b,6)==0;
}
}
