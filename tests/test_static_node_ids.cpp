#include <PipelineProtocol.h>
#include <cassert>
#include <cstdio>
#include <cstring>
int main() {
    using namespace noodlepipe;
    uint8_t host[6]={};
    assert(node_mac(HOST_ID,host));
    const uint8_t prefix[5]={0x02,0x4E,0x44,0x4C,0x00};
    assert(memcmp(host,prefix,5)==0 && host[5]==0);
    assert((host[0]&0x03)==0x02); // locally administered + unicast
    for(uint8_t id=1;id<=MAX_WORKERS;++id) {
        uint8_t mac[6]={};
        assert(node_mac(id,mac));
        assert(memcmp(mac,prefix,5)==0 && mac[5]==id);
        assert(node_mac_matches(id,mac));
        assert(!node_mac_matches(HOST_ID,mac));
        uint8_t other[6];
        assert(node_mac(id==1?HOST_ID:1,other));
        if(id!=1)assert(!node_mac_matches(id,other));
    }
    uint8_t invalid[6]={};
    assert(!node_mac(MAX_WORKERS+1,invalid));
    assert(!node_mac(255,invalid));
    assert(!node_mac(0,nullptr));
    Packet cmd=packet(FORWARD,HOST_ID,71,3);
    cmd.payload[0]=9;
    assert(valid(cmd,21));
    assert(sizeof(Packet)==250 && offsetof(Packet,payload)==20);
    std::puts("STATIC_MAC_ID_PROTOCOL_PASS");
}
