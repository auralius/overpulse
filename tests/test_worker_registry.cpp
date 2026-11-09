// Compile actual host into this translation unit so anonymous-namespace
// registry/scheduler functions can be exercised with a mocked radio.
#include "../src/main_host.cpp"
#include <cassert>
#include <cstdio>

int main() {
    using namespace noodlepipe;
    static_assert(NOODLE_MAX_WORKERS==9, "test expects 9 worker slots");
    const uint8_t selected[] = {1,2,3,4,5,6,7,8,9};
    for(unsigned i=0;i<sizeof(selected);++i) {
        const uint8_t id=selected[i];
        Message msg={};
        msg.p=packet(HELLO,id);
        msg.p.flags=1;
        msg.bytes=20;
        msg.mac[0]=0x02;
        msg.mac[5]=id;
        receive_radio(msg);
        assert(workers[id].online);
    }
    assert(connected()==9);
    assert(idle_workers()==9);
    assert(select_worker()==1);
    workers[1].busy=true;
    assert(select_worker()==2);
    assert(select_worker(2)==3);
    assert(idle_workers()==8);
    assert(host_can_admit(0,6,9,8,1));
    Message invalid={};
    invalid.p=packet(HELLO,10);
    invalid.p.flags=1;
    invalid.bytes=20;
    receive_radio(invalid); // must not index past the registry
    assert(connected()==9);
    assert(!worker_id_valid(0));
    assert(!worker_id_valid(10));
    assert(worker_id_valid(9));
    std::puts("NINE_WORKER_REGISTRY_PASS");
}
