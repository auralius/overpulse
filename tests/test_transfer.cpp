#include <PipelineTransfer.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
static void run(uint8_t stage,bool loss,bool release_early) {
    noodlepipe::NowLink a(0),b(2);a.connect(&b);b.connect(&a);
    a.inject(loss);b.inject(loss);
    noodlepipe::Sender sender(0);noodlepipe::Receiver receiver(2);
    const uint16_t n=noodlepipe::tensor_input(stage);
    int8_t input[4096];for(int i=0;i<4096;++i)input[i]=static_cast<int8_t>(i*13+9);
    assert(sender.start(a,b.address(),29,stage,input,n));
    bool done=false;
    for(uint32_t t=1;t<25000 && sender.state()!=noodlepipe::Sender::COMPLETE;++t){
        sender.tick(a,t);
        noodlepipe::Message m;
        while(b.poll(m))receiver.handle(b,m);
        if(receiver.complete()) {
            assert(receiver.size()==n);
            assert(memcmp(receiver.data(),input,n)==0);
            done=true;
            if(release_early)receiver.release();
        }
        while(a.poll(m))sender.handle(m);
    }
    assert(sender.state()==noodlepipe::Sender::COMPLETE);
    assert(done);
    if(!loss) {
        const unsigned expected=(noodlepipe::frag_count(n)+NOODLE_TRANSPORT_WINDOW-1)/NOODLE_TRANSPORT_WINDOW;
        assert(b.ack_sent_==expected);
        assert(sender.retransmitted()==0);
    }
    if(loss && stage==2)assert(sender.retransmitted()>0);
    std::printf("PASS stage=%u bytes=%u injected_loss=%u early_release=%u data=%lu ack=%lu resent=%lu\n",
                stage,n,loss,release_early,(unsigned long)a.data_sent_,
                (unsigned long)b.ack_sent_,(unsigned long)sender.retransmitted());
}
static void mismatched_mode_is_rejected() {
    noodlepipe::NowLink a(0),b(2);a.connect(&b);b.connect(&a);
    noodlepipe::Sender sender(0);noodlepipe::Receiver receiver(2);
    int8_t input[512]={};
    assert(sender.start(a,b.address(),71,0,input,sizeof(input)));
    sender.tick(a,1);
    noodlepipe::Message m;
    assert(b.poll(m));
    assert(m.p.kind==noodlepipe::PREPARE);
    // Simulate a sender built with the opposite CRC configuration.
    m.p.flags = NOODLE_TRANSPORT_CRC ? 0 : 1;
    receiver.handle(b,m);
    assert(!receiver.busy());
    assert(a.poll(m));
    assert(m.p.kind==noodlepipe::ERROR_PACKET && m.p.flags==3);
    sender.handle(m);
    assert(sender.state()==noodlepipe::Sender::FAILED);
    std::printf("PASS CRC-mode mismatch rejected mode=%d\n", NOODLE_TRANSPORT_CRC);
}
static void corruption_handling() {
    noodlepipe::NowLink a(0),b(2);a.connect(&b);b.connect(&a);
    noodlepipe::Receiver receiver(2);
    int8_t original[512];
    for(unsigned i=0;i<512;++i)original[i]=static_cast<int8_t>(i*17+3);
    noodlepipe::Packet prepare=noodlepipe::packet(noodlepipe::PREPARE,0,84,0);
    prepare.size=sizeof(original);
    prepare.flags=NOODLE_TRANSPORT_CRC;
    prepare.crc=noodlepipe::tensor_crc32(original,sizeof(original));
    assert(a.send(b.address(),prepare));
    noodlepipe::Message msg;
    assert(b.poll(msg));receiver.handle(b,msg);
    for(unsigned seq=0;seq<noodlepipe::frag_count(sizeof(original));++seq) {
        const unsigned off=seq*noodlepipe::PAYLOAD;
        const unsigned n=(sizeof(original)-off)<noodlepipe::PAYLOAD ?
            sizeof(original)-off : noodlepipe::PAYLOAD;
        noodlepipe::Packet fragment=noodlepipe::packet(noodlepipe::DATA,0,84,0);
        fragment.seq=seq;fragment.size=n;
        memcpy(fragment.payload,original+off,n);
        if(seq==1)fragment.payload[0]^=0x40; // emulate corruption after Wi-Fi FCS check
        assert(a.send(b.address(),fragment,n));
        assert(b.poll(msg));receiver.handle(b,msg);
    }
#if NOODLE_TRANSPORT_CRC
    assert(!receiver.complete());
    bool rejected=false;
    while(a.poll(msg)) if(msg.p.kind==noodlepipe::ERROR_PACKET && msg.p.flags==2)rejected=true;
    assert(rejected);
    std::puts("PASS corrupted tensor rejected with CRC ON");
#else
    assert(receiver.complete());
    assert(memcmp(receiver.data(),original,sizeof(original))!=0);
    std::puts("PASS corrupted tensor accepted with CRC OFF (expected risk)");
#endif
}
int main(){
    mismatched_mode_is_rejected();
    corruption_handling();
    run(5,false,false); // final tensor
    run(0,false,false);
    run(1,false,false);
    run(2,false,false);
    run(2,true,false); // damaged/lost data, ACK, PREPARED
    run(2,true,true);  // final ACK lost after receiver releases
    run(5,true,true);  // even short final tensor can be acknowledged
    return 0;
}
