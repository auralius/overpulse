#include <PipelineTransfer.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
static void run(uint8_t stage,bool loss,bool release_early,bool tail_loss=false) {
    noodlepipe::NowLink a(0),b(2);a.connect(&b);b.connect(&a);
    a.inject(loss);b.inject(loss);
    if(tail_loss){a.inject_tail_loss();}
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
    if(!loss && !tail_loss) {
        const unsigned expected=(noodlepipe::frag_count(n)+NOODLE_ACK_INTERVAL-1)/NOODLE_ACK_INTERVAL;
        assert(b.ack_sent_==expected);
        assert(sender.retransmitted()==0);
    }
    if((loss || tail_loss) && stage==2)assert(sender.retransmitted()>0);
    if(tail_loss) {
        assert(a.ack_request_sent_ >= 2); // first ACK_REQUEST intentionally lost
        assert(a.ack_request_dropped_==1);
    }
    std::printf("PASS interval=%d stage=%u bytes=%u loss=%u tail=%u early_release=%u "
                "data=%lu ack=%lu probes=%lu resent=%lu\n",
                NOODLE_ACK_INTERVAL,stage,n,loss,tail_loss,release_early,
                (unsigned long)a.data_sent_,(unsigned long)b.ack_sent_,
                (unsigned long)a.ack_request_sent_,
                (unsigned long)sender.retransmitted());
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
    m.p.flags = noodlepipe::TRANSPORT_CONFIG_FLAGS ^ 1u;
    receiver.handle(b,m);
    assert(!receiver.busy());
    assert(a.poll(m));
    assert(m.p.kind==noodlepipe::ERROR_PACKET && m.p.flags==3);
    sender.handle(m);
    assert(sender.state()==noodlepipe::Sender::FAILED);
    std::printf("PASS CRC-mode mismatch rejected mode=%d\n", NOODLE_TRANSPORT_CRC);
}
static void mismatched_ack_interval_is_rejected() {
    noodlepipe::NowLink a(0),b(2);a.connect(&b);b.connect(&a);
    noodlepipe::Sender sender(0);noodlepipe::Receiver receiver(2);
    int8_t input[512]={};
    assert(sender.start(a,b.address(),72,0,input,sizeof(input)));
    sender.tick(a,1);
    noodlepipe::Message m;
    assert(b.poll(m) && m.p.kind==noodlepipe::PREPARE);
    m.p.flags ^= 2u; // Preserve CRC mode, mismatch only the interval
    receiver.handle(b,m);
    assert(!receiver.busy());
    assert(a.poll(m));
    assert(m.p.kind==noodlepipe::ERROR_PACKET && m.p.flags==4);
    sender.handle(m);
    assert(sender.state()==noodlepipe::Sender::FAILED);
    std::puts("PASS ACK interval mismatch rejected");
}
static void corruption_handling() {
    noodlepipe::NowLink a(0),b(2);a.connect(&b);b.connect(&a);
    noodlepipe::Receiver receiver(2);
    int8_t original[512];
    for(unsigned i=0;i<512;++i)original[i]=static_cast<int8_t>(i*17+3);
    noodlepipe::Packet prepare=noodlepipe::packet(noodlepipe::PREPARE,0,84,0);
    prepare.size=sizeof(original);
    prepare.flags=noodlepipe::TRANSPORT_CONFIG_FLAGS;
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
static void previous_frame_can_be_acknowledged_while_next_is_active() {
    noodlepipe::NowLink a(0),b(2);a.connect(&b);b.connect(&a);
    noodlepipe::Receiver receiver(2);
    noodlepipe::Message msg;
    noodlepipe::Packet prep=noodlepipe::packet(noodlepipe::PREPARE,0,201,5);
    prep.size=256;prep.flags=noodlepipe::TRANSPORT_CONFIG_FLAGS;
    int8_t payload[256]={};
    prep.crc=noodlepipe::tensor_crc32(payload,sizeof(payload));
    assert(a.send(b.address(),prep));
    assert(b.poll(msg));receiver.handle(b,msg);
    for(unsigned seq=0;seq<2;++seq) {
        noodlepipe::Packet d=noodlepipe::packet(noodlepipe::DATA,0,201,5);
        d.seq=seq;d.size=(seq==0)?230:26;
        assert(a.send(b.address(),d,d.size));
        assert(b.poll(msg));receiver.handle(b,msg);
    }
    assert(receiver.complete());
    receiver.release();
    noodlepipe::Packet next=noodlepipe::packet(noodlepipe::PREPARE,0,202,0);
    next.size=512;next.flags=noodlepipe::TRANSPORT_CONFIG_FLAGS;
    assert(a.send(b.address(),next));
    assert(b.poll(msg));receiver.handle(b,msg);
    assert(receiver.busy()); // receiver has admitted the next frame
    noodlepipe::Packet query=noodlepipe::packet(noodlepipe::ACK_REQUEST,0,201,5);
    query.seq=0;
    assert(a.send(b.address(),query));
    assert(b.poll(msg));receiver.handle(b,msg);
    bool old_ack=false;
    while(a.poll(msg))if(msg.p.kind==noodlepipe::ACK && msg.p.frame==201 &&
                         msg.p.crc==3u)old_ack=true;
    assert(old_ack);
    std::puts("PASS prior-frame ACK replay while next frame active");
}

int main(){
    previous_frame_can_be_acknowledged_while_next_is_active();
    mismatched_mode_is_rejected();
    mismatched_ack_interval_is_rejected();
    corruption_handling();
    run(5,false,false); // final tensor
    run(0,false,false);
    run(1,false,false);
    run(2,false,false);
    run(2,true,false); // damaged/lost data, ACK, PREPARED
    run(2,true,true);  // final ACK lost after receiver releases
    run(2,false,true,true);  // last DATA + first ACK_REQUEST lost
    run(5,true,true);  // even short final tensor can be acknowledged
    return 0;
}
