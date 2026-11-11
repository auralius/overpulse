#pragma once
// Compiled once through main_workerN.cpp; no worker-specific kernel code.
#include <Arduino.h>
#include <NoodleNow.h>
#include <PipelineProtocol.h>
#include <PipelineTransfer.h>
#include "StageRunner.h"
// Lightweight, optional activation sparsity diagnostics (ESP8266 workers only).
// Keep disabled for latency/throughput comparisons.
#ifndef NOODLE_WORKER_ACTIVATION_STATS
#define NOODLE_WORKER_ACTIVATION_STATS 0
#endif
#ifndef PIPE_WORKER_ID
#error "PIPE_WORKER_ID must be set by the PlatformIO worker environment"
#endif
#if PIPE_WORKER_ID < 1 || PIPE_WORKER_ID > NOODLE_MAX_WORKERS
#error "PIPE_WORKER_ID must be in 1..NOODLE_MAX_WORKERS"
#endif
namespace {
noodlepipe::NowLink radio;
noodlepipe::Sender tx(PIPE_WORKER_ID);
noodlepipe::Receiver rx(PIPE_WORKER_ID);
noodlepipe::StageRunner runner;
enum WorkerMode { FREE, COMPUTE, OUTPUT_READY, SENDING };
WorkerMode mode=FREE;
uint32_t active_frame=0;
uint8_t completed_stage=0;
uint32_t last_done=0,last_hello=0,last_compute_us=0;
uint32_t errors=0,finished=0;

void control(uint8_t kind,uint32_t frame,uint8_t stage,uint16_t size=0,uint32_t crc=0) {
    noodlepipe::Packet p=noodlepipe::packet(kind,PIPE_WORKER_ID,frame,stage);
    p.size=size;p.crc=crc;
    radio.send_to(noodlepipe::HOST_ID,p);
}
void advertise() {
    noodlepipe::Packet p=noodlepipe::packet(noodlepipe::HELLO,PIPE_WORKER_ID);
    p.flags=mode==FREE?1:0;
    radio.send(noodlepipe::BROADCAST,p);
}
#if NOODLE_WORKER_ACTIVATION_STATS
void print_activation_stats(uint32_t frame,uint8_t stage) {
    const int8_t* values=runner.output();
    const uint16_t length=runner.output_bytes();
    if(!values || !length)return;

    uint16_t zeros=0;
    int8_t minimum=127,maximum=-128;
    for(uint16_t i=0;i<length;++i) {
        const int8_t value=values[i];
        if(value==0)++zeros;
        if(value<minimum)minimum=value;
        if(value>maximum)maximum=value;
    }
    // Integer-only percentage (one decimal place); no float formatting.
    const uint32_t percent10=(uint32_t(zeros)*1000u)/length;
    Serial.printf("ACT worker=%u frame=%lu stage=%u zeros=%u/%u sparsity=%lu.%01lu%% min=%d max=%d\n",
        unsigned(PIPE_WORKER_ID),(unsigned long)frame,unsigned(stage),
        unsigned(zeros),unsigned(length),
        (unsigned long)(percent10/10u),(unsigned long)(percent10%10u),
        int(minimum),int(maximum));
}
#endif

void execute(uint8_t stage,const int8_t* data,uint16_t len,uint32_t frame) {
    mode=COMPUTE;active_frame=frame;
    const uint32_t t0=micros();
    digitalWrite(LED_BUILTIN, LOW);  // ESP8266 NodeMCU: active-low LED ON
    const bool ok=runner.run(stage,data,len);
    digitalWrite(LED_BUILTIN, HIGH); // LED OFF, including inference failures
    last_compute_us=micros()-t0;
    if(!ok) {
        ++errors;control(noodlepipe::ERROR_PACKET,frame,stage);mode=FREE;
        Serial.printf("ERROR worker=%u stage=%u inference failed\n",PIPE_WORKER_ID,stage);
        return;
    }
    completed_stage=stage;mode=OUTPUT_READY;last_done=0;
    Serial.printf("DONE worker=%u frame=%lu stage=%u time_us=%lu output=%u heap=%lu\n",
        PIPE_WORKER_ID,(unsigned long)frame,stage,(unsigned long)last_compute_us,
        unsigned(runner.output_bytes()),(unsigned long)ESP.getFreeHeap());
#if NOODLE_WORKER_ACTIVATION_STATS
    // Output is fully materialized here, before FORWARD/RUN_LOCAL can reuse it.
    // The inference time_us measurement above excludes this scan/print.
    print_activation_stats(frame,stage);
#endif
}
void receive(const noodlepipe::Message& msg) {
    const noodlepipe::Packet& p=msg.p;
    // Source addresses are verified against sender IDs by NowLink::receive.
    if(p.kind==noodlepipe::HELLO && p.sender==noodlepipe::HOST_ID)return;
    // The sender owns output storage until all fragment ACKs are received.
    if(tx.handle(msg))return;
    if(p.kind==noodlepipe::DATA || p.kind==noodlepipe::ACK_REQUEST) {
        // Also ACK duplicate final fragments after the compute has completed.
        rx.handle(radio,msg);
        return;
    }
    if(p.kind==noodlepipe::PREPARE) {
        if(p.stage>=noodlepipe::NUM_STAGES)return;
        if(mode==FREE || rx.busy())rx.handle(radio,msg);
        else {
            noodlepipe::Packet e=noodlepipe::packet(noodlepipe::ERROR_PACKET,PIPE_WORKER_ID,p.frame,p.stage);
            e.flags=1;radio.send(msg.mac,e);
        }
        return;
    }
    if(p.sender!=noodlepipe::HOST_ID ||
       !noodlepipe::node_mac_matches(noodlepipe::HOST_ID,msg.mac))return;
    if(p.kind==noodlepipe::FORWARD && mode==OUTPUT_READY &&
       p.frame==active_frame && p.stage==completed_stage+1 &&
       msg.bytes==21) {
        const uint8_t target_id=p.payload[0];
        // Result must go to the host; intermediate tensors go to real workers.
        const bool valid_target=(p.stage==noodlepipe::RESULT_STAGE)
            ? (target_id==noodlepipe::HOST_ID)
            : noodlepipe::worker_id_valid(target_id);
        uint8_t target_mac[6];
        if(valid_target && target_id!=PIPE_WORKER_ID &&
           noodlepipe::node_mac(target_id,target_mac) &&
           tx.start(radio,target_mac,p.frame,p.stage,runner.output(),runner.output_bytes())) {
            mode=SENDING;
        } else {++errors;control(noodlepipe::ERROR_PACKET,p.frame,p.stage);}
    } else if(p.kind==noodlepipe::RUN_LOCAL && mode==OUTPUT_READY &&
              p.frame==active_frame && p.stage==completed_stage+1 && p.stage<noodlepipe::NUM_STAGES) {
        // The previous output is in buffer B; StageRunner copies B to A before
        // executing the next stage. No additional 4 KB temporary is needed.
        execute(p.stage,runner.output(),runner.output_bytes(),p.frame);
    }
}
}
void setup() {
    digitalWrite(LED_BUILTIN, HIGH); // Set output level before enabling GPIO
    pinMode(LED_BUILTIN, OUTPUT);
    Serial.begin(115200);delay(300);
    Serial.printf("NOODLE2_ESP8266_WORKER_%u\n",PIPE_WORKER_ID);
    if(!runner.begin()) {
        Serial.println("FATAL: Noodle2 model flash bind failed");for(;;)delay(1000);
    }
    if(!radio.begin(PIPE_WORKER_ID)) {
        Serial.println("FATAL: ESP-NOW init failed");for(;;)delay(1000);
    }
    Serial.printf("NOODLE_TRANSPORT_CRC=%d (0=OFF, 1=ON)\n", NOODLE_TRANSPORT_CRC);
    Serial.printf("NOODLE_ACK_INTERVAL=%d (4=baseline, 18=whole tensor)\n",
                  NOODLE_ACK_INTERVAL);
    Serial.printf("NOODLE_WORKER_ACTIVATION_STATS=%d (0=OFF, 1=ON)\n", NOODLE_WORKER_ACTIVATION_STATS);
    Serial.printf("MAC %02X:%02X:%02X:%02X:%02X:%02X channel=%u heap=%lu\n",
        radio.address()[0],radio.address()[1],radio.address()[2],
        radio.address()[3],radio.address()[4],radio.address()[5],
        noodlepipe::CHANNEL,(unsigned long)ESP.getFreeHeap());
}
void loop() {
    const uint32_t now=millis();
    noodlepipe::Message m;
    while(radio.poll(m))receive(m);
    if(rx.complete() && mode==FREE) {
        const uint32_t frame=rx.frame();const uint8_t stage=rx.stage();
        execute(stage,rx.data(),rx.size(),frame);
        rx.release();
    }
    tx.tick(radio,now);
    if(mode==SENDING && tx.state()==noodlepipe::Sender::COMPLETE) {
        Serial.printf("TX_V2 worker=%u frame=%lu stage=%u transfer_ms=%lu packets=%lu retries=%lu acks=%lu\n",
            PIPE_WORKER_ID,(unsigned long)tx.frame(),unsigned(tx.stage()),
            (unsigned long)tx.transfer_ms(now),(unsigned long)tx.sent_packets(),
            (unsigned long)tx.retransmitted(),(unsigned long)tx.ack_count());
        tx.clear();mode=FREE;++finished;
        control(noodlepipe::RELEASE,active_frame,completed_stage);
        Serial.printf("RELEASE worker=%u frame=%lu\n",PIPE_WORKER_ID,(unsigned long)active_frame);
    }
    if(mode==SENDING && tx.state()==noodlepipe::Sender::FAILED) {
        tx.clear();mode=OUTPUT_READY;++errors;
        control(noodlepipe::ERROR_PACKET,active_frame,completed_stage);
        Serial.println("WARN: transfer failed, preserving output");
    }
    if(mode==OUTPUT_READY && (last_done==0 || now-last_done>=350)) {
        control(noodlepipe::DONE,active_frame,completed_stage,runner.output_bytes(),
            noodlepipe::tensor_crc32(runner.output(),runner.output_bytes()));
        last_done=now?now:1;
    }
    if(now-last_hello>=1000) {advertise();last_hello=now;}
    delay(1); // Cooperative ESP8266 Wi-Fi service.
}
