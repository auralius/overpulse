#include "NoodleNow.h"
#include <string.h>
#if defined(ARDUINO_ARCH_ESP8266)
#include <ESP8266WiFi.h>
extern "C" {
#include <espnow.h>
#include <user_interface.h>
}
#elif defined(ARDUINO_ARCH_ESP32)
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_idf_version.h>
#else
#error ESP-NOW pipeline supports only ESP8266 or ESP32-S3
#endif
namespace noodlepipe {
static NowLink* instance_=nullptr;
#if defined(ARDUINO_ARCH_ESP32)
static portMUX_TYPE queue_lock=portMUX_INITIALIZER_UNLOCKED;
#endif
static void lock_queue() {
#if defined(ARDUINO_ARCH_ESP32)
    portENTER_CRITICAL(&queue_lock);
#else
    noInterrupts();
#endif
}
static void unlock_queue() {
#if defined(ARDUINO_ARCH_ESP32)
    portEXIT_CRITICAL(&queue_lock);
#else
    interrupts();
#endif
}
#if defined(ARDUINO_ARCH_ESP8266)
static void receive_callback(uint8_t* mac,uint8_t* data,uint8_t len) {
    if(instance_) instance_->receive(mac,data,len);
}
#elif ESP_IDF_VERSION_MAJOR>=5
static void receive_callback(const esp_now_recv_info_t* info,const uint8_t* data,int len) {
    if(instance_ && info) instance_->receive(info->src_addr,data,len);
}
#else
static void receive_callback(const uint8_t* mac,const uint8_t* data,int len) {
    if(instance_) instance_->receive(mac,data,len);
}
#endif
NowLink::NowLink() : head_(0),tail_(0),dropped_(0),local_mac_{0},node_id_(0) {}
bool NowLink::begin(uint8_t id) {
    uint8_t configured_mac[6];
    if(!node_mac(id,configured_mac))return false;
    node_id_=id;
    WiFi.persistent(false); // Never write station settings to flash at startup.
    if(!WiFi.mode(WIFI_STA))return false;
    WiFi.disconnect();
#if defined(ARDUINO_ARCH_ESP8266)
    WiFi.setSleepMode(WIFI_NONE_SLEEP);
    if(!wifi_set_macaddr(STATION_IF,configured_mac))return false;
    wifi_set_channel(CHANNEL);
    if(!wifi_get_macaddr(STATION_IF,local_mac_))return false;
    if(esp_now_init()!=0)return false;
    if(esp_now_set_self_role(ESP_NOW_ROLE_COMBO)!=0)return false;
#else
    WiFi.setSleep(false);
    if(esp_wifi_set_mac(WIFI_IF_STA,configured_mac)!=ESP_OK)return false;
    if(esp_wifi_set_channel(CHANNEL,WIFI_SECOND_CHAN_NONE)!=ESP_OK)return false;
    if(esp_wifi_get_mac(WIFI_IF_STA,local_mac_)!=ESP_OK)return false;
    if(esp_now_init()!=ESP_OK)return false;
#endif
    if(!equal_mac(local_mac_,configured_mac))return false;
    instance_=this;
    esp_now_register_recv_cb(receive_callback);
    if(!peer(BROADCAST))return false;
    // Pre-register all known identities once. 1 host + MAX_WORKERS <= 20 peers
    // including broadcast; no MAC directory, discovery or first-send add needed.
    for(uint8_t other=HOST_ID;other<=MAX_WORKERS;++other) {
        if(other==node_id_)continue;
        uint8_t target[6];node_mac(other,target);
        if(!peer(target))return false;
    }
    return true;
}
bool NowLink::peer(const uint8_t mac[6]) {
#if defined(ARDUINO_ARCH_ESP8266)
    uint8_t m[6];memcpy(m,mac,6);
    if(esp_now_is_peer_exist(m))return true;
    return esp_now_add_peer(m,ESP_NOW_ROLE_COMBO,CHANNEL,nullptr,0)==0;
#else
    if(esp_now_is_peer_exist(mac))return true;
    esp_now_peer_info_t p={};
    memcpy(p.peer_addr,mac,6);
    p.channel=CHANNEL;p.ifidx=WIFI_IF_STA;p.encrypt=false;
    return esp_now_add_peer(&p)==ESP_OK;
#endif
}
bool NowLink::send(const uint8_t mac[6],const Packet& p,uint16_t body) {
    if(body>PAYLOAD || !peer(mac))return false;
    const uint16_t n=static_cast<uint16_t>(offsetof(Packet,payload)+body);
#if defined(ARDUINO_ARCH_ESP8266)
    uint8_t dst[6];memcpy(dst,mac,6);
    return esp_now_send(dst,const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&p)),static_cast<uint8_t>(n))==0;
#else
    return esp_now_send(mac,reinterpret_cast<const uint8_t*>(&p),n)==ESP_OK;
#endif
}
bool NowLink::send_to(uint8_t id, const Packet& p, uint16_t body) {
    uint8_t target[6];
    return node_mac(id,target) && send(target,p,body);
}
void NowLink::receive(const uint8_t* mac,const uint8_t* data,int n) {
    if(!mac||!data||n<20||n>MAX_PACKET)return;
    Packet p={};memcpy(&p,data,static_cast<size_t>(n));
    if(!valid(p,static_cast<uint16_t>(n)) || !node_mac_matches(p.sender,mac))return;
    lock_queue();
    const uint8_t next=static_cast<uint8_t>((head_+1)%QCAP);
    if(next==tail_) {++dropped_;}
    else {
        Message& m=queue_[head_];m.p=p;m.bytes=static_cast<uint16_t>(n);
        memcpy(m.mac,mac,6);head_=next;
    }
    unlock_queue();
}
bool NowLink::poll(Message& out) {
    lock_queue();
    const bool ok=head_!=tail_;
    if(ok) {out=queue_[tail_];tail_=static_cast<uint8_t>((tail_+1)%QCAP);}
    unlock_queue();
    return ok;
}
}
