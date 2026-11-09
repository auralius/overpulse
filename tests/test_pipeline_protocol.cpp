#include <PipelineProtocol.h>
#include <cstdio>
#include <cstring>
#include <cassert>
int main(){
  using namespace noodlepipe;
  assert(sizeof(Packet)==250);
  assert(tensor_input(0)==512 && tensor_input(2)==4096 && tensor_input(5)==256);
  assert(tensor_output(4)==256);
  uint8_t a[4096],b[4096]={};
  for(int i=0;i<4096;++i)a[i]=static_cast<uint8_t>((i*91+11)%256);
  uint32_t original=crc32(a,4096);
  unsigned n=0;
  for(int at=0;at<4096;) {
    Packet p=packet(DATA,2,10,3);
    p.seq=n++;
    p.size=static_cast<uint16_t>((4096-at)<PAYLOAD?(4096-at):PAYLOAD);
    memcpy(p.payload,a+at,p.size);
    assert(valid(p,static_cast<uint16_t>(20+p.size)));
    memcpy(b+at,p.payload,p.size);
    at+=p.size;
  }
  assert(n==18 && crc32(b,4096)==original && memcmp(a,b,4096)==0);
  std::printf("PASS: 4 KB activation = %u v1 fragments, whole-tensor CRC=%08lx\n",n,
      (unsigned long)original);
}
