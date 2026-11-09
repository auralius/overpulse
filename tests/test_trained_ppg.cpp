#include <Noodle2.h>
#include "PPGPreprocess.h"
#include "peaknet_ppg_model.h"
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

int main(int argc, char** argv) {
    if(argc!=3) { std::fprintf(stderr,"usage: %s input.bin expected.bin\n",argv[0]);return 2; }
    int8_t input[512],expected[256];
    FILE* fi=std::fopen(argv[1],"rb");
    if(!fi || std::fread(input,1,sizeof(input),fi)!=sizeof(input))return 3;
    std::fclose(fi);
    FILE* fo=std::fopen(argv[2],"rb");
    if(!fo || std::fread(expected,1,sizeof(expected),fo)!=sizeof(expected))return 4;
    std::fclose(fo);
    peaknet_ppg::Model model;
    if(model.bind_flash()!=noodle2::OK) return 5;
    noodle2::NoodleGraph net;
    if(net.set_arena_memory(noodle2::ARENA_SRAM)!=noodle2::OK)return 6;
    auto x=net.input(2,1,256);
    x=net.conv2d(x,model.conv01,noodle2::Conv2D::same(1,9),noodle2::RELU);
    x=net.conv2d(x,model.conv02,noodle2::Conv2D::same(1,7),noodle2::RELU);
    x=net.conv2d(x,model.conv03,noodle2::Conv2D::same(1,7),noodle2::RELU);
    x=net.conv2d(x,model.conv04,noodle2::Conv2D::same(1,7),noodle2::RELU);
    x=net.conv2d(x,model.conv05,noodle2::Conv2D::same(1,1),noodle2::RELU);
    net.conv2d(x,model.conv06,noodle2::Conv2D::same(1,1),noodle2::LOGITS);
    if(net.compile()!=noodle2::OK || net.set_input(input)!=noodle2::OK || net.run()!=noodle2::OK)return 7;
    unsigned mismatch=0;
    for(unsigned i=0;i<256;i++) {
        int8_t got=0;
        if(net.output().get(i,got)!=noodle2::OK)return 8;
        mismatch+=got!=expected[i];
    }
    if(mismatch) { std::printf("GOLDEN_FAIL mismatches=%u\n",mismatch);return 9; }
    std::printf("TRAINED_PPG_GOLDEN_PASS exact=256/256 arena=%lu bytes\n",(unsigned long)net.arena_bytes());
    return 0;
}
