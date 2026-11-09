#include <Noodle2.h>
#include "../src/pipeline/StageRunner.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
static bool read_file(const char* path,void* dst,size_t n) {
    FILE* f=std::fopen(path,"rb");
    if(!f)return false;
    const bool ok=std::fread(dst,1,n,f)==n && std::fgetc(f)==EOF;
    std::fclose(f);return ok;
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    int8_t data[noodlepipe::MAX_TENSOR]={},scratch[noodlepipe::MAX_TENSOR]={};
    int8_t expected[256]={};
    if(!read_file(argv[1],data,512)||!read_file(argv[2],expected,256))return 3;
    noodlepipe::StageRunner runner;
    if(!runner.begin())return 4;
    uint16_t n=512;
    for(unsigned stage=0;stage<noodlepipe::NUM_STAGES;++stage) {
        if(!runner.run(stage,data,n))return 5;
        n=runner.output_bytes();
        std::memcpy(scratch,runner.output(),n);
        std::memcpy(data,scratch,n);
        std::printf("stage=%u bytes=%u crc=%08lx\n",stage,n,
            (unsigned long)noodlepipe::crc32(data,n));
    }
    if(n!=256 || std::memcmp(data,expected,256)!=0) {
        unsigned mismatches=0;
        for(unsigned i=0;i<256;++i)if(data[i]!=expected[i])++mismatches;
        std::printf("FAIL mismatches=%u\n",mismatches);return 6;
    }
    std::puts("PASS: all five pipeline stages, six Conv1D ops, 256/256 exact logits");
    return 0;
}
