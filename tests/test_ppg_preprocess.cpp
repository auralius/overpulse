#include "PPGPreprocess.h"
#include <stdio.h>
#include <stdint.h>
int main(int argc,char** argv) {
    if(argc!=3) {fprintf(stderr,"usage: test_ppg_preprocess ir_red_raw_512xuint16.bin out_q.bin\n");return 1;}
    uint16_t raw[512];int8_t out[512];
    FILE* f=fopen(argv[1],"rb");
    if(!f || fread(raw,2,512,f)!=512)return 2;
    fclose(f);
    ppg::raw_to_chw_i8(raw,raw+256,out);
    f=fopen(argv[2],"wb");
    if(!f || fwrite(out,1,512,f)!=512)return 3;
    fclose(f);
    return 0;
}
