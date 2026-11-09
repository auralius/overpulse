// Desktop diagnostic smoke test: compile actual worker code with stats ON,
// execute a real Noodle2 stage, and verify an output is produced.
#define PIPE_WORKER_ID 1
#include "pipeline/worker_app.h"
#include <cstdio>

int main(int argc,char** argv) {
    if(argc!=2)return 2;
    int8_t input[512]={};
    FILE* f=std::fopen(argv[1],"rb");
    if(!f)return 3;
    const bool read_ok=std::fread(input,1,sizeof(input),f)==sizeof(input) && std::fgetc(f)==EOF;
    std::fclose(f);
    if(!read_ok)return 4;
    setup();
    execute(0,input,sizeof(input),42);
    if(mode!=OUTPUT_READY || runner.output_bytes()!=2048)return 5;
    std::puts("WORKER_ACTIVATION_STATS_PASS");
    return 0;
}
