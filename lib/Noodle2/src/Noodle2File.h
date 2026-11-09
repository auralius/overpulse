#pragma once
// Model-level forward-only file source. Board wiring belongs in build flags,
// not in NoodleGraph, operators, or application topology.
#include "Noodle2Stream.h"
#if defined(ARDUINO) && defined(NOODLE2_ENABLE_SDFAT)
#include <SdFat.h>
#endif
#if !defined(ARDUINO)
#include <stdio.h>
#endif

namespace noodle2 {

class NoodleParamSource {
public:
    virtual ~NoodleParamSource() {}
    virtual Status open() = 0;
    virtual Status read(uint8_t* dst, uint32_t count) = 0;
    virtual void close() = 0;
    static Status read_adapter(void* context, uint8_t* dst, uint32_t count) {
        return static_cast<NoodleParamSource*>(context)->read(dst, count);
    }
};

// One model path, one stream per inference, never seek. When SD is enabled,
// a single SdFat session is initialized at bind_file(), but the model file
// itself is opened exactly once per net.run().
class NoodleParamFile : public NoodleParamSource {
public:
    NoodleParamFile();
    ~NoodleParamFile();
    NoodleParamFile(const NoodleParamFile&) = delete;
    NoodleParamFile& operator=(const NoodleParamFile&) = delete;
    Status bind_file(const char* path);
    Status open() override;
    Status read(uint8_t* dst, uint32_t count) override;
    void close() override;
    uint32_t reads() const { return reads_; }
    uint32_t bytes() const { return bytes_; }
    uint32_t opens() const { return opens_; }
    uint32_t closes() const { return closes_; }
    uint32_t read_us() const { return read_us_; }
    uint32_t open_us() const { return open_us_; }
    uint32_t close_us() const { return close_us_; }
    // No seek method intentionally exists.
private:
    const char* path_;
    bool initialized_, active_;
    uint32_t reads_, bytes_, opens_, closes_;
    uint32_t read_us_, open_us_, close_us_;
#if defined(ARDUINO) && defined(NOODLE2_ENABLE_SDFAT)
    SdFat32 sd_;
    File32 file_;
#elif !defined(ARDUINO)
    FILE* file_;
#endif
};

}  // namespace noodle2
