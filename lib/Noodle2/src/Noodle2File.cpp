#include "Noodle2File.h"
#if defined(ARDUINO)
#include <Arduino.h>
#endif
#if !defined(ARDUINO)
#include <chrono>
#endif
#if defined(ARDUINO) && defined(NOODLE2_ENABLE_SDFAT)
#include <SPI.h>
#ifndef NOODLE2_SD_CS
#define NOODLE2_SD_CS 5
#endif
#ifndef NOODLE2_SD_SCK
#define NOODLE2_SD_SCK 18
#endif
#ifndef NOODLE2_SD_MISO
#define NOODLE2_SD_MISO 19
#endif
#ifndef NOODLE2_SD_MOSI
#define NOODLE2_SD_MOSI 23
#endif
#ifndef NOODLE2_SD_SPI_MHZ
#define NOODLE2_SD_SPI_MHZ 16
#endif
#ifndef NOODLE2_SD_DEDICATED
#define NOODLE2_SD_DEDICATED 0
#endif
#endif

namespace noodle2 {
static uint32_t source_clock_us() {
#if defined(ARDUINO)
    return micros();
#else
    using namespace std::chrono;
    return uint32_t(duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count());
#endif
}
NoodleParamFile::NoodleParamFile() : path_(0), initialized_(false), active_(false),
    reads_(0), bytes_(0), opens_(0), closes_(0),
    read_us_(0), open_us_(0), close_us_(0)
#if !defined(ARDUINO)
    , file_(0)
#endif
{}
NoodleParamFile::~NoodleParamFile() { close(); }
Status NoodleParamFile::bind_file(const char* path) {
    if (active_ || !path || !*path) return BAD_PARAMS;
    path_ = path; // caller/generated model owns the literal for its lifetime
#if defined(ARDUINO) && defined(NOODLE2_ENABLE_SDFAT)
    if (!initialized_) {
        SPI.begin(NOODLE2_SD_SCK, NOODLE2_SD_MISO, NOODLE2_SD_MOSI, NOODLE2_SD_CS);
#if NOODLE2_SD_DEDICATED
        initialized_ = sd_.begin(SdSpiConfig(NOODLE2_SD_CS, DEDICATED_SPI, SD_SCK_MHZ(NOODLE2_SD_SPI_MHZ)));
#else
        initialized_ = sd_.begin(SdSpiConfig(NOODLE2_SD_CS, SHARED_SPI, SD_SCK_MHZ(NOODLE2_SD_SPI_MHZ)));
#endif
        if (!initialized_) return IO_ERROR;
    }
#elif defined(ARDUINO)
    return UNSUPPORTED; // board adapter must be provided, never pretend SD works
#else
    initialized_ = true;
#endif
    return OK;
}
Status NoodleParamFile::open() {
    if (!initialized_ || active_ || !path_) return BAD_PARAMS;
    const uint32_t tick = source_clock_us();
#if defined(ARDUINO) && defined(NOODLE2_ENABLE_SDFAT)
    active_ = file_.open(path_, O_RDONLY);
#elif defined(ARDUINO)
    return UNSUPPORTED;
#else
    file_ = fopen(path_, "rb");
    active_ = file_ != 0;
#endif
    open_us_ += source_clock_us()-tick;
    if (active_) ++opens_;
    return active_ ? OK : IO_ERROR;
}
Status NoodleParamFile::read(uint8_t* dst, uint32_t count) {
    if (!active_ || !dst) return IO_ERROR;
    if (count == 0) return OK;
    const uint32_t tick = source_clock_us();
#if defined(ARDUINO) && defined(NOODLE2_ENABLE_SDFAT)
    const int32_t n = file_.read(dst, count);
    if (n < 0 || uint32_t(n) != count) return IO_ERROR;
#elif defined(ARDUINO)
    return UNSUPPORTED;
#else
    if (fread(dst, 1, count, file_) != count) return IO_ERROR;
#endif
    read_us_ += source_clock_us()-tick;
    ++reads_; bytes_ += count;
    return OK;
}
void NoodleParamFile::close() {
    if (!active_) return;
    const uint32_t tick = source_clock_us();
#if defined(ARDUINO) && defined(NOODLE2_ENABLE_SDFAT)
    file_.close();
#elif !defined(ARDUINO)
    fclose(file_); file_ = 0;
#endif
    active_ = false;
    close_us_ += source_clock_us()-tick;
    ++closes_;
}
}  // namespace noodle2
