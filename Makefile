CXX ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -Wextra -Werror
CPPFLAGS := -Ilib/NoodlePipeline/src -Ilib/Noodle2/src -Iinclude -Isrc -DNOODLE2_GRAPH_MAX_NODES=16
NOODLE_SRC := $(wildcard lib/Noodle2/src/*.cpp)

.PHONY: test test-bpm test-trained test-max30100-mock clean

test: test-bpm test-nn-bpm test-trained test-max30100-mock test-max30100-tutorial-led test-dual-wave-display test-pipeline test-protocol test-transfer test-transfer-no-crc test-transfer-ack8 test-transfer-ack18 test-transfer-ack18-crc test-host-mock test-worker-mock test-esp8266-flash-mock test-host-admission test-worker-stats test-worker-registry test-worker9-mock test-static-ids

build:
	mkdir -p build

test-nn-bpm: | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_nn_bpm.cpp -o build/test_nn_bpm
	./build/test_nn_bpm

test-bpm: | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_bpm.cpp -o build/test_bpm
	./build/test_bpm

test-trained: | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NOODLE_SRC) tests/test_trained_ppg.cpp -o build/test_trained_ppg
	./build/test_trained_ppg model_export/golden_input_chw_i8.bin model_export/golden_logits_i8.bin

test-max30100-tutorial-led: | build
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Werror -DPPG_LED_CONFIG=0x8F -Itests/i2c_stub -Isrc src/Max30100Raw.cpp tests/test_max30100_driver.cpp -o build/test_max30100_led8f
	./build/test_max30100_led8f

test-max30100-mock: | build
	$(CXX) -std=c++11 -O2 -Wall -Wextra -Werror -Itests/i2c_stub -Isrc src/Max30100Raw.cpp tests/test_max30100_driver.cpp -o build/test_max30100_driver
	./build/test_max30100_driver

clean:
	rm -rf build

.PHONY: test-pipeline test-protocol
test-pipeline: | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(NOODLE_SRC) tests/test_pipeline_golden.cpp -o build/test_pipeline_golden
	./build/test_pipeline_golden model_export/golden_input_chw_i8.bin model_export/golden_logits_i8.bin

test-protocol: | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_pipeline_protocol.cpp -o build/test_pipeline_protocol
	./build/test_pipeline_protocol

.PHONY: test-transfer test-transfer-no-crc test-host-mock test-worker-mock
test-transfer: | build
	$(CXX) -Itests/transfer_stub -Ilib/NoodlePipeline/src -Isrc $(CXXFLAGS) tests/test_transfer.cpp -o build/test_transfer
	./build/test_transfer

# Same transport/retransmission tests with application CRC disabled.
.PHONY: test-transfer-no-crc
test-transfer-no-crc: | build
	$(CXX) -DNOODLE_TRANSPORT_CRC=0 -Itests/transfer_stub -Ilib/NoodlePipeline/src -Isrc $(CXXFLAGS) tests/test_transfer.cpp -o build/test_transfer_no_crc
	./build/test_transfer_no_crc


# Parameter sweep: ACK every 8 or 18 fragments (18 = whole 4KB tensor).
.PHONY: test-transfer-ack8 test-transfer-ack18 test-transfer-ack18-crc
test-transfer-ack8: | build
	$(CXX) -DNOODLE_ACK_INTERVAL=8 -DNOODLE_TRANSPORT_CRC=0 -Itests/transfer_stub -Ilib/NoodlePipeline/src -Isrc $(CXXFLAGS) tests/test_transfer.cpp -o build/test_transfer_ack8
	./build/test_transfer_ack8

test-transfer-ack18: | build
	$(CXX) -DNOODLE_ACK_INTERVAL=18 -DNOODLE_TRANSPORT_CRC=0 -Itests/transfer_stub -Ilib/NoodlePipeline/src -Isrc $(CXXFLAGS) tests/test_transfer.cpp -o build/test_transfer_ack18
	./build/test_transfer_ack18

test-transfer-ack18-crc: | build
	$(CXX) -DNOODLE_ACK_INTERVAL=18 -DNOODLE_TRANSPORT_CRC=1 -Itests/transfer_stub -Ilib/NoodlePipeline/src -Isrc $(CXXFLAGS) tests/test_transfer.cpp -o build/test_transfer_ack18_crc
	./build/test_transfer_ack18_crc

test-host-mock: | build
	$(CXX) -Itests/pipeline_stub -Itests/display_stub -Itests/esp32_capture_stub -Itests/esp32_stub $(CPPFLAGS) -DNOODLE_MAX_WORKERS=9 $(CXXFLAGS) -DARDUINO_ARCH_ESP32=1 -DARDUINO=1 -DNOODLE2_DEFAULT_ARENA_PSRAM=1 -include tests/display_stub/gpio_shim.h $(NOODLE_SRC) src/Max30100Raw.cpp src/main_host.cpp tests/esp32_stub/entry.cpp -o build/test_host_mock
	./build/test_host_mock

test-worker-mock: | build
	$(CXX) -Itests/pipeline_stub -Itests/esp32_capture_stub -Itests/esp32_stub $(CPPFLAGS) -DNOODLE_MAX_WORKERS=9 $(CXXFLAGS) -DARDUINO_ARCH_ESP32=1 -DARDUINO=1 -DNOODLE2_DEFAULT_ARENA_PSRAM=0 $(NOODLE_SRC) -DPIPE_WORKER_ID=1 src/main_worker.cpp tests/esp32_stub/entry.cpp -o build/test_worker_mock
	./build/test_worker_mock

.PHONY: test-esp8266-flash-mock
test-esp8266-flash-mock: | build
	$(CXX) -Itests/esp8266_stub $(CPPFLAGS) $(CXXFLAGS) -DARDUINO_ARCH_ESP8266=1 $(NOODLE_SRC) tests/test_pipeline_golden.cpp -o build/test_esp8266_flash_mock
	./build/test_esp8266_flash_mock model_export/golden_input_chw_i8.bin model_export/golden_logits_i8.bin

.PHONY: test-host-admission
test-host-admission: | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_host_admission.cpp -o build/test_host_admission
	./build/test_host_admission

# Build real worker execution path with the diagnostic switch enabled.
.PHONY: test-worker-stats
test-worker-stats: | build
	$(CXX) -Itests/pipeline_stub -Itests/esp32_capture_stub -Itests/esp32_stub $(CPPFLAGS) $(CXXFLAGS) -DNOODLE_WORKER_ACTIVATION_STATS=1 -DARDUINO_ARCH_ESP32=1 -DARDUINO=1 -DNOODLE2_DEFAULT_ARENA_PSRAM=0 $(NOODLE_SRC) tests/test_worker_activation_stats.cpp -o build/test_worker_stats
	./build/test_worker_stats model_export/golden_input_chw_i8.bin

# Nine-worker registration and selection regression (host build, radio mocked).
.PHONY: test-worker-registry
test-worker-registry: | build
	$(CXX) -Itests/pipeline_stub -Itests/display_stub -Itests/esp32_capture_stub -Itests/esp32_stub $(CPPFLAGS) -DNOODLE_MAX_WORKERS=9 $(CXXFLAGS) -DARDUINO_ARCH_ESP32=1 -DARDUINO=1 -DNOODLE2_DEFAULT_ARENA_PSRAM=1 -include tests/display_stub/gpio_shim.h $(NOODLE_SRC) src/Max30100Raw.cpp tests/test_worker_registry.cpp -o build/test_worker_registry
	./build/test_worker_registry

# Shared firmware source with the highest worker ID.
.PHONY: test-worker9-mock
test-worker9-mock: | build
	$(CXX) -Itests/pipeline_stub -Itests/esp32_capture_stub -Itests/esp32_stub $(CPPFLAGS) -DNOODLE_MAX_WORKERS=9 $(CXXFLAGS) -DARDUINO_ARCH_ESP32=1 -DARDUINO=1 -DNOODLE2_DEFAULT_ARENA_PSRAM=0 $(NOODLE_SRC) -DPIPE_WORKER_ID=9 src/main_worker.cpp tests/esp32_stub/entry.cpp -o build/test_worker9_mock
	./build/test_worker9_mock

# Actual host UI code against a recording TFT mock: Red/IR same x positions,
# separate y scale, two traces, and overlaid NN candidate dots.
.PHONY: test-dual-wave-display
test-dual-wave-display: | build
	$(CXX) -Itests/pipeline_stub -Itests/display_stub -Itests/esp32_capture_stub -Itests/esp32_stub $(CPPFLAGS) -DNOODLE_MAX_WORKERS=9 $(CXXFLAGS) -DARDUINO_ARCH_ESP32=1 -DARDUINO=1 -DNOODLE2_DEFAULT_ARENA_PSRAM=1 -include tests/display_stub/gpio_shim.h $(NOODLE_SRC) src/Max30100Raw.cpp tests/test_dual_wave_plot.cpp -o build/test_dual_wave_plot
	./build/test_dual_wave_plot

.PHONY: test-static-ids
test-static-ids: | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DNOODLE_MAX_WORKERS=9 tests/test_static_node_ids.cpp -o build/test_static_ids
	./build/test_static_ids
