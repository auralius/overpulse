# OVERPULSE

**OVERPULSE** is a distributed TinyML demonstration for real-time PPG peak detection and heart-rate estimation.

It uses:
- **1× ESP32-S3 CYD host**
- **up to 9× ESP8266 workers**
- **1× MAX30100** optical pulse sensor
- **Noodle2 INT8 inference** with a distributed PeakNet1D model
- **ESP-NOW** for wireless activation transfer between workers

The project began as an intentionally overengineered heartbeat counter, but it also serves as a concrete experimental platform for:
- distributed inference,
- communication-aware TinyML,
- dynamic scheduling across MCUs,
- and Noodle2 runtime validation on real hardware.

> **Important:** OVERPULSE is a research / educational demonstrator. It is **not** a medical device and is **not clinically validated**.

---

## Hardware

![OVERPULSE hardware setup](docs/overpulse_hardware.jpg)

https://github.com/user-attachments/assets/8dbacf89-0eab-4d13-8e8f-18bc4e772a5a


**Shown above:** the ESP32-S3 CYD host, the MAX30100 sensor, and a 9-worker ESP8266 setup used for the latest OVERPULSE experiments.

The current acquisition driver targets **MAX30100**; MAX30102 is a future
host-side sensor-driver upgrade and is **not yet supported** by this package.


---

## Main features

- **Distributed PeakNet1D inference** using Noodle2 INT8
- **Dynamic worker scheduling** over ESP-NOW
- **Up to 9 ESP8266 workers** with the full model stored in Flash on every worker
- **NN BPM** estimated from neural peak detections
- **DSP BPM** estimated independently from filtered IR autocorrelation
- **Independent live waveform display** on the CYD: **Red** and **IR** traces continue even if no ESP8266 is online
- **CRC optional** transport mode (`NOODLE_TRANSPORT_CRC=0/1`)
- **Configurable selective ACK interval**: 4 (baseline), 8, or 18 (one bitmap ACK for 4 KiB tensors)
- **ACK_REQUEST recovery** for lost final fragments and lost group acknowledgments
- **Host admission and occupancy control** for multi-frame pipeline scheduling

---

## Current baseline configuration

- **Host:** ESP32-S3 CYD (ESP32-1732S019)
- **Workers:** 9× ESP8266
- **Sensor:** MAX30100
- **Sampling rate:** **100 Hz**
- **Input window:** **256 samples per channel**
- **Input shape to PeakNet1D:** **CHW `[2, 1, 256]`**
- **Window stride:** **64 samples**
- **Wireless transport:** ESP-NOW v1-compatible
- **Channel:** 6
- **Transport pacing:** 2 ms
- **ACK interval:** 4 fragments (baseline); 18 is whole-tensor experimental mode
- **Application protocol:** v4 (deterministic MAC and ID-only forwarding retained)
- **CRC:** disabled in the latest fast baseline

Observed end-to-end latency in the latest working setup was about **830–860 ms** with CRC disabled.

---

## Project structure

```text
OVERPULSE/
├── include/
│   └── peaknet_ppg_model.h
├── lib/
│   ├── Noodle2/
│   └── NoodlePipeline/
├── model_export/
│   ├── peaknet_ppg_int8.bin
│   ├── golden_input_chw_i8.bin
│   ├── golden_logits_i8.bin
│   └── model_manifest.json
├── src/
│   ├── main_host.cpp
│   ├── main_worker.cpp
│   ├── Max30100Raw.*
│   ├── BpmEstimator.h
│   ├── NnBpmEstimator.h
│   ├── PPGPreprocess.h
│   └── pipeline/
├── tests/
├── tools/
├── docs/
│   └── overpulse_hardware.jpg
├── TECHNICAL_DETAILS.md
├── platformio.ini
└── README.md
```

---

## Build environments

The `platformio.ini` includes:
- **1 host environment**
- **9 worker environments** (`worker1` ... `worker9`)

Typical usage:

```bash
pio run -e host -t upload
pio run -e worker1 -t upload
pio run -e worker2 -t upload
# ...
pio run -e worker9 -t upload
```

---

## Deterministic worker identities (protocol v3)

Each device uses a fixed, locally administered Wi-Fi station MAC derived from its node ID: host `02:4E:44:4C:00:00`, workers W1..W9 `02:4E:44:4C:00:01` through `...:09`. The host sends only the **destination worker ID** in `FORWARD`, and workers derive the MAC locally. All peers are registered at boot. This is **not cryptographic authentication**.

**Important:** Reflash **all 10 devices** when upgrading to ACK-interval protocol v4. It intentionally rejects protocol v3 messages and mismatched ACK interval or CRC settings. See `TECHNICAL_DETAILS.md` for implementation details.

## Configuration knobs

Some key build flags currently used:

```ini
-D NOODLE_MAX_WORKERS=9
-D NOODLE_MAX_OCCUPANCY=5
-D NOODLE_HOST_OCCUPANCY=3
-D NOODLE_RESERVED_WORKERS=1
-D NOODLE_TRANSPORT_CRC=0
-D NOODLE_ACK_INTERVAL=4
-D NOODLE_ESPNOW_PACING_MS=2
-D PPG_LED_CONFIG=0x77
```

These control worker scaling, admission policy, reserved handoff capacity, transport integrity mode, pacing, and LED drive current.

---

## ACK interval experiment (NoodleTransport, protocol v4)

A 4,096-byte activation tensor occupies **18 ESP-NOW DATA packets** (up to 230 bytes each).
The ACK interval controls the maximum number of consecutive fragments sent before
waiting for a bitmap acknowledgment:

| `NOODLE_ACK_INTERVAL` | ACK packets on a clean 4 KiB transfer | Expected behavior |
|---:|---:|---|
| `4` | 5 | **Default**, reproduces current 4-fragment ACK grouping |
| `8` | 3 | Intermediate setting |
| `18` | 1 | Whole-tensor bitmap ACK for the largest activation |

For a first A/B test, change **both** `[env:host]` and `[esp8266_common]` in `platformio.ini`:

```ini
-D NOODLE_ACK_INTERVAL=18
```

Keep `NOODLE_ESPNOW_PACING_MS=2`, `NOODLE_TRANSPORT_CRC=0`, and the same
occupancy settings during the comparison. All ten boards need v4 firmware with
**matching ACK intervals**. The receiver rejects mismatched configurations.

The bitmap is 32-bit, carried in the existing ACK packet's `crc` field (not a
CRC check); all tensor bytes remain unchanged. A missing final DATA packet or
ACK is recovered using `ACK_REQUEST` and selective retransmission. The new
interval does **not** remove per-packet radio pacing, and fewer ACKs do not
necessarily guarantee better performance on a busy channel.

Run local regression tests with `make test`. The tests simulate packet loss,
ACK loss, last-fragment loss, and exact INT8 inference; real-device performance
must be benchmarked separately.

---

## What Noodle2 demonstrates here

Each ESP8266 stores the complete PeakNet1D model, but at runtime it executes only the stage assigned by the host.

A core idea demonstrated by OVERPULSE is that the Noodle2 inference API stays simple even when deployed in a distributed setting.

At a conceptual level, a worker stage is just:

```cpp
NoodleTensor x = ...;           // input activation
NoodleParams p = ...;           // exported model parameters
NoodleTensor y = noodle_conv2d(x, p);
```

The distribution, transport, scheduling, and handoff logic are layered around this simple inference core.

---

## Technical documentation

For the detailed implementation notes, see:

- **[TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md)**

That document covers:
- LPF and HPF details,
- whether the NN uses filtered or raw input,
- the PeakNet1D architecture,
- Noodle2 equivalent code,
- sampling rate and input dimensions,
- producer–consumer design,
- scheduler behavior,
- and occupancy parameters.

---

## Notes on signal processing

A particularly important implementation detail:

- **The neural network does _not_ receive the filtered display waveform.**
- PeakNet1D uses the **raw IR and Red samples**, normalized per 256-sample window using mean and MAD.
- The host-side filters are used for:
  - **live waveform display**, and
  - the independent **DSP BPM** estimator.

---

## License / release note

This repository is prepared as a clean publication snapshot, without the historical incremental README files used during development.

---

## Live waveform display (host-side update)

The ESP32-S3 updates both Red/IR traces directly from its filtered sensor ring
about every **400 ms** (`PPG_UI_REFRESH_MS=400`) whether workers are online,
frames have completed, or NN BPM is valid. NN markers are time-aligned to
their original samples and disappear as they leave the scrolling window.
DSP BPM remains fully independent of wireless inference.

---

## A final remark

Yes, this may be the world's most overengineered heartbeat counter.
And **yes, we are still irrationally proud of it**.
