# VAANI JAGRAN ESP32-S3 Firmware

This guide covers the ESP32-S3 firmware in `VAANI_JAGRAN_ASR.ino`.

The firmware performs keyword spotting locally and uses the ESP32's two cores:

- **Core 1 / Arduino loop**: I2S capture, RMS tracking, MFCC extraction, KWS inference, VAD, state transitions, OLED updates.
- **Core 0 / `wsTask`**: WiFi/WebSocket lifecycle, PCM queue transmission, telemetry, reconnect handling.

The firmware sends only audio captured after the `Sakha` keyword is detected. It does not transmit a pre-roll containing the keyword.

## Folder

```text
essp32/VAANI_JAGRAN_ASR/
├── VAANI_JAGRAN_ASR.ino
├── model_data.h
├── secrets.example.h
├── dct_matrix.h
├── hann_window.h
├── mel_sparse.h
├── kiss_fft.c
├── kiss_fft.h
├── kiss_fftr.c
├── kiss_fftr.h
├── kiss_fft_log.h
└── _kiss_fft_guts.h
```

`model_data.h` is the generated C header containing the embedded int8 KWS model. A source `.tflite` file is optional and is not required to flash the current sketch.

## Hardware Wiring

### INMP441 microphone

| INMP441 pin | ESP32-S3 pin | Purpose |
|---|---:|---|
| VDD | 3.3V | Power |
| GND | GND | Ground |
| L/R | GND | Left channel |
| SCK | GPIO 41 | I2S bit clock |
| WS | GPIO 40 | I2S word select |
| SD | GPIO 42 | I2S data |

### SSD1306 OLED

| OLED pin | ESP32-S3 pin | Purpose |
|---|---:|---|
| VCC | 3.3V | Power |
| GND | GND | Ground |
| SDA | GPIO 8 | I2C data |
| SCL | GPIO 9 | I2C clock |

## Arduino IDE Setup

Install or select:

- Board package: ESP32 by Espressif
- Board: `ESP32S3 Dev Module`
- CPU frequency: `240MHz`
- Partition scheme: `Huge APP (3MB No OTA/1MB SPIFFS)`
- USB mode: Hardware CDC and JTAG, if available on the board
- Upload speed: `921600`, if stable for the board

Install these libraries through Arduino Library Manager or the project/team's approved library source:

- `TFLiteMicro_ArduinoESP32S3`
- `Adafruit SSD1306`
- `Adafruit GFX Library`
- `WebSockets` / `arduinoWebSockets` by Markus Sattler (Links2004)

Open the sketch from its matching folder:

```text
esp32/VAANI_JAGRAN_ASR/VAANI_JAGRAN_ASR.ino
```

Arduino requires the sketch folder and `.ino` basename to match.

## Local Network Configuration

Create the ignored local secrets file:

```bash
cp esp32/VAANI_JAGRAN_ASR/secrets.example.h \
   esp32/VAANI_JAGRAN_ASR/secrets.h
```

Edit `secrets.h`:

```cpp
#define WIFI_SSID "your_wifi_name"
#define WIFI_PASS "your_wifi_password"
#define WS_HOST "192.168.1.50"
```

Use the computer's LAN IP for `WS_HOST`. Do not use `localhost` or `127.0.0.1` because those addresses refer to the ESP32 itself.

`secrets.h` is ignored by Git. Keep real credentials out of commits and public documentation.

## Firmware State Machine

```text
LISTENING
    |
    | KWS confidence threshold reached
    v
DETECTED
    |
    | start_stream sent by Core 0 task
    v
STREAMING
    |
    | 1500 ms short-RMS silence, 10 s timeout, or disconnect
    v
LISTENING
```

OLED states include calibration, listening, detected, and streaming. Streaming shows elapsed time and live status information.

## Audio Pipeline

- Sample rate: `16 kHz`
- I2S input: 32-bit samples converted to 24-bit signed values
- KWS hop: `320 samples` / `20 ms`
- FFT size: `512`
- MFCC count: `13`
- KWS input: rolling MFCC frame tensor defined by the embedded model
- PCM stream: mono signed `int16`, `640 bytes` per 20 ms chunk

The firmware continuously fills the long RMS ring for calibration/listening gates and uses the 200 ms `g_short_rms` window for stream-end VAD. The long one-second RMS is deliberately not used to end speech streams.

## Streaming and Queue Behavior

When KWS fires:

1. The firmware enters `DETECTED`.
2. Core 0 sends `start_stream` JSON with the ESP32 timestamp.
3. The firmware enters `STREAMING`.
4. New 20 ms audio hops are converted from float to clamped `int16` samples.
5. PCM chunks enter a 20-item FreeRTOS queue.
6. Core 0 drains the queue and sends binary WebSocket frames.

If the queue is full, the oldest chunk is discarded so the newest audio remains available. If WiFi/WebSocket disconnects during streaming, the firmware falls back to `LISTENING`.

## Backend Contract

The backend must be running at the address configured in `WS_HOST` and port `8000` with this endpoint:

```text
ws://<WS_HOST>:8000/ws/esp32
```

The backend expects:

- JSON control and telemetry frames
- Binary signed-int16 mono PCM frames
- 16 kHz sample rate
- 640-byte PCM chunks

Start the backend before booting or reconnecting the ESP32. See the root `README.md` for the complete project startup procedure.

## Calibration and Boot Sequence

1. WiFi connects with a 10-second timeout.
2. Large buffers, I2S, FFT, and the TensorFlow Lite model initialize.
3. A five-second ambient calibration runs while the OLED displays `Stay quiet`.
4. WebSocket setup starts after calibration, so the calibration screen is not interrupted by a WebSocket handshake.
5. The firmware enters `LISTENING`.

Speak quietly during calibration. The firmware estimates gain and a silence threshold from ambient audio.

## Model Replacement

The sketch includes:

```text
model_data.h
```

To regenerate it from a compatible TensorFlow Lite model:

```bash
python3 esp32/convert_model.py path/to/model.tflite
```

The converter writes a header containing:

```cpp
const unsigned char g_model[];
const int g_model_len;
```

These names must remain compatible with `VAANI_JAGRAN_ASR.ino`.

## Serial Monitor

Use `115200 baud`. Useful messages include:

- WiFi connection result
- calibration progress and calculated threshold
- model allocation status
- WebSocket connection/reconnection status
- KWS detection
- stream-end reason
- CPU, RMS, queue, and telemetry information

## Troubleshooting

### Sketch does not compile

- Confirm the `.ino` is opened from `esp32/VAANI_JAGRAN_ASR/`.
- Confirm all local headers in that folder are present.
- Install the listed Arduino libraries.
- Check that the selected board is ESP32-S3.
- Ensure `model_data.h` contains `g_model` and `g_model_len`.

### WiFi connects but WebSocket does not

- Confirm the backend is running on port `8000`.
- Use the computer's LAN IP in `WS_HOST`.
- Put the ESP32 and computer on the same network.
- Check the computer firewall.
- Verify the backend route `/ws/esp32` is available.

### Audio stream contains the keyword

Reflash the current `VAANI_JAGRAN_ASR.ino`. The current firmware has no pre-roll transmission path; it sends only PCM chunks queued after detection.

### KWS is unstable

Tune these firmware constants carefully:

```cpp
#define KW_THRESHOLD 0.78f
#define KW_STREAK_N 3
#define KW_COOLDOWN_MS 2000
```

Changes affect false positives, missed detections, and repeated triggers.
