# VAANI JAGRAN

Real-time keyword spotting, post-detection speech streaming, and multilingual ASR using an ESP32-S3, FastAPI, and a browser dashboard.

This repository contains three connected parts:

1. **ESP32-S3 firmware**: captures microphone audio, runs the Sakha keyword spotter, performs VAD, and sends only audio captured after keyword detection.
2. **FastAPI backend**: receives PCM audio and telemetry over WebSocket, runs faster-whisper, saves WAV recordings, and broadcasts live results.
3. **Web dashboard**: displays connection state, transcripts, telemetry, waveform activity, and saved recordings.

## System Flow

```text
INMP441 microphone
        |
        v
ESP32-S3 Core 1: I2S -> MFCC -> KWS -> short-RMS VAD
        |
        | keyword detected
        v
ESP32-S3 Core 0: WebSocket -> 16-bit PCM chunks + telemetry
        |
        v
FastAPI backend: buffer -> faster-whisper -> WAV recordings
        |
        v
Browser dashboard: live transcript, state, telemetry, recordings
```

The firmware does not transmit the `Sakha` keyword pre-roll. Streaming starts with audio captured after keyword detection.

## Repository Layout

```text
SIH_anti/
├── backend/
│   ├── main.py                 # FastAPI app and WebSocket/HTTP endpoints
│   ├── asr.py                  # faster-whisper configuration and wrapper
│   ├── audio_buffer.py         # PCM buffering and WAV finalization
│   ├── requirements.txt        # Python dependencies
│   └── recordings/             # Runtime WAV files; ignored by Git
├── frontend/
│   ├── index.html              # Dashboard markup
│   ├── script.js               # WebSocket client and UI logic
│   └── style.css               # Dashboard styling
├── esp32/
│   ├── VAANI_JAGRAN_ASR/
│   │   ├── VAANI_JAGRAN_ASR.ino # ESP32-S3 firmware
│   │   ├── model_data.h         # Embedded int8 KWS model
│   │   ├── secrets.example.h    # Firmware configuration template
│   │   └── DSP/model headers    # KissFFT, MFCC, mel filterbank data
│   ├── convert_model.py         # Optional .tflite -> model_data.h converter
│   └── README_ESP32.md          # Hardware and flashing guide
├── .gitignore
└── README.md
```

`backend/venv`, Python caches, runtime recordings, local secrets, and downloaded firmware ZIPs are intentionally excluded from Git.

## Requirements

### Backend and dashboard

- macOS, Linux, or Windows
- Python 3.10+ recommended
- Internet access on first ASR model startup
- A browser such as Chrome, Edge, Safari, or Firefox

### Hardware

- ESP32-S3 development board
- INMP441 I2S microphone
- SSD1306 I2C OLED display
- Arduino IDE with ESP32 board support

## Start the Backend

From the repository root:

```bash
cd backend
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend exposes:

- `ws://localhost:8000/ws/esp32`: ESP32 audio and telemetry input
- `ws://localhost:8000/ws/frontend`: browser dashboard WebSocket
- `http://localhost:8000/api/recordings`: recording metadata
- `http://localhost:8000/api/recordings/{filename}`: WAV playback/download
- `http://localhost:8000/health`: backend and ESP32 status

The first startup downloads and caches the configured faster-whisper model. Keep the terminal running while using the dashboard.

### Optional: activate the virtual environment

```bash
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The virtual environment is local and must not be committed. `requirements.txt` is the reproducible dependency definition.

## Start the Dashboard

In a second terminal, from the repository root:

```bash
python3 -m http.server 5173 --directory frontend
```

Open:

```text
http://localhost:5173
```

The dashboard automatically connects to the backend on port `8000`. It refreshes recordings over WebSocket events and periodic API synchronization.

Opening `frontend/index.html` directly may also work, but a local HTTP server is recommended for consistent WebSocket and browser behavior.

## Whisper Configuration

The ASR wrapper is in `backend/asr.py`:

```python
MODEL_NAME = "base"
COMPUTE_TYPE = "int8"
DEVICE = "cpu"
LANGUAGE = None
```

Current behavior:

- `base`: multilingual Whisper model
- `int8`: lower memory and CPU-friendly inference
- `cpu`: works on Apple Silicon without requiring CUDA
- `LANGUAGE = None`: automatic language detection
- `vad_filter=True`: removes non-speech sections
- `beam_size=5`: decoding quality/speed tradeoff

For English-only recognition, set `LANGUAGE = "en"`. For a fixed language, use its Whisper language code. Restart the backend after changing the model or ASR configuration.

## ESP32 Configuration

Copy the local secrets template:

```bash
cp esp32/VAANI_JAGRAN_ASR/secrets.example.h \
   esp32/VAANI_JAGRAN_ASR/secrets.h
```

Edit `secrets.h`:

```cpp
#define WIFI_SSID "your_wifi_name"
#define WIFI_PASS "your_wifi_password"
#define WS_HOST "your_backend_ip"
```

`secrets.h` is ignored by Git. Never commit real WiFi credentials or a private backend address.

The backend IP must be reachable from the ESP32. `localhost` will not work on the ESP32 because it refers to the ESP32 itself.

For hardware wiring, Arduino settings, library installation, and flashing, read [esp32/README_ESP32.md](esp32/README_ESP32.md).

## Audio and WebSocket Contract

The firmware sends:

- PCM format: signed `int16`
- Sample rate: `16,000 Hz`
- Channels: mono
- Hop size: `320 samples` / `20 ms`
- Binary chunk size: `640 bytes`
- Endpoint: `/ws/esp32`

Control messages are JSON. Important messages include `start_stream`, `end_stream`, and `telemetry`. The backend forwards status, transcript, audio-level, and recording events to `/ws/frontend`.

## Recording Behavior

- One keyword event becomes one WAV recording.
- The backend saves on silence endpoint, timeout, or disconnect finalization.
- The backend does not save an intermediate 10-second chunk.
- Deleted WAV files are removed from the in-memory recording list during API/status synchronization.
- Runtime WAV files are stored under `backend/recordings/` and ignored by Git.

## Model Data

The firmware currently includes the generated model header:

```text
esp32/VAANI_JAGRAN_ASR/model_data.h
```

To replace the embedded model, provide a compatible `.tflite` file and run:

```bash
python3 esp32/convert_model.py path/to/model.tflite
```

The converter generates `model_data.h` with the symbols expected by the firmware: `g_model` and `g_model_len`.

## Troubleshooting

### Backend imports fail

Run commands from `backend/`, or use the direct virtual-environment executable:

```bash
./backend/venv/bin/uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Then verify:

```bash
curl http://localhost:8000/health
```

### Dashboard says server offline

Confirm the backend is running on port `8000`, then reload `http://localhost:5173`.

### ESP32 does not connect

Check that:

- `secrets.h` exists and contains correct values.
- `WS_HOST` is the computer's LAN IP, not `localhost`.
- The ESP32 and computer are on the same network.
- Port `8000` is reachable through the computer firewall.
- The backend is running before the ESP32 attempts to connect.

### Whisper is slow

Keep `MODEL_NAME = "base"`, `COMPUTE_TYPE = "int8"`, and `DEVICE = "cpu"` on an M1 Mac. Larger models improve accuracy but require more memory and processing time.

## GitHub Hygiene

Commit application source, firmware source, headers, documentation, and `requirements.txt`. Do not commit:

- `backend/venv/`
- `__pycache__/` or `*.pyc`
- `backend/recordings/*.wav`
- `esp32/VAANI_JAGRAN_ASR/secrets.h`
- WiFi passwords, API keys, or private local IP configuration
- Local editor settings or downloaded library archives
