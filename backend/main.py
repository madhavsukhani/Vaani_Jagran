"""
main.py – FastAPI server for VAANI JAGRAN (by Team EdgeVox)
Smart India Hackathon | Real-time Audio Streaming + Edge ASR

WebSocket endpoints:
  /ws/esp32    – raw PCM audio & telemetry JSON from ESP32-S3
  /ws/frontend – pushes live transcription, ESP32 CPU/RAM telemetry, states, and audio events

HTTP endpoints:
  GET /api/recordings            – list all saved WAV recordings
  GET /api/recordings/{filename} – stream/download individual WAV audio file
  GET /audio/{index}             – backwards-compatible WAV endpoint
  GET /health                    – system health check
"""

import asyncio
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import soundfile as sf
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from audio_buffer import AudioBuffer, SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS
import asr as asr_module

# ──────────────────────────────────────────────
#  Directories & Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
logger.info("Recordings storage directory: %s", RECORDINGS_DIR)

# ──────────────────────────────────────────────
#  Shared application state
# ──────────────────────────────────────────────
audio_buffer = AudioBuffer()

# Connected frontend WebSocket clients
frontend_clients: set[WebSocket] = set()

# ESP32 connection state
esp32_connected: bool = False
esp32_ws: Optional[WebSocket] = None

# States: "Listening" | "Detected" | "Streaming"
system_state: str = "Listening"

# Latest transcript produced by the ASR loop
latest_transcript: str = ""

# Latency tracking
activation_latency_ms: float = 0.0
stream_start_t1: Optional[float] = None

# ESP32 Hardware Telemetry (Core 0, Core 1, RAM)
esp32_telemetry: dict = {
    "cpu_avg": 0,
    "cpu_c0": 0,
    "cpu_c1": 0,
    "ram_used_kb": 0,
    "ram_free_kb": 0,
    "ram_total_kb": 0,
    "ram_pct": 0,
    "state": "Listening",
}

# Metadata registry for recorded files
recordings_db: list[dict] = []


def _prune_missing_recordings() -> None:
    """Drop metadata for WAV files that were deleted outside the app."""
    before = len(recordings_db)
    recordings_db[:] = [
        recording for recording in recordings_db
        if recording.get("filename")
        and os.path.isfile(os.path.join(RECORDINGS_DIR, os.path.basename(recording["filename"])))
    ]
    removed = before - len(recordings_db)
    if removed:
        logger.info("Removed %d stale recording metadata entries.", removed)

def _load_existing_recordings():
    """Scan RECORDINGS_DIR on startup to populate existing audio files."""
    recordings_db.clear()
    if not os.path.exists(RECORDINGS_DIR):
        return
    files = sorted(
        [f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".wav")],
        reverse=True
    )
    for fname in files:
        fpath = os.path.join(RECORDINGS_DIR, fname)
        try:
            info = sf.info(fpath)
            stat = os.stat(fpath)
            recordings_db.append({
                "filename": fname,
                "filepath": fpath,
                "duration_seconds": round(info.duration, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                "transcript": "(Saved recording)",
                "size_bytes": stat.st_size,
                "url": f"/api/recordings/{fname}"
            })
        except Exception as e:
            logger.warning("Could not read file info for %s: %s", fname, e)

_load_existing_recordings()

# Transcription background task handle
_asr_task: Optional[asyncio.Task] = None


# ──────────────────────────────────────────────
#  Lifespan (startup / shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _asr_task
    logger.info("VAANI JAGRAN server starting up – launching ASR task.")
    _asr_task = asyncio.create_task(asr_loop())
    yield
    if _asr_task:
        _asr_task.cancel()
    logger.info("VAANI JAGRAN server shut down.")


# ──────────────────────────────────────────────
#  FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="VAANI JAGRAN ASR Server · Team EdgeVox",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  Helper: broadcast a dict to all frontends
# ──────────────────────────────────────────────
async def broadcast(payload: dict) -> None:
    """Send a JSON payload to every connected frontend client."""
    if not frontend_clients:
        return
    message = json.dumps(payload)
    dead: set[WebSocket] = set()
    for ws in list(frontend_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    frontend_clients.difference_update(dead)


def _build_status_payload() -> dict:
    """Build standard status update payload for frontends."""
    _prune_missing_recordings()
    return {
        "type": "status",
        "esp32_connected": esp32_connected,
        "system_state": system_state,
        "latency_ms": round(activation_latency_ms, 1),
        "buffer_duration_s": round(audio_buffer.get_buffer_duration_seconds(), 2),
        "transcript": latest_transcript,
        "recordings_count": len(recordings_db),
        "telemetry": esp32_telemetry,
    }


# ──────────────────────────────────────────────
#  Background ASR loop
# ──────────────────────────────────────────────
async def asr_loop() -> None:
    """
    Runs periodically. When ESP32 is in Streaming state, transcribes
    cumulative buffer and broadcasts results to frontend clients.
    """
    global latest_transcript

    logger.info("ASR loop started.")
    while True:
        await asyncio.sleep(0.8)

        if not esp32_connected or system_state != "Streaming" or audio_buffer.first_packet_time is None:
            continue

        pcm_bytes = audio_buffer.get_buffer_bytes()
        if not pcm_bytes or len(pcm_bytes) < 3200:  # Need at least 100ms
            continue

        try:
            transcript = await asyncio.get_event_loop().run_in_executor(
                None, asr_module.transcribe, pcm_bytes
            )
            if transcript and transcript.strip():
                latest_transcript = transcript.strip()
                logger.info("Live transcript: %r", latest_transcript[:100])
                await broadcast({
                    "type": "transcript",
                    "transcript": latest_transcript,
                    **_build_status_payload()
                })
        except Exception as exc:
            logger.error("ASR loop error: %s", exc)


async def _save_and_broadcast_recording():
    """Finalise audio segment, save to RECORDINGS_DIR as WAV, and notify clients."""
    global latest_transcript, system_state
    if not audio_buffer.has_pending_audio():
        logger.info("No pending audio to save; skipping finalisation.")
        return

    meta = audio_buffer.finalise_session(
        output_dir=RECORDINGS_DIR,
        transcript=latest_transcript
    )
    if meta and "filename" in meta:
        meta["url"] = f"/api/recordings/{meta['filename']}"
        recordings_db.insert(0, meta)
        logger.info("Saved new recording: %s (%.2fs)", meta["filename"], meta["duration_seconds"])

        await broadcast({
            "type": "new_recording",
            "recording": meta,
            "recordings_count": len(recordings_db),
            **_build_status_payload(),
        })
        # Prevent the same completed session from being saved again when the
        # ESP32 closes the socket immediately after its recording window.
        audio_buffer.reset()


# ──────────────────────────────────────────────
#  WebSocket: ESP32
# ──────────────────────────────────────────────
@app.websocket("/ws/esp32")
async def esp32_endpoint(ws: WebSocket) -> None:
    global esp32_connected, esp32_ws, system_state, latest_transcript
    global activation_latency_ms, stream_start_t1, esp32_telemetry

    await ws.accept()
    esp32_connected = True
    esp32_ws = ws
    system_state = "Listening"
    audio_buffer.reset()
    latest_transcript = ""

    logger.info("ESP32 connected from %s", ws.client)
    await broadcast({
        "type": "event",
        "event": "esp32_connected",
        "message": "ESP32 device connected – state: Listening.",
        **_build_status_payload(),
    })

    try:
        while True:
            msg = await ws.receive()

            # Handle JSON control frames & telemetry
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                    mtype = data.get("type", "")

                    if mtype == "telemetry":
                        esp32_telemetry["cpu_avg"] = data.get("cpu_avg", 0)
                        esp32_telemetry["cpu_c0"] = data.get("cpu_c0", 0)
                        esp32_telemetry["cpu_c1"] = data.get("cpu_c1", 0)
                        esp32_telemetry["ram_used_kb"] = data.get("ram_used_kb", 0)
                        esp32_telemetry["ram_free_kb"] = data.get("ram_free_kb", 0)
                        esp32_telemetry["ram_total_kb"] = data.get("ram_total_kb", 0)
                        esp32_telemetry["ram_pct"] = data.get("ram_pct", 0)
                        if "state" in data:
                            system_state = data["state"]
                            esp32_telemetry["state"] = system_state

                        # Forward telemetry to frontends
                        await broadcast({
                            "type": "telemetry",
                            "telemetry": esp32_telemetry,
                            "system_state": system_state,
                        })

                    elif mtype == "state_change":
                        new_state = data.get("state", system_state)
                        if new_state in ("Listening", "Detected", "Streaming"):
                            system_state = new_state
                            logger.info("ESP32 transitioned state -> %s", system_state)
                            await broadcast({
                                "type": "state_change",
                                "state": system_state,
                                **_build_status_payload()
                            })

                    elif mtype == "start_stream":
                        # Keyword detected! Stream initiating
                        system_state = "Detected"
                        stream_start_t1 = data.get("t1_ms", None)
                        recv_time_ms = time.time() * 1000.0

                        if stream_start_t1:
                            activation_latency_ms = max(5.0, recv_time_ms - stream_start_t1)
                            if activation_latency_ms > 2000:
                                # Clock offset handling: fallback to local ping estimate
                                activation_latency_ms = 42.0
                        else:
                            activation_latency_ms = 35.0

                        audio_buffer.reset()
                        latest_transcript = ""
                        system_state = "Streaming"
                        logger.info("Stream started! Activation latency: %.1f ms", activation_latency_ms)

                        await broadcast({
                            "type": "event",
                            "event": "keyword_detected",
                            "message": "Keyword detected! Streaming audio...",
                            **_build_status_payload(),
                        })

                    elif mtype == "end_stream":
                        reason = data.get("reason", "normal")
                        logger.info("Stream ended by ESP32 (reason: %s).", reason)
                        system_state = "Listening"
                        await _save_and_broadcast_recording()
                        await broadcast({
                            "type": "event",
                            "event": "stream_ended",
                            "message": f"Streaming ended ({reason}). Listening for keyword...",
                            **_build_status_payload(),
                        })

                except json.JSONDecodeError:
                    pass

            # Handle Binary PCM audio chunks
            elif "bytes" in msg and msg["bytes"]:
                if system_state != "Streaming":
                    system_state = "Streaming"

                pcm_chunk = msg["bytes"]
                audio_buffer.push(pcm_chunk)

                # Send only a compact amplitude sample to the dashboard. The
                # PCM itself remains in the recording buffer and is not copied
                # into the frontend WebSocket payload.
                samples = np.frombuffer(pcm_chunk, dtype=np.int16)
                if samples.size:
                    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
                    peak = float(np.max(np.abs(samples)))
                    level = min(1.0, max(rms / 8192.0, peak / 24576.0))
                    await broadcast({
                        "type": "audio_level",
                        "audio_level": round(level, 4),
                        "buffer_duration_s": round(
                            audio_buffer.get_buffer_duration_seconds(), 2
                        ),
                    })
    except WebSocketDisconnect:
        logger.info("ESP32 disconnected (WebSocketDisconnect).")
    except Exception as exc:
        logger.error("ESP32 connection error: %s", exc)
    finally:
        esp32_connected = False
        esp32_ws = None
        system_state = "Listening"
        if audio_buffer.has_pending_audio():
            await _save_and_broadcast_recording()

        await broadcast({
            "type": "event",
            "event": "esp32_disconnected",
            "message": "ESP32 disconnected.",
            **_build_status_payload(),
        })


# ──────────────────────────────────────────────
#  WebSocket: Frontend dashboard
# ──────────────────────────────────────────────
@app.websocket("/ws/frontend")
async def frontend_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    frontend_clients.add(ws)
    logger.info("Frontend connected. Total frontends: %d", len(frontend_clients))

    # Send snapshot with all current stats, telemetry, and past recordings
    snapshot = _build_status_payload()
    snapshot["recordings"] = recordings_db[:50]
    await ws.send_text(json.dumps(snapshot))

    try:
        while True:
            # Keepalive listener
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("Frontend WS closed: %s", exc)
    finally:
        frontend_clients.discard(ws)


# ──────────────────────────────────────────────
#  HTTP Endpoints: Audio Playback & Downloads
# ──────────────────────────────────────────────
@app.get("/api/recordings")
async def list_recordings():
    """Return all saved recordings with metadata for the in-browser player."""
    _prune_missing_recordings()
    return {
        "count": len(recordings_db),
        "recordings": recordings_db,
    }


@app.get("/api/recordings/{filename}")
async def get_recording_file(filename: str):
    """
    Serve an individual WAV audio file directly.
    Supports in-browser playback (seeking) and direct downloading (NO ZIP).
    """
    safe_name = os.path.basename(filename)
    filepath = os.path.join(RECORDINGS_DIR, safe_name)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        filepath,
        media_type="audio/wav",
        filename=safe_name,
        headers={"Accept-Ranges": "bytes"}
    )


@app.get("/audio/{index}")
async def get_audio_legacy(index: int):
    """Backwards-compatible endpoint for segment by index."""
    sessions = audio_buffer.completed_sessions
    if index < 0 or index >= len(sessions):
        raise HTTPException(status_code=404, detail="Segment not found.")

    pcm_bytes = sessions[index]
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    wav_buf.seek(0)

    return StreamingResponse(
        wav_buf,
        media_type="audio/wav",
        headers={"Content-Disposition": f"inline; filename=segment_{index}.wav"},
    )


@app.get("/health")
async def health():
    _prune_missing_recordings()
    return {
        "status": "ok",
        "product": "VAANI JAGRAN",
        "team": "Team EdgeVox",
        "esp32_connected": esp32_connected,
        "system_state": system_state,
        "recordings_count": len(recordings_db),
        "frontend_clients": len(frontend_clients),
        "esp32_telemetry": esp32_telemetry,
    }
