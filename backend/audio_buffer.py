"""
audio_buffer.py – Rolling audio buffer for a single ESP32 session.

Responsibilities:
  - Accept 20 ms PCM chunks and append them to an in-memory buffer
  - Enforce a maximum buffer duration (default: 15 seconds)
  - Track session timing and first-packet receipt time
  - Provide the current buffer as bytes for the ASR module
  - Save completed audio bursts to disk as standard WAV files
"""

import logging
import os
import time
from collections import deque
from typing import Optional

import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

# Audio parameters (must match ESP32 and asr.py)
SAMPLE_RATE  = 16_000   # Hz
SAMPLE_WIDTH = 2        # bytes per sample (16-bit PCM)
CHANNELS     = 1

# How many bytes represent 1 second of audio
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS   # 32 000

# ESP32 sends one burst per keyword event. Keep enough room for the 2 s
# pre-roll plus the 10 s live-stream limit as one recording.
CHUNK_DURATION_SECONDS = 10
MAX_BUFFER_SECONDS     = 15   # rolling window kept in memory


class AudioBuffer:
    """
    Thread-safe rolling buffer for an ESP32 streaming session.
    Chunks are stored as a deque of bytes objects so trimming is O(1).
    """

    def __init__(self, chunk_duration: int = CHUNK_DURATION_SECONDS):
        self._chunks: deque[bytes] = deque()
        self._total_bytes: int = 0
        self._max_bytes: int = MAX_BUFFER_SECONDS * BYTES_PER_SECOND
        self._chunk_bytes: int = chunk_duration * BYTES_PER_SECOND

        # Timing
        self.session_start_time: Optional[float] = None
        self.first_packet_time:  Optional[float] = None

        # Completed sessions (bytes objects ready for playback)
        self.completed_sessions: list[bytes] = []

        # Current chunk accumulation
        self._current_chunk_bytes: int = 0
        self._current_chunk_buf: bytearray = bytearray()

        logger.info(
            "AudioBuffer initialised – chunk=%ds max=%ds",
            chunk_duration, MAX_BUFFER_SECONDS,
        )

    def push(self, pcm_data: bytes) -> bool:
        """
        Append a raw PCM chunk.
        Returns True if the current 10-second chunk limit was reached.
        """
        now = time.monotonic()
        if self.first_packet_time is None:
            self.first_packet_time = now
            self.session_start_time = now
            logger.info("First audio packet received – session started.")

        self._chunks.append(pcm_data)
        self._total_bytes += len(pcm_data)
        self._current_chunk_buf.extend(pcm_data)
        self._current_chunk_bytes += len(pcm_data)

        # Enforce rolling max window
        while self._total_bytes > self._max_bytes and self._chunks:
            dropped = self._chunks.popleft()
            self._total_bytes -= len(dropped)

        # Do not split a keyword event at 10 s. The ESP32 sends end_stream
        # after its hard limit, and the complete burst must become one WAV.
        return False

    def get_buffer_bytes(self) -> bytes:
        """Return the full rolling buffer as a single bytes object for ASR."""
        return b"".join(self._chunks)

    def get_elapsed_ms(self) -> float:
        """Milliseconds since the first audio packet arrived."""
        if self.first_packet_time is None:
            return 0.0
        return (time.monotonic() - self.first_packet_time) * 1000.0

    def get_buffer_duration_seconds(self) -> float:
        """How many seconds of audio are currently in the rolling buffer."""
        return self._total_bytes / BYTES_PER_SECOND

    def has_pending_audio(self) -> bool:
        """Return whether the current session contains audio to save."""
        return self._total_bytes > 0 or bool(self._current_chunk_buf)

    def finalise_session(self, output_dir: Optional[str] = None, transcript: str = "") -> Optional[dict]:
        """
        Called when a streaming burst finishes (silence endpoint or 10s max).
        Saves the audio as a WAV file if output_dir is given.
        Returns a metadata dict.
        """
        if self._total_bytes <= 0:
            return None

        audio_bytes = self.get_buffer_bytes()
        self.completed_sessions.append(audio_bytes)
        duration_sec = round(len(audio_bytes) / BYTES_PER_SECOND, 2)

        meta = {
            "index": len(self.completed_sessions) - 1,
            "duration_seconds": duration_sec,
            "transcript": transcript,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "size_bytes": len(audio_bytes)
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            micros = time.time_ns() // 1000 % 1_000_000
            filename = f"audio_{timestamp}_{micros:06d}.wav"
            filepath = os.path.join(output_dir, filename)

            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sf.write(filepath, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            meta["filename"] = filename
            meta["filepath"] = filepath
            logger.info("Saved WAV recording to: %s", filepath)

        return meta

    def reset(self) -> None:
        """Reset for a brand-new connection or burst."""
        self._chunks.clear()
        self._total_bytes = 0
        self._current_chunk_buf = bytearray()
        self._current_chunk_bytes = 0
        self.completed_sessions.clear()
        self.session_start_time = None
        self.first_packet_time = None
        logger.info("AudioBuffer reset for new burst.")

    def _finalise_chunk(self, force: bool = False) -> None:
        """Move the current chunk into completed_sessions and reset counters."""
        if not self._current_chunk_buf:
            return
        audio_bytes = bytes(self._current_chunk_buf)
        self.completed_sessions.append(audio_bytes)
        logger.info(
            "Audio chunk #%d finalised – %.2f s (force=%s).",
            len(self.completed_sessions),
            len(audio_bytes) / BYTES_PER_SECOND,
            force,
        )
        self._current_chunk_buf = bytearray()
        self._current_chunk_bytes = 0
