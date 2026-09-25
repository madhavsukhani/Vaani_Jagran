"""
asr.py – ASR (Automatic Speech Recognition) module using faster-whisper.

Responsibilities:
  - Load and cache the Whisper model on startup
  - Expose a single transcribe() function that accepts raw PCM bytes
    and returns a plain text transcript

To swap to a larger model later, change MODEL_NAME and/or COMPUTE_TYPE.
"""

import io
import logging
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Model configuration – easy to change later
# ──────────────────────────────────────────────
MODEL_NAME   = "base"       # swap to "large-v3-turbo" when ready
COMPUTE_TYPE = "int8"       # "int8" for CPU-friendly inference
DEVICE       = "cpu"        # "cuda" if a GPU is available
LANGUAGE     = None         # auto-detect Hindi, English, or other languages

# Audio stream parameters (must match ESP32 config)
SAMPLE_RATE  = 16_000       # 16 kHz
CHANNELS     = 1            # mono
SAMPLE_WIDTH = 2            # 16-bit PCM → 2 bytes per sample


def _load_model() -> WhisperModel:
    """Load the faster-whisper model once at import time."""
    logger.info(
        "Loading Whisper model '%s' (compute_type=%s, device=%s) …",
        MODEL_NAME, COMPUTE_TYPE, DEVICE,
    )
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Whisper model loaded successfully.")
    return model


# Module-level singleton – loaded once, reused for every transcription call
_model: WhisperModel = _load_model()


def transcribe(pcm_bytes: bytes) -> str:
    """
    Transcribe raw 16-bit PCM audio bytes.

    Args:
        pcm_bytes: Raw PCM audio data (16-bit, 16 kHz, mono).

    Returns:
        Transcribed text as a single string, or an empty string if the
        audio contained no recognisable speech.
    """
    if not pcm_bytes:
        return ""

    try:
        # Convert raw PCM bytes → float32 numpy array in [-1.0, 1.0]
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        audio_np /= 32768.0   # normalise 16-bit range

        if audio_np.size == 0:
            return ""

        # Write to an in-memory WAV so faster-whisper can read it
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_np, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)

        # Run inference
        segments, info = _model.transcribe(
            wav_buffer,
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,          # suppress non-speech segments
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        transcript = " ".join(seg.text.strip() for seg in segments)
        logger.debug(
            "Transcribed %.2f s of audio → %d chars",
            info.duration, len(transcript),
        )
        return transcript.strip()

    except Exception as exc:
        logger.error("Transcription error: %s", exc, exc_info=True)
        return ""
