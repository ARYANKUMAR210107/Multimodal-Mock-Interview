"""
faster-whisper Transcription Service
=====================================
Reusable transcription service using faster-whisper (CPU, int8, small model).
Loads the model once and reuses it across all transcription requests.

Audio is decoded with Librosa into a numpy array and passed directly to
WhisperModel.transcribe(), avoiding PyAV/av.open() compatibility issues.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional

import librosa
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Singleton faster-whisper transcription service.

    The Whisper model is loaded exactly once (first instantiation) and
    reused for every subsequent transcription request. Configuration is
    read from environment variables with safe CPU/int8 defaults.
    """

    _instance: Optional["TranscriptionService"] = None
    _model: Optional[WhisperModel] = None

    def __new__(cls) -> "TranscriptionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._model is None:
            model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
            device = os.getenv("WHISPER_DEVICE", "cpu")
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            logger.info(
                "Loading faster-whisper model: %s on %s with %s",
                model_size, device, compute_type,
            )
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            logger.info(
                "faster-whisper model loaded successfully: %s (%s/%s)",
                model_size, device, compute_type,
            )

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe a mono numpy audio array with faster-whisper."""
        if audio_array is None or len(audio_array) == 0:
            return {"transcript": "", "detected_language": "unknown", "language_probability": 0.0}
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
        segments, info = self._model.transcribe(
            audio_array.astype(np.float32),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        transcript = " ".join(segment.text for segment in segments).strip()
        return {
            "transcript": transcript,
            "detected_language": getattr(info, "language", "unknown"),
            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        }

    def transcribe(
        self,
        audio_bytes: bytes,
        file_ext: str = ".wav",
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio bytes using faster-whisper.

        Args:
            audio_bytes: Raw audio file bytes (WAV, MP3, M4A, WebM, etc.)
            file_ext: File extension for temp file creation.
            language: Optional language hint (e.g. "en", "hi"). None = auto-detect.

        Returns:
            Dict with keys: transcript (str), detected_language (str),
            language_probability (float).
        """
        tmp_path: Optional[str] = None
        try:
            logger.info(
                "whisper_transcribe_started", file_ext=file_ext,
                audio_bytes=len(audio_bytes), model="faster-whisper-small"
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            audio_array, sample_rate = librosa.load(tmp_path, sr=16000, mono=True)
            logger.info(
                "whisper_audio_decoded", samples=len(audio_array),
                sample_rate=sample_rate, duration_sec=round(len(audio_array) / max(sample_rate, 1), 2)
            )

            result = self.transcribe_array(audio_array, sample_rate=16000, language=language)
            transcript = result.get("transcript", "")
            logger.info(
                "whisper_transcribe_complete", transcript_length=len(transcript),
                detected_language=result.get("detected_language"),
                language_probability=result.get("language_probability", 0.0),
                transcript_preview=transcript[:120],
            )
            return result
        except Exception as exc:
            logger.error("faster-whisper transcription failed: %s", type(exc).__name__, exc_info=True)
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# Module-level singleton instance
transcription_service = TranscriptionService()
