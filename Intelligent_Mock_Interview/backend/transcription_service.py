"""
faster-whisper Transcription Service
=====================================
Reusable transcription service using faster-whisper (CPU, int8, small model).
Loads the model once and reuses it across all transcription requests.

Supports both in-process WhisperModel execution and isolated out-of-process
worker execution for hardened OS / Application Control policy environments.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import librosa
import numpy as np

try:
    from faster_whisper import WhisperModel
except (ImportError, Exception):
    WhisperModel = None

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Singleton faster-whisper transcription service.

    The Whisper model is lazy-loaded on first transcription request and
    reused for subsequent requests. Configuration is read from environment
    variables with safe CPU/int8 defaults and automatic fallback.
    """

    _instance: Optional["TranscriptionService"] = None
    _model: Optional[WhisperModel] = None

    def __new__(cls) -> "TranscriptionService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_model(self) -> Optional[WhisperModel]:
        if self._model is not None:
            return self._model

        if WhisperModel is None:
            return None

        model_size = os.getenv("WHISPER_MODEL_SIZE", "base.en")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

        candidate_models = [model_size]
        for fallback in ("base.en", "tiny.en", "base", "small"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for candidate in candidate_models:
            try:
                logger.info("Loading faster-whisper model: %s on %s with %s", candidate, device, compute_type)
                self._model = WhisperModel(candidate, device=device, compute_type=compute_type)
                logger.info("faster-whisper model loaded successfully: %s (%s/%s)", candidate, device, compute_type)
                return self._model
            except Exception as exc:
                last_error = exc
                logger.warning("Could not load faster-whisper model %s: %s; trying fallback.", candidate, exc)

        return None

    def _transcribe_subprocess(self, audio_path: str, language: Optional[str] = None, prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
        candidate_pythons = [
            os.getenv("WHISPER_PYTHON_PATH", ""),
            r"C:\Users\babul\Desktop\Botbro-Local-AI-Desktop-Orchestration-System\activate\Scripts\python.exe",
            sys.executable,
        ]
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        worker_path = os.path.join(base_dir, "src", "whisper_worker.py")
        if not os.path.exists(worker_path):
            return None

        model_size = os.getenv("WHISPER_MODEL_SIZE", "small.en")
        for py_exe in candidate_pythons:
            if py_exe and os.path.exists(py_exe):
                try:
                    cmd = [py_exe, worker_path, "--audio", audio_path, "--model", model_size]
                    if language:
                        cmd.extend(["--language", language])
                    if prompt:
                        cmd.extend(["--prompt", prompt])
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                    if proc.returncode == 0 and "RESULT_JSON:" in proc.stdout:
                        line = [l for l in proc.stdout.splitlines() if "RESULT_JSON:" in l][0]
                        payload = json.loads(line.split("RESULT_JSON:", 1)[1])
                        if payload.get("status") == "ok":
                            return {
                                "transcript": payload.get("transcript", ""),
                                "detected_language": payload.get("detected_language", "unknown"),
                                "language_probability": payload.get("language_probability", 0.0),
                            }
                except Exception as e:
                    logger.warning("Subprocess whisper transcription with %s failed: %s", py_exe, e)
        return None

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
        
        model = self._ensure_model()
        if model is not None:
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
            segments, info = model.transcribe(
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

        # In-process unavailable: write to temporary WAV and execute via worker
        import soundfile as sf
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_wav = tmp.name
        try:
            sf.write(tmp_wav, audio_array, sample_rate)
            res = self._transcribe_subprocess(tmp_wav, language=language)
            return res or {"transcript": "", "detected_language": "unknown", "language_probability": 0.0}
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass

    def transcribe(
        self,
        audio_bytes: bytes,
        file_ext: str = ".wav",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio bytes using faster-whisper.

        Args:
            audio_bytes: Raw audio file bytes (WAV, MP3, M4A, WebM, etc.)
            file_ext: File extension for temp file creation.
            language: Optional language hint (e.g. "en", "hi"). None = auto-detect.
            prompt: Optional technical context/prompt to prime Whisper decoding.

        Returns:
            Dict with keys: transcript (str), detected_language (str),
            language_probability (float).
        """
        tmp_path: Optional[str] = None
        try:
            logger.info(
                "whisper_transcribe_started", file_ext=file_ext,
                audio_bytes=len(audio_bytes), model=os.getenv("WHISPER_MODEL_SIZE", "small.en")
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            result = None
            model = self._ensure_model()

            if model is not None:
                # First try decoding via Librosa into float32 array
                decoded_array = None
                sr = 16000
                try:
                    decoded_array, sr = librosa.load(tmp_path, sr=16000, mono=True)
                except Exception as decode_err:
                    logger.warning("Librosa decode failed: %s; falling back to direct WhisperModel file transcribe.", decode_err)

                if decoded_array is not None and len(decoded_array) > 0:
                    try:
                        result = self.transcribe_array(decoded_array, sample_rate=sr, language=language)
                    except Exception as arr_err:
                        logger.warning("transcribe_array failed: %s", arr_err)

                if result is None or not result.get("transcript"):
                    try:
                        segments, info = model.transcribe(tmp_path, language=language, beam_size=5, vad_filter=True, initial_prompt=prompt)
                        transcript = " ".join(segment.text for segment in segments).strip()
                        result = {
                            "transcript": transcript,
                            "detected_language": getattr(info, "language", "unknown"),
                            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
                        }
                    except Exception as in_err:
                        logger.warning("In-process model.transcribe failed: %s", in_err)

            # If in-process model was not available or did not produce a transcript, use robust subprocess bridge
            if result is None or not result.get("transcript"):
                sub_res = self._transcribe_subprocess(tmp_path, language=language, prompt=prompt)
                if sub_res and sub_res.get("transcript"):
                    result = sub_res

            if result is None:
                result = {"transcript": "", "detected_language": "unknown", "language_probability": 0.0}

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


# Global singleton instance
transcription_service = TranscriptionService()
