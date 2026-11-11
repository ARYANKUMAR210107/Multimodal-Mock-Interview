"""
=============================================================================
FastAPI Backend Server
Intelligent Multimodal Mock Interview Assessment System
=============================================================================
"""

import os, sys, tempfile, logging, math, re, json, time, uuid, hashlib, io, wave, base64
import numbers
from contextvars import ContextVar
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from backend.transcription_service import transcription_service

# Add project root and src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)

for env_path in (os.path.join(ROOT_DIR, ".env"), os.path.join(BASE_DIR, ".env")):
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from src.text_pipeline import (
    InterviewTextAssessor,
    calculate_text_assessment_score,
    parse_instruction_parquet,
    clean_text,
)
from src.audio_feature_extractor import CandidateAudioAnalyzer
from src.database import initialize_database
from backend.management_api import persist_assessment, router as management_router, _json_safe

# Initialize FastAPI application
app = FastAPI(
    title="Intelligent Multimodal Mock Interview Assessment API",
    description="Backend ML & NLP Inference Service for Employability Prediction, Text Assessment, and Audio Analysis.",
    version="1.0.0"
)
app.include_router(management_router)
initialize_database()

logger = logging.getLogger(__name__)
REQUEST_ID = ContextVar("interviewai_request_id", default="-")
MAX_AUDIO_BYTES = 25 * 1024 * 1024
TRANSCRIPTION_CHUNK_TARGET_SEC = float(os.getenv("TRANSCRIPTION_CHUNK_TARGET_SEC", "4.0"))
TRANSCRIPTION_FINAL_MAX_SEC = float(os.getenv("TRANSCRIPTION_FINAL_MAX_SEC", "300.0"))
AUDIO_FORMATS = {
    ".wav": ("audio/wav", "audio/x-wav", "audio/wave"),
    ".mp3": ("audio/mpeg", "audio/mp3", "audio/x-mp3"),
    ".m4a": ("audio/mp4", "audio/x-m4a", "audio/m4a", "audio/aac"),
    ".webm": ("audio/webm", "video/webm"),
}


def _log_event(event: str, level: int = logging.INFO, **fields) -> None:
    """Write a structured, content-safe event for one API operation."""
    entry = {
        "event": event,
        "request_id": REQUEST_ID.get(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    logger.log(level, json.dumps(entry, separators=(",", ":"), default=str))


_log_event("transcription_config", provider="faster-whisper", model=os.getenv("WHISPER_MODEL_SIZE", "small"), device=os.getenv("WHISPER_DEVICE", "cpu"), compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"))


@app.middleware("http")
async def request_logging_middleware(request, call_next):
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9._-]{1,80}", supplied_id) else uuid.uuid4().hex
    token = REQUEST_ID.set(request_id)
    started = time.perf_counter()
    _log_event("request_started", method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        level = logging.ERROR if response.status_code >= 500 else logging.WARNING if response.status_code >= 400 else logging.INFO
        _log_event("request_completed", level=level, method=request.method, path=request.url.path,
                   status_code=response.status_code, elapsed_ms=elapsed_ms)
        return response
    except Exception as exc:
        _log_event("request_failed", level=logging.ERROR, method=request.method, path=request.url.path,
                   error_type=type(exc).__name__, elapsed_ms=round((time.perf_counter() - started) * 1000, 1))
        raise
    finally:
        REQUEST_ID.reset(token)


def _validate_audio_bytes(filename: str, content_type: str, data: bytes) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio must be 25 MB or smaller.")

    # Check magic container signatures
    is_wav = data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WAVE"
    is_mp3 = data.startswith(b"ID3") or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)
    is_m4a = len(data) > 12 and data[4:8] == b"ftyp"
    is_webm = data.startswith(b"\x1a\x45\xdf\xa3")

    # If browser recorded WebM but labeled as .wav, auto-align:
    if is_webm and ext in (".wav", ""):
        ext = ".webm"
    elif is_wav and ext in (".webm", ""):
        ext = ".wav"

    if ext not in AUDIO_FORMATS:
        raise HTTPException(status_code=415, detail="Supported audio formats are WAV, MP3, M4A, and WebM.")

    valid_signature = {
        ".wav": is_wav,
        ".mp3": is_mp3,
        ".m4a": is_m4a,
        ".webm": is_webm,
    }.get(ext, False)

    if not valid_signature:
        raise HTTPException(status_code=415, detail="The file content does not match its audio extension.")

    mime = content_type.split(";", 1)[0].strip().lower()
    # Normalize browser mismatched mime types
    if ext == ".webm" and mime in ("audio/wav", "application/octet-stream", "audio/webm"):
        mime = "audio/webm"
    elif ext == ".wav" and mime in ("audio/webm", "application/octet-stream", "audio/wav"):
        mime = "audio/wav"

    if mime and mime not in AUDIO_FORMATS[ext] and mime != "application/octet-stream":
        raise HTTPException(status_code=415, detail="The audio content type does not match the file extension.")
    return ext


def _filler_word_count(transcript: str) -> int:
    import re
    matches = re.findall(r"\b(?:um+|uh+|erm+|ah+|like|you know|i mean)\b", transcript.lower())
    return len(matches)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _valid_unit_score(value) -> Optional[float]:
    """Accept only finite numeric scores in [0, 1]; never coerce bad values to zero."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number_value = float(value)
    if not math.isfinite(number_value) or not 0.0 <= number_value <= 1.0:
        return None
    return number_value


def _safe_remove_temp(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        _log_event("temporary_file_cleanup_failed", level=logging.WARNING,
                   error_type=type(exc).__name__)


def _analyze_audio_bytes(filename: str, data: bytes, transcript: str, response_id: Optional[str] = None) -> dict:
    ext = os.path.splitext(filename)[1].lower()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        _log_event("librosa_processing_started", file_format=ext, audio_bytes=len(data), response_id=response_id)
        metrics = audio_analyzer.analyze_audio_file(tmp_path) or {}
        if not isinstance(metrics, dict):
            raise ValueError("Audio analyzer returned an invalid measurement payload.")
        duration = _finite_or_none(metrics.get("duration_sec"))
        if duration is None:
            waveform, sample_rate = audio_analyzer.load_audio(tmp_path)
            duration = len(waveform) / sample_rate if sample_rate else None
            metrics["duration_sec"] = duration
        if duration is None or duration < 0.25:
            raise HTTPException(status_code=422, detail="Audio is too short to analyze; record at least 0.25 seconds.")
        words = len(transcript.split())
        has_transcript = bool(transcript.strip())
        metrics["speaking_rate_wpm"] = round(words / (duration / 60.0), 1) if has_transcript else None
        metrics["transcript_word_count"] = words if has_transcript else None
        metrics["filler_word_count"] = _filler_word_count(transcript) if has_transcript else None
        for key in ("mean_pitch_hz", "pitch_std_hz", "min_pitch_hz", "max_pitch_hz", "mean_energy_rms", "pause_ratio", "pause_duration_sec", "pause_count"):
            metrics[key] = _finite_or_none(metrics.get(key))
        mfcc = metrics.get("mfcc_mean")
        if isinstance(mfcc, np.ndarray):
            mfcc = mfcc.tolist()
        elif isinstance(mfcc, tuple):
            mfcc = list(mfcc)
        elif not isinstance(mfcc, list):
            mfcc = []
        metrics["mfcc_mean"] = [_finite_or_none(v) for v in mfcc]
        for key in (
            "duration_sec", "speaking_duration_sec", "pause_ratio", "pause_duration_sec",
            "pause_count", "speaking_rate_wpm", "transcript_word_count", "filler_word_count",
            "mean_pitch_hz", "pitch_std_hz", "min_pitch_hz", "max_pitch_hz", "mean_energy_rms",
        ):
            metrics.setdefault(key, None)
        # Pitch estimates are only meaningful for voiced recordings. Do not report
        # fallback defaults from the extractor as measured values for silence.
        if metrics["mean_energy_rms"] is not None and metrics["mean_energy_rms"] < 1e-5:
            metrics["mean_pitch_hz"] = None
            metrics["pitch_std_hz"] = None
            metrics["min_pitch_hz"] = None
            metrics["max_pitch_hz"] = None
        # These measurements have no validated conversion to an audio quality
        # score. Keep the report descriptive and avoid fabricating score fields.
        metrics.pop("estimated_fluency_score", None)
        metrics.pop("estimated_prosody_score", None)
        _log_event("librosa_processing_completed", response_id=response_id,
                   duration_sec=metrics.get("duration_sec"), mfcc_count=len(metrics.get("mfcc_mean", [])))
        return metrics
    except HTTPException:
        raise
    except Exception as exc:
        _log_event("librosa_processing_failed", level=logging.WARNING, error_type=type(exc).__name__)
        raise HTTPException(status_code=422, detail="Could not decode or analyze this audio. Check the file and try again.") from exc
    finally:
        _safe_remove_temp(tmp_path)


# Enable CORS for local Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Pre-Trained Models on Startup ─────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
GB_MODEL_PATH = os.path.join(MODELS_DIR, "gradient_boosting_employability.joblib")
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, "interview_tfidf_vectorizer.joblib")
QUESTIONS_PARQUET = os.path.join(BASE_DIR, "dataset", "interview_questions", "eval-00000-of-00001.parquet")

employability_model = None
text_assessor = None
audio_analyzer = None
cached_questions = []

try:
    if os.path.exists(GB_MODEL_PATH):
        employability_model = joblib.load(GB_MODEL_PATH)
        print(f"[OK] Loaded Gradient Boosting Model from {GB_MODEL_PATH}")
    else:
        print(f"[WARNING] Model not found at {GB_MODEL_PATH}")

    if os.path.exists(TFIDF_MODEL_PATH):
        vectorizer = joblib.load(TFIDF_MODEL_PATH)
        text_assessor = InterviewTextAssessor(vectorizer=vectorizer)
        text_assessor.is_fitted = True
        print(f"[OK] Loaded TF-IDF Vectorizer from {TFIDF_MODEL_PATH}")
    else:
        text_assessor = InterviewTextAssessor()
        print(f"[WARNING] TF-IDF Vectorizer not found, initialized empty")

    audio_analyzer = CandidateAudioAnalyzer(target_sr=16000)
    print("[OK] Initialized CandidateAudioAnalyzer with Librosa")

    if os.path.exists(QUESTIONS_PARQUET):
        df_q = parse_instruction_parquet(QUESTIONS_PARQUET)
        # Select top 50 diverse questions for mock interviews
        cached_questions = df_q.head(50).to_dict(orient="records")
        print(f"[OK] Cached {len(cached_questions)} reference interview questions")
except Exception as e:
    print(f"[ERROR] Startup loading error: {e}")

_log_event(
    "backend_startup_completed",
    gradient_boosting_loaded=employability_model is not None,
    text_assessor_loaded=text_assessor is not None and getattr(text_assessor, "is_fitted", False),
    audio_analyzer_loaded=audio_analyzer is not None,
    cached_question_count=len(cached_questions),
)


# ── Pydantic Request & Response Schemas ───────────────────────────────────
class EmployabilityRequest(BaseModel):
    manner_of_speaking: int = Field(..., ge=1, le=5, description="Speaking manner & clarity (1-5)")
    mental_alertness: int = Field(..., ge=1, le=5, description="Comprehension & mental agility (1-5)")
    self_confidence: int = Field(..., ge=1, le=5, description="Poise and confidence (1-5)")
    ability_to_present_ideas: int = Field(..., ge=1, le=5, description="Structure & presentation (1-5)")
    communication_skills: int = Field(..., ge=1, le=5, description="Overall communication (1-5)")

class EmployabilityResponse(BaseModel):
    prediction: str
    probability_employable: float
    probability_less_employable: float
    confidence_score: float
    soft_skill_mean: float
    soft_skill_min: int
    recommendations: List[str]

class TextAssessmentRequest(BaseModel):
    question: str
    candidate_answer: str
    reference_answer: Optional[str] = ""
    session_id: Optional[str] = None
    position: Optional[int] = None

class TextAssessmentResponse(BaseModel):
    relevance_score: float
    question_alignment_score: float
    keyword_overlap: float
    structure_score: float
    quality_score: float
    performance_tier: str
    feedback: List[str]


class MultimodalReportResponse(BaseModel):
    """Typed report contract; audio measurements never feed the text score."""
    model_config = ConfigDict(extra="allow")

    assessment_type: str
    response_id: str
    timestamp: str
    interview_information: Dict[str, Any]
    text_assessment: Dict[str, Any]
    text_assessment_score: Optional[float]
    text_assessment_score_status: str
    text_assessment_score_formula: str
    text_assessment_score_limitation: str
    voice_assessment: Optional[Dict[str, Any]]
    voice_analysis_status: str
    overall_multimodal_assessment: Dict[str, str]


# ── API Endpoints ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    whisper_model = os.getenv("WHISPER_MODEL_SIZE", "base.en")
    return {
        "status": "online",
        "system": "Intelligent Multimodal Mock Interview Assessment System",
        "version": "1.0.0",
        "transcription_provider": "faster-whisper",
        "transcription_configured": True,
        "transcription_status": "ready",
        "transcription_model": whisper_model,
        "endpoints": [
            "/api/audio/transcribe",
            "/api/audio/tts",
            "/health",
            "/api/interview/multimodal-report",
            "/api/v1/questions",
            "/api/v1/predict/employability",
            "/api/v1/assess/text",
            "/api/v1/assess/audio",
            "/api/v1/assess/multimodal",
            "/api/v1/models/meta",
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "online",
        "version": app.version,
        "transcription_provider": "faster-whisper",
        "transcription_configured": True,
        "transcription_model": os.getenv("WHISPER_MODEL_SIZE", "base.en"),
        "whisper_device": os.getenv("WHISPER_DEVICE", "cpu"),
        "whisper_compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
    }


@app.post("/api/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    response_id: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
):
    data = await file.read(MAX_AUDIO_BYTES + 1)
    try:
        ext = _validate_audio_bytes(file.filename or "", file.content_type or "", data)
    finally:
        await file.close()

    import hashlib
    if not response_id:
        response_id = f"RESP-{hashlib.sha256(data).hexdigest()[:12].upper()}"
    if not re.fullmatch(r"RESP-[A-F0-9]{10,32}", response_id):
        raise HTTPException(status_code=400, detail="Invalid response identifier.")
    _log_event("audio_upload_received", file_format=ext, audio_bytes=len(data), response_id=response_id)

    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".m4a": "audio/m4a",
         ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    audio_mime = mime_map.get(ext, file.content_type or "audio/wav")
    if ";" in audio_mime:
        audio_mime = audio_mime.split(";")[0].strip()

    if audio_analyzer is None:
        raise HTTPException(status_code=503, detail="Audio validation is unavailable because Librosa did not load.")

    _log_event("audio_validation_started", file_format=ext, audio_bytes=len(data), response_id=response_id)
    try:
        result = await run_in_threadpool(
            transcription_service.transcribe,
            data,
            ext,
            prompt=prompt,
        )
        transcript = result.get("transcript", "")
        detected_language = result.get("detected_language", "unknown")
        language_probability = result.get("language_probability", 0.0)
        _log_event("transcription_response_ready", model="faster-whisper-small",
                   transcript_characters=len(transcript), detected_language=detected_language,
                   response_id=response_id)
        if not transcript:
            raise HTTPException(status_code=422, detail="No speech detected in the audio. Please speak clearly and retry.")
        return {
            "transcript": transcript,
            "detected_language": detected_language,
            "language_probability": language_probability,
            "model": "faster-whisper-small",
            "provider": "faster-whisper",
            "response_id": response_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        err_msg = str(exc)
        _log_event("whisper_transcription_error", level=logging.ERROR,
                   error_type=type(exc).__name__, error_prefix=err_msg[:120], response_id=response_id)
        raise HTTPException(
            status_code=502,
            detail=f"Transcription failed: {err_msg}. Please retry."
        ) from exc


@app.post("/api/interview/multimodal-report", response_model=MultimodalReportResponse)
async def multimodal_report(
    question: str = Form(...),
    reference_answer: str = Form(""),
    corrected_transcript: str = Form(...),
    response_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    position: Optional[int] = Form(None),
    original_transcript: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    if text_assessor is None or not getattr(text_assessor, "is_fitted", False):
        raise HTTPException(status_code=503, detail="Text assessment engine is not loaded.")
    if audio_analyzer is None:
        raise HTTPException(status_code=503, detail="Audio analysis engine is not loaded.")
    question = question.strip()
    transcript = corrected_transcript.strip()
    reference = reference_answer.strip()
    if not question or not transcript:
        raise HTTPException(status_code=400, detail="Question and corrected transcript are required.")
    if len(transcript.split()) < 3:
        raise HTTPException(status_code=400, detail="Transcript is too short to assess (minimum 3 words).")
    data = await file.read(MAX_AUDIO_BYTES + 1)
    try:
        ext = _validate_audio_bytes(file.filename or "", file.content_type or "", data)
    finally:
        await file.close()

    audio_digest = hashlib.sha256(data).hexdigest()[:12]
    durable_audio_name = None
    if session_id and position:
        try:
            recordings_dir = os.path.join(BASE_DIR, "data", "audio_recordings")
            os.makedirs(recordings_dir, exist_ok=True)
            durable_audio_name = f"{session_id}_q{position}_{audio_digest}{ext}"
            durable_path = os.path.join(recordings_dir, durable_audio_name)
            with open(durable_path, "wb") as f_out:
                f_out.write(data)
        except Exception as e:
            logger.warning(f"Could not persist candidate audio recording: {e}")
    if not response_id:
        response_id = f"RESP-{hashlib.sha256((question + transcript + audio_digest).encode()).hexdigest()[:12].upper()}"
    if not re.fullmatch(r"RESP-[A-F0-9]{10,32}", response_id):
        raise HTTPException(status_code=400, detail="Invalid response identifier.")
    _log_event("audio_upload_received", file_format=ext, audio_bytes=len(data), response_id=response_id)
    audio_metrics = await run_in_threadpool(
        _analyze_audio_bytes, file.filename or "recording.wav", data, transcript, response_id
    )
    scores = await run_in_threadpool(text_assessor.assess_answer, question, transcript, reference or question)

    strengths = list(scores.get("strengths", []))
    improvements = list(scores.get("improvements", []))
    pause_ratio = audio_metrics.get("pause_ratio")
    if pause_ratio is not None:
        if pause_ratio > 0.35:
            improvements.append("The recording contains a high measured pause ratio; practice connecting key points with shorter pauses.")
        elif pause_ratio < 0.08:
            improvements.append("The recording contains very little silence; consider brief pauses between major points.")
        else:
            strengths.append("The measured pause ratio falls within the configured conversational range.")
    filler_count = audio_metrics.get("filler_word_count")
    if isinstance(filler_count, numbers.Real) and filler_count > 0:
        improvements.append(f"The corrected transcript contains {filler_count} detected filler-word occurrence(s).")
    elif filler_count == 0 and transcript:
        strengths.append("No configured filler words were found in the corrected transcript.")

    component_fields = (
        "relevance_score",
        "question_alignment_score",
        "keyword_overlap",
        "structure_score",
    )
    reported_scores = dict(scores)
    for field in component_fields:
        reported_scores[field] = _valid_unit_score(scores.get(field))
    try:
        text_quality = calculate_text_assessment_score(reported_scores)
    except (TypeError, ValueError):
        text_quality = None
    reported_scores["quality_score"] = text_quality
    reported_scores["reference_similarity"] = reported_scores["relevance_score"]
    reported_scores["strengths"] = strengths
    reported_scores["improvements"] = improvements

    measured_fields = (
        "duration_sec", "speaking_duration_sec", "pause_ratio", "pause_duration_sec",
        "pause_count", "mean_pitch_hz", "pitch_std_hz", "min_pitch_hz", "max_pitch_hz",
        "mean_energy_rms",
    )
    available_audio_fields = [
        field for field in measured_fields
        if _finite_or_none(audio_metrics.get(field)) is not None
    ]
    mfcc_values = audio_metrics.get("mfcc_mean")
    mfcc_values = mfcc_values if isinstance(mfcc_values, (list, tuple)) else []
    if any(_finite_or_none(value) is not None for value in mfcc_values):
        available_audio_fields.append("mfcc_mean")
    voice_available = any(field != "duration_sec" for field in available_audio_fields)
    voice_analysis_status = "Available" if voice_available else "Not available"
    if text_quality is not None and voice_available:
        overall_status = "Available"
        overall_explanation = (
            "The text assessment score and separate voice measurements are both available. "
            "They are reported independently; no combined multimodal numeric score is calculated."
        )
    elif text_quality is None and voice_available:
        overall_status = "Not available"
        overall_explanation = "Voice measurements are available, but a valid text assessment score could not be calculated."
    elif text_quality is not None:
        overall_status = "Not available"
        overall_explanation = "The text assessment score is available, but no usable audio feature measurements were returned."
    else:
        overall_status = "Not available"
        overall_explanation = "Neither a valid text assessment score nor usable audio measurements are available."

    score_formula = (
        "0.50 * TF-IDF reference cosine similarity + 0.20 * TF-IDF question cosine similarity "
        "+ 0.15 * Jaccard keyword overlap + 0.15 * answer structure heuristic"
    )
    text_score_status = "Available" if text_quality is not None else "Not available"
    report = {
        "assessment_type": "Independent Text and Audio Assessment",
        "response_id": response_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interview_information": {
            "question": question,
            "reference_answer": reference,
            "corrected_transcript": transcript,
            "original_transcript": original_transcript.strip() if original_transcript else None,
            "response_id": response_id,
        },
        "text_assessment": reported_scores,
        "text_assessment_score": text_quality,
        "text_assessment_score_status": text_score_status,
        "text_assessment_score_formula": score_formula,
        "text_assessment_score_limitation": (
            "A heuristic reference-based practice score, not calibrated against human ratings and not a hiring decision."
        ),
        "voice_analysis_status": voice_analysis_status,
        "voice_analysis_available_fields": available_audio_fields,
        "voice_assessment": audio_metrics,
        "overall_multimodal_assessment": {
            "status": overall_status,
            "explanation": overall_explanation,
        },
        "feedback": {"strengths": strengths, "areas_for_improvement": improvements},
        "multimodal_summary": {
            "content_summary": f"Text assessment score: {text_quality:.1%}." if text_quality is not None else "Text assessment score is unavailable.",
            "delivery_summary": "Audio feature measurements are reported separately in voice_assessment.",
            "composite_verdict": overall_explanation,
        }
    }
    _log_event("assessment_report_generated", response_id=response_id,
               duration_sec=audio_metrics.get("duration_sec"), text_quality=text_quality)
    if durable_audio_name:
        report["audio_file_name"] = durable_audio_name
    if session_id:
        if position is None or position < 1:
            raise HTTPException(status_code=422, detail="A valid session question position is required with session_id.")
        persist_assessment(session_id, position, "voice", report)
    return report

@app.get("/api/audio/recordings/{filename}")
def get_audio_recording(filename: str):
    recordings_dir = os.path.join(BASE_DIR, "data", "audio_recordings")
    safe_filename = os.path.basename(filename)
    audio_path = os.path.join(recordings_dir, safe_filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio recording not found.")
    ext = os.path.splitext(safe_filename)[1].lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
    }
    media_type = mime_map.get(ext, "audio/wav")
    return FileResponse(audio_path, media_type=media_type)


@app.get("/api/v1/models/meta")
def get_model_metadata():
    """Returns trained employability model benchmark metrics and training metadata."""
    meta_path = os.path.join(MODELS_DIR, "employability_model_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load employability model metadata: {e}")
    return {
        "features": [
            "MANNER OF SPEAKING", "MENTAL ALERTNESS", "SELF-CONFIDENCE",
            "ABILITY TO PRESENT IDEAS", "COMMUNICATION SKILLS",
            "soft_skill_mean", "soft_skill_min"
        ],
        "best_model": "Gradient Boosting",
        "best_model_file": "gradient_boosting_employability.joblib",
        "val_results": {},
        "test_results": {},
    }


@app.post("/api/audio/tts")
@app.get("/api/audio/tts")
async def text_to_speech(request: Request, text: str = ""):
    """Generate or synthesize speech audio for interviewer question text."""
    clean_prompt = text.strip()
    if not clean_prompt:
        try:
            body = await request.json()
            if isinstance(body, dict):
                clean_prompt = (body.get("text") or "").strip()
        except Exception:
            pass
    if not clean_prompt:
        # Check form data if applicable
        try:
            form = await request.form()
            clean_prompt = (form.get("text") or "").strip()
        except Exception:
            pass

    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty.")
    
    # Generate speech WAV using SAPI on Windows or fallback synthetic audio
    try:
        import subprocess
        tts_dir = os.path.join(BASE_DIR, "data", "tts_cache")
        os.makedirs(tts_dir, exist_ok=True)
        text_hash = hashlib.sha256(clean_prompt.encode("utf-8")).hexdigest()[:16]
        out_wav = os.path.join(tts_dir, f"tts_{text_hash}.wav")
        if not os.path.exists(out_wav) or os.path.getsize(out_wav) == 0:
            # Safe Windows PowerShell SpeechSynthesizer script
            escaped_text = clean_prompt.replace("'", "''").replace('"', '`"')
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$synth.SetOutputToWaveFile('{out_wav}'); "
                f"$synth.Speak('{escaped_text}'); "
                "$synth.Dispose()"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                check=True,
                timeout=15,
                capture_output=True,
            )
        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
            return FileResponse(out_wav, media_type="audio/wav")
    except Exception as exc:
        logger.warning(f"Native TTS generation failed: {exc}; falling back to browser synthesis.")
    
    return {
        "status": "ready",
        "text": clean_prompt,
        "message": "Use client-side speech synthesis",
    }

@app.get("/api/v1/questions")
def get_interview_questions(limit: int = 15):
    """Returns curated mock interview questions with expert reference answers."""
    if not cached_questions:
        return {"questions": []}
    return {"questions": cached_questions[:limit]}


@app.post("/api/v1/predict/employability", response_model=EmployabilityResponse)
def predict_employability(req: EmployabilityRequest):
    """
    Predicts student employability based on 5 non-appearance soft-skill features.
    Computes engineered soft_skill_mean and soft_skill_min before inference.
    """
    if employability_model is None:
        raise HTTPException(status_code=503, detail="Employability model pipeline is not loaded.")
    
    # Feature engineering matching the trained model pipeline
    features_dict = {
        "MANNER OF SPEAKING": req.manner_of_speaking,
        "MENTAL ALERTNESS": req.mental_alertness,
        "SELF-CONFIDENCE": req.self_confidence,
        "ABILITY TO PRESENT IDEAS": req.ability_to_present_ideas,
        "COMMUNICATION SKILLS": req.communication_skills,
    }
    raw_vals = list(features_dict.values())
    features_dict["soft_skill_mean"] = float(np.mean(raw_vals))
    features_dict["soft_skill_min"] = int(np.min(raw_vals))
    
    input_df = pd.DataFrame([features_dict])
    
    # Predict
    pred = employability_model.predict(input_df)[0]
    proba = employability_model.predict_proba(input_df)[0]
    
    emp_idx = list(employability_model.classes_).index("Employable")
    less_idx = 1 - emp_idx
    
    p_emp = float(proba[emp_idx])
    p_less = float(proba[less_idx])
    
    # Generate tailored actionable feedback
    feedback = []
    if req.communication_skills < 3:
        feedback.append("Focus on verbal conciseness and vocabulary precision in technical descriptions.")
    if req.manner_of_speaking < 3:
        feedback.append("Practice steady pacing and clear vocal articulation to enhance delivery.")
    if req.self_confidence < 3:
        feedback.append("Engage in timed mock interviews to reduce hesitation under pressure.")
    if req.ability_to_present_ideas < 3:
        feedback.append("Utilize the STAR method (Situation, Task, Action, Result) to organize answers.")
    if req.mental_alertness < 3:
        feedback.append("Take 3 seconds to structure your thoughts before answering complex questions.")
    if not feedback:
        feedback.append("Excellent soft-skill profile! Continue reinforcing technical domain knowledge.")

    return EmployabilityResponse(
        prediction=pred,
        probability_employable=round(p_emp, 4),
        probability_less_employable=round(p_less, 4),
        confidence_score=round(max(p_emp, p_less) * 100, 1),
        soft_skill_mean=round(features_dict["soft_skill_mean"], 2),
        soft_skill_min=features_dict["soft_skill_min"],
        recommendations=feedback
    )


@app.post("/api/v1/assess/text", response_model=TextAssessmentResponse)
def assess_interview_text(req: TextAssessmentRequest):
    """
    Evaluates candidate typed or transcribed answer against question and reference answer.
    """
    if text_assessor is None:
        raise HTTPException(status_code=503, detail="Text assessment engine not loaded.")
    
    q = req.question.strip()
    ans = req.candidate_answer.strip()
    ref = req.reference_answer.strip() if req.reference_answer else q
    
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(ans.split()) < 3:
        raise HTTPException(status_code=400, detail="Answer is too short to evaluate (minimum 3 words).")
        
    scores = text_assessor.assess_answer(q, ans, ref)
    q_score = scores["quality_score"]
    
    if q_score >= 0.35:
        tier = "Proficient / Strong Answer"
    elif q_score >= 0.15:
        tier = "Intermediate / Partially Correct"
    else:
        tier = "Developing / Insufficient Coverage"
        
    feedback = []
    if scores["relevance_score"] < 0.20:
        feedback.append("Content relevance is low compared to the reference standard; ensure you address specific technical keywords.")
    else:
        feedback.append("Good semantic alignment with the expected technical concepts.")
        
    if scores["structure_score"] < 0.40:
        feedback.append("Response could be improved by using logical transitions (e.g. 'for example', 'specifically', 'therefore').")
    else:
        feedback.append("Well-structured sentence organization and adequate response length.")
        
    response = TextAssessmentResponse(
        relevance_score=scores["relevance_score"],
        question_alignment_score=scores["question_alignment_score"],
        keyword_overlap=scores["keyword_overlap"],
        structure_score=scores["structure_score"],
        quality_score=q_score,
        performance_tier=tier,
        feedback=feedback
    )
    if req.session_id:
        if req.position is None or req.position < 1:
            raise HTTPException(status_code=422, detail="A valid session question position is required with session_id.")
        persist_assessment(req.session_id, req.position, "text", {
            "interview_information": {"question": q, "candidate_answer": ans},
            "text_assessment": {**scores, "quality_score": q_score,
                                "strengths": scores.get("strengths", []),
                                "improvements": scores.get("improvements", [])},
        })
    return response


@app.post("/api/v1/assess/audio")
async def assess_interview_audio(file: UploadFile = File(...)):
    """
    Accepts supported audio and returns descriptive acoustic measurements.
    """
    if audio_analyzer is None:
        raise HTTPException(status_code=503, detail="Audio analyzer not loaded.")

    data = await file.read(MAX_AUDIO_BYTES + 1)
    try:
        ext = _validate_audio_bytes(file.filename or "", file.content_type or "", data)
    finally:
        await file.close()

    _log_event("audio_upload_received", file_format=ext, audio_bytes=len(data))
    metrics = await run_in_threadpool(_analyze_audio_bytes, file.filename or "recording.wav", data, "")
    # Generate speech feedback
    audio_feedback = []
    if metrics.get("pause_ratio") is None:
        audio_feedback.append("Pause ratio was unavailable for this recording.")
    elif metrics["pause_ratio"] > 0.35:
        audio_feedback.append("High silence ratio detected (>35%); try to reduce long hesitations between sentences.")
    elif metrics["pause_ratio"] < 0.08:
        audio_feedback.append("Very low pause ratio (<8%); remember to breathe and pause between major thoughts.")
    else:
        audio_feedback.append("Healthy speech rhythm and natural pacing.")
        
    if metrics.get("pitch_std_hz") is not None and metrics["pitch_std_hz"] < 10.0:
        audio_feedback.append("Low pitch variation was measured in this clip; this is a descriptive signal measurement.")
    elif metrics.get("pitch_std_hz") is not None:
        audio_feedback.append("Pitch variation was measured; it is a descriptive acoustic feature.")
        
    return {
        "audio_metrics": metrics,
        "speech_feedback": audio_feedback
    }


@app.post("/api/v1/assess/multimodal")
async def assess_multimodal(
    question: str = Form(...),
    reference_answer: str = Form(""),
    candidate_answer: str = Form(...),
    manner_of_speaking: int = Form(4),
    mental_alertness: int = Form(4),
    self_confidence: int = Form(4),
    ability_to_present_ideas: int = Form(4),
    communication_skills: int = Form(4),
    audio_file: Optional[UploadFile] = File(None)
):
    """
    Comprehensive Multimodal Assessment: Text + Soft Skills + Optional Audio.
    """
    if text_assessor is None:
        raise HTTPException(status_code=503, detail="Text assessment engine not loaded.")
    if employability_model is None:
        raise HTTPException(status_code=503, detail="Employability model pipeline is not loaded.")

    q = question.strip()
    ans = candidate_answer.strip()
    ref = reference_answer.strip() if reference_answer else q
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(ans.split()) < 3:
        raise HTTPException(status_code=400, detail="Answer is too short to evaluate (minimum 3 words).")

    ratings = [manner_of_speaking, mental_alertness, self_confidence, ability_to_present_ideas, communication_skills]
    if not all(1 <= r <= 5 for r in ratings):
        raise HTTPException(status_code=422, detail="Soft skill ratings must be integers between 1 and 5.")

    # 1. Text assessment
    text_res = await run_in_threadpool(text_assessor.assess_answer, q, ans, ref)
    
    # 2. Soft skills employability prediction
    features_dict = {
        "MANNER OF SPEAKING": manner_of_speaking,
        "MENTAL ALERTNESS": mental_alertness,
        "SELF-CONFIDENCE": self_confidence,
        "ABILITY TO PRESENT IDEAS": ability_to_present_ideas,
        "COMMUNICATION SKILLS": communication_skills,
    }
    raw_vals = list(features_dict.values())
    features_dict["soft_skill_mean"] = float(np.mean(raw_vals))
    features_dict["soft_skill_min"] = int(np.min(raw_vals))
    
    prediction_frame = pd.DataFrame([features_dict])
    emp_pred, proba = await run_in_threadpool(
        lambda: (employability_model.predict(prediction_frame)[0], employability_model.predict_proba(prediction_frame)[0])
    )
    emp_idx = list(employability_model.classes_).index("Employable")
    p_emp = float(proba[emp_idx])
    
    # 3. Audio assessment if file provided
    audio_metrics = None
    if audio_file is not None:
        audio_data = await audio_file.read(MAX_AUDIO_BYTES + 1)
        try:
            audio_ext = _validate_audio_bytes(audio_file.filename or "", audio_file.content_type or "", audio_data)
        finally:
            await audio_file.close()
        _log_event("audio_upload_received", file_format=audio_ext, audio_bytes=len(audio_data))
        audio_metrics = await run_in_threadpool(
            _analyze_audio_bytes, audio_file.filename or "recording.wav", audio_data, ans
        )
                    
    result = {
        "overall_status": "Evaluation Complete",
        "employability_result": {
            "prediction": emp_pred,
            "employability_probability": round(p_emp, 4)
        },
        "text_evaluation": text_res,
        "speech_acoustic_evaluation": audio_metrics
    }
    _log_event("multimodal_assessment_completed", employability_prediction=emp_pred,
               audio_included=audio_metrics is not None)
    return result


def _merge_transcript_text(existing: str, incoming: str) -> str:
    """Append incoming transcript while trimming duplicated overlap from chunk boundaries."""
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    e_words = existing.split()
    i_words = incoming.split()
    max_overlap = min(len(e_words), len(i_words), 12)
    for k in range(max_overlap, 0, -1):
        if [w.lower().strip(".,!?;:") for w in e_words[-k:]] == [w.lower().strip(".,!?;:") for w in i_words[:k]]:
            merged = e_words + i_words[k:]
            return " ".join(merged).strip()
    if incoming.lower() in existing.lower():
        return existing
    return (existing + " " + incoming).strip()


def _write_wav_from_float32(path: str, audio: np.ndarray, sample_rate: int = 16000) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


@app.websocket("/api/audio/transcribe/ws")
async def ws_transcribe_audio(websocket: WebSocket):
    """Incremental chunk transcription over WebSocket using faster-whisper.

    Protocol (JSON text messages):
      - client -> {"type":"start","session_id":"...","position":1,"sample_rate":16000}
      - client -> {"type":"chunk","audio_b64":"<pcm16 mono base64>","sample_rate":16000,"is_final_chunk":false}
      - client -> {"type":"finalize"} or {"type":"stop"}
      - client -> {"type":"reset"}
      - server -> {"type":"status","state":"ready|listening|processing|finalizing|completed|error","message":"..."}
      - server -> {"type":"partial","text":"...","full_text":"...","detected_language":"..."}
      - server -> {"type":"final","text":"...","detected_language":"...","language_probability":0.0}
    """
    await websocket.accept()
    state = {
        "session_id": None,
        "position": None,
        "sample_rate": 16000,
        "chunks_pcm": [],  # list[np.ndarray float32]
        "partial_text": "",
        "detected_language": "unknown",
        "language_probability": 0.0,
        "last_chunk_samples": 0,
    }
    overlap_sec = 0.8
    target_sec = max(2.0, float(TRANSCRIPTION_CHUNK_TARGET_SEC))

    async def send(obj):
        await websocket.send_text(json.dumps(_json_safe(obj) if "_json_safe" in globals() else obj, default=str))

    try:
        await send({"type": "status", "state": "ready", "message": "Waiting for microphone audio chunks."})
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    await send({"type": "status", "state": "error", "message": "Invalid JSON message."})
                    continue
                mtype = data.get("type")
                if mtype == "start":
                    state["session_id"] = data.get("session_id")
                    state["position"] = data.get("position")
                    state["sample_rate"] = int(data.get("sample_rate") or 16000)
                    state["chunks_pcm"] = []
                    state["partial_text"] = ""
                    state["detected_language"] = "unknown"
                    await send({"type": "status", "state": "listening", "message": "Recording started. Listening and transcribing."})
                elif mtype == "reset":
                    state["chunks_pcm"] = []
                    state["partial_text"] = ""
                    await send({"type": "status", "state": "ready", "message": "Recording cleared."})
                elif mtype == "chunk":
                    b64 = data.get("audio_b64")
                    if not b64:
                        continue
                    try:
                        raw = base64.b64decode(b64)
                    except Exception:
                        await send({"type": "status", "state": "error", "message": "Invalid audio chunk encoding."})
                        continue
                    # Expect PCM16 mono from browser
                    pcm16 = np.frombuffer(raw, dtype=np.int16)
                    if pcm16.size == 0:
                        continue
                    f32 = (pcm16.astype(np.float32) / 32768.0)
                    state["chunks_pcm"].append(f32)
                    state["last_chunk_samples"] += len(f32)
                    sr = int(data.get("sample_rate") or state["sample_rate"] or 16000)
                    state["sample_rate"] = sr
                    accumulated_sec = state["last_chunk_samples"] / max(sr, 1)
                    # Transcribe when we have enough new audio for a stable incremental update
                    if accumulated_sec >= target_sec or data.get("is_final_chunk"):
                        await send({"type": "status", "state": "processing", "message": "Listening and transcribing..."})
                        # Use recent window with overlap for continuity, but keep full recording separately
                        all_audio = np.concatenate(state["chunks_pcm"]) if state["chunks_pcm"] else np.array([], dtype=np.float32)
                        max_window = int((target_sec + overlap_sec) * sr)
                        window = all_audio[-max_window:] if len(all_audio) > max_window else all_audio
                        try:
                            res = await run_in_threadpool(transcription_service.transcribe_array, window, sr, None, 5, True)
                            incoming = (res.get("transcript") or "").strip()
                            if incoming:
                                state["partial_text"] = _merge_transcript_text(state["partial_text"], incoming)
                            lang = res.get("detected_language") or state.get("detected_language") or "unknown"
                            state["detected_language"] = lang
                            state["language_probability"] = res.get("language_probability", state.get("language_probability", 0.0))
                            await send({
                                "type": "partial",
                                "text": incoming,
                                "full_text": state["partial_text"],
                                "detected_language": state["detected_language"],
                                "language_probability": state.get("language_probability", 0.0),
                            })
                        except Exception as exc:
                            await send({"type": "status", "state": "error", "message": f"Incremental transcription failed: {type(exc).__name__}. Recording preserved; you can retry final transcription."})
                        state["last_chunk_samples"] = 0
                        await send({"type": "status", "state": "listening", "message": "Recording..."})
                elif mtype in ("finalize", "stop"):
                    await send({"type": "status", "state": "finalizing", "message": "Finalizing transcript from complete recording..."})
                    all_audio = np.concatenate(state["chunks_pcm"]) if state["chunks_pcm"] else np.array([], dtype=np.float32)
                    sr = int(state.get("sample_rate") or 16000)
                    duration = len(all_audio) / max(sr, 1)
                    if duration < 0.25:
                        await send({"type": "status", "state": "error", "message": "Recording is too short to transcribe. Please speak for at least a moment and try again."})
                        continue
                    if duration > TRANSCRIPTION_FINAL_MAX_SEC:
                        await send({"type": "status", "state": "error", "message": "Recording is too long for local CPU transcription. Please record a shorter answer."})
                        continue
                    tmp_wav = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp_wav = tmp.name
                        await run_in_threadpool(_write_wav_from_float32, tmp_wav, all_audio, sr)
                        with open(tmp_wav, "rb") as f:
                            wav_bytes = f.read()
                        res = await run_in_threadpool(transcription_service.transcribe, wav_bytes, ".wav", None)
                        final_text = (res.get("transcript") or "").strip() or state["partial_text"].strip()
                        payload = {
                            "type": "final",
                            "text": final_text,
                            "detected_language": res.get("detected_language", state.get("detected_language", "unknown")),
                            "language_probability": res.get("language_probability", 0.0),
                            "duration_sec": round(duration, 2),
                        }
                        state["partial_text"] = final_text
                        await send(payload)
                        await send({"type": "status", "state": "completed", "message": "Transcription completed. Review and edit the final transcript."})
                    except Exception as exc:
                        # Keep partial transcript available
                        await send({
                            "type": "error",
                            "message": f"Final transcription failed: {type(exc).__name__}. Partial transcript preserved; you can retry finalization.",
                            "partial_text": state.get("partial_text", ""),
                        })
                    finally:
                        _safe_remove_temp(tmp_wav)
                else:
                    await send({"type": "status", "state": "error", "message": f"Unknown message type: {mtype}"})
            elif "bytes" in msg and msg["bytes"] is not None:
                # Fallback: raw PCM16 chunk
                raw = msg["bytes"]
                pcm16 = np.frombuffer(raw, dtype=np.int16)
                if pcm16.size:
                    state["chunks_pcm"].append(pcm16.astype(np.float32) / 32768.0)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        try:
            await send({"type": "error", "message": f"Transcription session error: {type(exc).__name__}"})
        except Exception:
            pass
        return

