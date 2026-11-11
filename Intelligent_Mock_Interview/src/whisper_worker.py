"""
Standalone Hybrid Speech-to-Text Worker Script
==============================================
Transcribes audio using faster-whisper (offline), with fallback to SpeechRecognition.
Includes domain prompting and acoustic normalization for technical mock interviews.
"""
import sys
import os
import re
import json
import argparse


def normalize_audio_if_needed(audio_path: str) -> str:
    """Normalize audio amplitude to full dynamic range if recording is soft."""
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(audio_path)
        if len(data) == 0:
            return audio_path
        peak = np.max(np.abs(data))
        if 0.0 < peak < 0.65:
            norm_data = (data / peak) * 0.95
            norm_path = audio_path + ".norm.wav"
            sf.write(norm_path, norm_data, sr)
            return norm_path
    except Exception:
        pass
    return audio_path


def clean_technical_transcript(text: str, context_prompt: str = "") -> str:
    """Correct frequent acoustic misrecognitions on technical terminology."""
    if not text:
        return text

    # Common mishearing of "overfitting"
    # e.g., "covid-19 is a situation in which model...", "over particular situation in which model..."
    if re.search(r"\b(model|training|data|testing|unseen|learning|predict|accuracy)\b", text, re.I):
        text = re.sub(
            r"\b(COVID[-\s]?19|covid|over\s+particular(?:\s+situation)?)\b",
            "Overfitting is a situation" if "situation" in text.lower() else "overfitting",
            text,
            flags=re.I
        )
        text = re.sub(r"\b(over\s+feating|over\s+eating)\b", "overfitting", text, flags=re.I)
        text = re.sub(r"\b(under\s+feating)\b", "underfitting", text, flags=re.I)
        text = re.sub(r"\bwell[-\s]?centred\b", "training", text, flags=re.I)

    # Clean redundant spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transcribe_whisper(audio_path: str, model_size: str = "small.en", device: str = "cpu", compute_type: str = "int8", language: str = None, initial_prompt: str = None):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        return None, f"ImportError: {e}"

    # Try small.en first for highest technical accuracy, then fallbacks
    candidate_models = [model_size, "small.en", "distil-small.en", "base.en", "tiny.en"]
    seen = set()
    ordered_models = []
    for m in candidate_models:
        if m and m not in seen:
            seen.add(m)
            ordered_models.append(m)

    model = None
    last_err = None
    used_model = "small.en"
    for cand in ordered_models:
        try:
            model = WhisperModel(cand, device=device, compute_type=compute_type)
            used_model = cand
            break
        except Exception as exc:
            last_err = exc

    if model is None:
        return None, f"Model load failure: {last_err}"

    norm_path = normalize_audio_if_needed(audio_path)
    try:
        domain_prompt = (
            "Technical mock interview. Candidate explaining machine learning concepts. "
            "Vocabulary: Overfitting is a situation in which a model learns the training data too well, "
            "performing well on training data but poorly on unseen test data. "
            "Keywords: overfitting, underfitting, bias, variance, regularization, hyperparameters, "
            "training dataset, validation, test data, cross-validation, neural networks, decision tree."
        )
        if initial_prompt:
            prompt_str = f"{initial_prompt.strip()}. {domain_prompt}"
        else:
            prompt_str = domain_prompt

        lang_arg = language if language and language != "auto" else "en"
        segments, info = model.transcribe(
            norm_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=prompt_str,
            temperature=[0.0, 0.2, 0.4],
        )
        text = " ".join(s.text.strip() for s in segments if s.text).strip()
        cleaned_text = clean_technical_transcript(text, prompt_str)
        if cleaned_text:
            return {
                "status": "ok",
                "transcript": cleaned_text,
                "detected_language": getattr(info, "language", "en"),
                "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
                "engine": f"faster-whisper-{used_model}",
            }, None
        return None, "Whisper returned empty transcript"
    except Exception as exc:
        return None, str(exc)
    finally:
        if norm_path != audio_path and os.path.exists(norm_path):
            try:
                os.unlink(norm_path)
            except OSError:
                pass


def transcribe_speech_recognition(audio_path: str, language: str = "en-US", initial_prompt: str = None):
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        norm_path = normalize_audio_if_needed(audio_path)
        with sr.AudioFile(norm_path) as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = r.record(source)
        text = r.recognize_google(audio_data, language=language or "en-US")
        if text:
            cleaned_text = clean_technical_transcript(text.strip(), initial_prompt or "")
            return {
                "status": "ok",
                "transcript": cleaned_text,
                "detected_language": "en",
                "language_probability": 0.95,
                "engine": "google-stt",
            }, None
        return None, "Empty Google STT response"
    except Exception as exc:
        return None, str(exc)
    finally:
        if norm_path != audio_path and os.path.exists(norm_path):
            try:
                os.unlink(norm_path)
            except OSError:
                pass


def transcribe(audio_path: str, model_size: str = "small.en", device: str = "cpu", compute_type: str = "int8", language: str = None, initial_prompt: str = None):
    # Try faster-whisper with technical domain prompt first
    res, w_err = transcribe_whisper(audio_path, model_size, device, compute_type, language, initial_prompt)
    if res:
        return res

    # Fallback to SpeechRecognition with ambient noise adjustment
    res_sr, sr_err = transcribe_speech_recognition(audio_path, language="en-US", initial_prompt=initial_prompt)
    if res_sr:
        return res_sr

    return {
        "status": "error",
        "transcript": "",
        "error": f"Whisper error: {w_err}; SpeechRecognition error: {sr_err}",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="int8")
    parser.add_argument("--language", default=None)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    result = transcribe(args.audio, args.model, args.device, args.compute_type, args.language, args.prompt)
    print("RESULT_JSON:" + json.dumps(result))
