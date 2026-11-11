"""Focused API tests for voice transcription and multimodal reports with Google Gemini.

Run from Intelligent_Mock_Interview with:
    python -m unittest src.test_voice_interview_api
Gemini API calls are mocked; no real API key or usage charges are consumed.
"""
import os
import sys
import unittest
import io
import math
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from fastapi.testclient import TestClient
from backend import main as api


def wav_stub(duration=1.0):
    buffer = io.BytesIO()
    rate = 16000
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        samples = (int(9000 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(int(rate * duration)))
        output.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    return buffer.getvalue()


class VoiceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_health_endpoint_and_request_id_header(self):
        response = self.client.get("/health", headers={"X-Request-ID": "RESP-0123456789AB"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")
        self.assertIsInstance(response.json()["transcription_configured"], bool)
        self.assertEqual(response.headers["X-Request-ID"], "RESP-0123456789AB")

    def test_transcription_uses_faster_whisper(self):
        mock_result = {
            "transcript": "A corrected answer.",
            "detected_language": "en",
            "language_probability": 0.99,
        }
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result):
            response = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("answer.wav", wav_stub(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["transcript"], "A corrected answer.")
        self.assertEqual(response.json()["provider"], "faster-whisper")
        self.assertIn("whisper", response.json()["model"].lower())

    def test_transcription_failure_maps_to_502(self):
        with patch.object(api.transcription_service, "transcribe", side_effect=RuntimeError("Whisper decode error")):
            response = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("answer.wav", wav_stub(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 502)
        self.assertIn("Transcription failed", response.json()["detail"])

    def test_mp3_upload_uses_supported_mime_and_is_forwarded(self):
        mock_result = {
            "transcript": "Mocked MP3 transcript.",
            "detected_language": "en",
            "language_probability": 0.98,
        }
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result):
            response = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("answer.mp3", b"ID3" + b"\x00" * 20, "audio/mpeg")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["transcript"], "Mocked MP3 transcript.")

    def test_empty_speech_transcription_returns_422(self):
        mock_result = {"transcript": "", "detected_language": "unknown", "language_probability": 0.0}
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result):
            response = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("answer.wav", wav_stub(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 422)

    def test_rejects_unsupported_and_over_limit_uploads(self):
        unsupported = self.client.post(
            "/api/audio/transcribe", files={"file": ("answer.txt", b"speech", "text/plain")}
        )
        oversized = self.client.post(
            "/api/audio/transcribe",
            files={"file": ("answer.wav", b"x" * (api.MAX_AUDIO_BYTES + 1), "audio/wav")},
        )
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(oversized.status_code, 413)

    def test_supported_extensions_match_audio_signatures_and_mime(self):
        samples = [
            ("answer.wav", wav_stub(), "audio/wav", ".wav"),
            ("answer.mp3", b"ID3" + b"\x00" * 20, "audio/mpeg", ".mp3"),
            ("answer.m4a", b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 8, "audio/mp4", ".m4a"),
            ("answer.webm", b"\x1a\x45\xdf\xa3" + b"\x00" * 12, "audio/webm", ".webm"),
        ]
        for name, data, mime, expected in samples:
            with self.subTest(extension=expected):
                self.assertEqual(api._validate_audio_bytes(name, mime, data), expected)

    def test_rejects_empty_and_mismatched_uploads(self):
        empty = self.client.post("/api/audio/transcribe", files={"file": ("answer.wav", b"", "audio/wav")})
        mismatch = self.client.post("/api/audio/transcribe", files={"file": ("answer.wav", b"not a wav", "audio/wav")})
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(mismatch.status_code, 415)

    def test_multimodal_report_uses_corrected_transcript(self):
        analyzer = MagicMock()
        analyzer.analyze_audio_file.return_value = {
            "duration_sec": 10.0,
            "speaking_duration_sec": 8.0,
            "pause_ratio": 0.2,
            "mean_energy_rms": 0.1,
            "mean_pitch_hz": 150.0,
            "pitch_std_hz": 18.0,
            "min_pitch_hz": 120.0,
            "max_pitch_hz": 190.0,
            "mfcc_mean": [0.0] * 13,
        }
        assessor = MagicMock()
        assessor.is_fitted = True
        assessor.assess_answer.return_value = {
            "relevance_score": 0.5,
            "question_alignment_score": 0.4,
            "keyword_overlap": 0.3,
            "structure_score": 0.6,
            "quality_score": 0.45,
            "strengths": [],
            "improvements": [],
            "suggested_structure": [],
        }
        with patch.object(api, "audio_analyzer", analyzer), patch.object(api, "text_assessor", assessor):
            response = self.client.post(
                "/api/interview/multimodal-report",
                data={"question": "Explain testing", "reference_answer": "Testing verifies behavior", "corrected_transcript": "Um testing verifies the expected application behavior."},
                files={"file": ("answer.wav", wav_stub(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(assessor.assess_answer.call_args.args[1], "Um testing verifies the expected application behavior.")
        self.assertEqual(response.json()["voice_assessment"]["filler_word_count"], 1)
        self.assertEqual(response.json()["voice_assessment"]["speaking_rate_wpm"], 42.0)
        self.assertEqual(response.json()["text_assessment_score"], 0.465)
        self.assertEqual(response.json()["text_assessment_score"], response.json()["text_assessment"]["quality_score"])

    def test_invalid_quality_score_is_recomputed_from_valid_components(self):
        assessor = MagicMock()
        assessor.is_fitted = True
        assessor.assess_answer.return_value = {
            "relevance_score": 0.5,
            "question_alignment_score": 0.4,
            "keyword_overlap": 0.3,
            "structure_score": 0.6,
            "quality_score": float("nan"),
            "strengths": [],
            "improvements": [],
            "suggested_structure": [],
        }
        with patch.object(api, "text_assessor", assessor):
            response = self.client.post(
                "/api/interview/multimodal-report",
                data={"question": "Explain testing", "corrected_transcript": "Testing checks expected software behavior."},
                files={"file": ("answer.wav", wav_stub(), "audio/wav")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["text_assessment_score"], 0.465)

    def test_multimodal_report_rejects_missing_transcript_and_zero_length_audio(self):
        base_data = {"question": "Explain testing"}
        missing_transcript = self.client.post(
            "/api/interview/multimodal-report",
            data=base_data,
            files={"file": ("answer.wav", wav_stub(), "audio/wav")},
        )
        empty_audio = self.client.post(
            "/api/interview/multimodal-report",
            data={**base_data, "corrected_transcript": "Testing checks software behavior."},
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        self.assertIn(missing_transcript.status_code, (400, 422))
        self.assertEqual(empty_audio.status_code, 400)

    def test_empty_transcript_is_rejected_before_audio_analysis(self):
        response = self.client.post(
            "/api/interview/multimodal-report",
            data={"question": "Explain testing", "corrected_transcript": ""},
            files={"file": ("answer.wav", wav_stub(), "audio/wav")},
        )
        self.assertIn(response.status_code, (400, 422))

    def test_corrupted_audio_is_reported_as_unprocessable(self):
        corrupted_wav = b"RIFF" + (4).to_bytes(4, "little") + b"WAVE" + b"data"
        response = self.client.post(
            "/api/interview/multimodal-report",
            data={
                "question": "Explain testing",
                "corrected_transcript": "Testing checks software behavior against expectations.",
            },
            files={"file": ("answer.wav", corrupted_wav, "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)

    def test_existing_text_assessment_endpoint_still_works(self):
        response = self.client.post(
            "/api/v1/assess/text",
            json={
                "question": "What is supervised learning?",
                "candidate_answer": "Supervised learning uses labeled examples to train a model for prediction.",
                "reference_answer": "Supervised learning uses labeled data to train predictive models.",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("quality_score", response.json())

    def test_existing_employability_endpoint_still_works(self):
        response = self.client.post(
            "/api/v1/predict/employability",
            json={
                "manner_of_speaking": 4,
                "mental_alertness": 4,
                "self_confidence": 4,
                "ability_to_present_ideas": 4,
                "communication_skills": 4,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn(response.json()["prediction"], ["Employable", "LessEmployable"])


if __name__ == "__main__":
    unittest.main()
