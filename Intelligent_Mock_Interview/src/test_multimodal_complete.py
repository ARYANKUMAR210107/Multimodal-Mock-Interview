"""Regression tests for descriptive audio analysis and multimodal reports."""
import io
import math
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = WORKSPACE / "Intelligent_Mock_Interview"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(PROJECT))

from fastapi.testclient import TestClient
from Intelligent_Mock_Interview.backend import main as api
from Intelligent_Mock_Interview.src.audio_feature_extractor import CandidateAudioAnalyzer
from Intelligent_Mock_Interview.src.text_pipeline import calculate_text_assessment_score
from Intelligent_Mock_Interview.src.voice_session import reset_for_question_change


def make_wav(duration=2.5, sample_rate=16000, frequency=180):
    samples = int(duration * sample_rate)
    signal = 0.35 * np.sin(2 * math.pi * frequency * np.arange(samples) / sample_rate)
    signal[int(1.0 * sample_rate):int(1.25 * sample_rate)] = 0
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes((signal * 32767).astype("<i2").tobytes())
    return buffer.getvalue(), signal.astype(np.float32)


class AudioFeatureTests(unittest.TestCase):
    def test_reports_descriptive_features_without_unvalidated_quality_scores(self):
        _, samples = make_wav()
        result = CandidateAudioAnalyzer(target_sr=16000).extract_features_from_array(samples)
        self.assertGreater(result["duration_sec"], 2.0)
        self.assertGreaterEqual(result["pause_ratio"], 0.0)
        self.assertEqual(len(result["mfcc_mean"]), 13)
        self.assertNotIn("estimated_fluency_score", result)
        self.assertNotIn("estimated_prosody_score", result)

    def test_empty_audio_has_missing_measurements_not_fabricated_zeros(self):
        result = CandidateAudioAnalyzer().extract_features_from_array(np.array([], dtype=np.float32))
        self.assertEqual(result["duration_sec"], 0.0)
        self.assertIsNone(result["mean_energy_rms"])
        self.assertIsNone(result["pause_ratio"])
        self.assertTrue(all(value is None for value in result["mfcc_mean"]))

    def test_invalid_sample_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            CandidateAudioAnalyzer().extract_features_from_array(np.zeros(8000), sr=0)


class TextAssessmentScoreTests(unittest.TestCase):
    def test_weighted_text_score_uses_documented_components(self):
        score = calculate_text_assessment_score({
            "relevance_score": 0.5,
            "question_alignment_score": 0.4,
            "keyword_overlap": 0.3,
            "structure_score": 0.6,
        })
        self.assertEqual(score, 0.465)

    def test_invalid_component_is_not_coerced_to_zero(self):
        with self.assertRaises(ValueError):
            calculate_text_assessment_score({
                "relevance_score": float("nan"),
                "question_alignment_score": 0.4,
                "keyword_overlap": 0.3,
                "structure_score": 0.6,
            })

    def test_stereo_input_is_downmixed_and_silence_has_no_pitch(self):
        _, mono = make_wav(duration=1.0)
        stereo = np.column_stack((mono, mono))
        result = CandidateAudioAnalyzer().extract_features_from_array(stereo, sr=16000)
        self.assertAlmostEqual(result["duration_sec"], 1.0, places=1)

        silence = CandidateAudioAnalyzer().extract_features_from_array(np.zeros(16000), sr=16000)
        self.assertIsNone(silence["mean_pitch_hz"])
        self.assertEqual(silence["mean_energy_rms"], 0.0)


class VoiceSessionTests(unittest.TestCase):
    def test_question_change_clears_audio_transcript_and_report(self):
        state = {
            "voice_question_context": "Old question",
            "voice_audio_bytes": b"old audio",
            "voice_audio_digest": "old digest",
            "voice_transcript_editor": "old transcript",
            "voice_report": {"response_id": "RESP-OLD"},
            "voice_recording": "old recording",
        }
        self.assertTrue(reset_for_question_change(state, "New question"))
        self.assertEqual(state["voice_transcript_editor"], "")
        self.assertEqual(state["pending_voice_transcript"], "")
        for key in ("voice_audio_bytes", "voice_audio_digest", "voice_report", "voice_recording"):
            self.assertNotIn(key, state)
        self.assertFalse(reset_for_question_change(state, "New question"))


class MultimodalReportTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        self.audio, _ = make_wav()

    def post_report(self, response_id="RESP-0123456789AB"):
        return self.client.post(
            "/api/interview/multimodal-report",
            data={
                "question": "Explain supervised learning.",
                "reference_answer": "Supervised learning uses labeled examples to train a model.",
                "corrected_transcript": "Supervised learning trains a model using labeled examples and known outcomes.",
                "response_id": response_id,
            },
            files={"file": ("answer.wav", self.audio, "audio/wav")},
        )

    def test_report_keeps_text_audio_separate_and_preserves_response_id(self):
        response = self.post_report()
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertEqual(report["response_id"], "RESP-0123456789AB")
        self.assertEqual(report["interview_information"]["response_id"], report["response_id"])
        self.assertIn("quality_score", report["text_assessment"])
        self.assertAlmostEqual(report["text_assessment_score"], report["text_assessment"]["quality_score"])
        self.assertEqual(report["text_assessment_score_status"], "Available")
        self.assertIn("0.50 * TF-IDF reference cosine", report["text_assessment_score_formula"])
        self.assertEqual(report["overall_multimodal_assessment"]["status"], "Available")
        self.assertIn("mean_energy_rms", report["voice_assessment"])
        self.assertNotIn("combined_score", report)

    def test_missing_audio_features_keeps_text_score_and_marks_multimodal_unavailable(self):
        with patch.object(api.audio_analyzer, "analyze_audio_file", return_value={}):
            response = self.post_report()
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertIsNotNone(report["text_assessment_score"])
        self.assertEqual(report["voice_analysis_status"], "Not available")
        self.assertEqual(report["overall_multimodal_assessment"]["status"], "Not available")
        self.assertIsNone(report["voice_assessment"]["mean_energy_rms"])

    def test_missing_audio_is_rejected_and_zero_length_audio_is_rejected(self):
        base_data = {
            "question": "Explain supervised learning.",
            "corrected_transcript": "Supervised learning trains a model using labeled examples.",
        }
        missing = self.client.post("/api/interview/multimodal-report", data=base_data)
        empty = self.client.post(
            "/api/interview/multimodal-report",
            data=base_data,
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        self.assertEqual(missing.status_code, 422)
        self.assertEqual(empty.status_code, 400)

    def test_invalid_text_component_makes_text_score_unavailable_not_zero(self):
        assessor = MagicMock()
        assessor.is_fitted = True
        assessor.assess_answer.return_value = {
            "relevance_score": float("nan"),
            "question_alignment_score": 0.5,
            "keyword_overlap": 0.4,
            "structure_score": 0.6,
            "quality_score": 0.5,
            "strengths": [],
            "improvements": [],
            "suggested_structure": [],
        }
        with patch.object(api, "text_assessor", assessor):
            response = self.post_report()
        self.assertEqual(response.status_code, 200, response.text)
        report = response.json()
        self.assertIsNone(report["text_assessment_score"])
        self.assertEqual(report["text_assessment_score_status"], "Not available")
        self.assertIsNone(report["text_assessment"]["quality_score"])
        self.assertEqual(report["overall_multimodal_assessment"]["status"], "Not available")

    def test_report_rejects_audio_shorter_than_minimum_duration(self):
        short_audio, _ = make_wav(duration=0.1)
        response = self.client.post(
            "/api/interview/multimodal-report",
            data={
                "question": "Explain supervised learning.",
                "corrected_transcript": "Supervised learning uses labeled examples.",
            },
            files={"file": ("short.wav", short_audio, "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)

    def test_transcription_uses_whisper_with_mocked_response(self):
        mock_result = {
            "transcript": "A mocked interview answer.",
            "detected_language": "en",
            "language_probability": 0.99,
        }
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result):
            response = self.client.post(
                "/api/audio/transcribe",
                data={"response_id": "RESP-0123456789AB"},
                files={"file": ("answer.wav", self.audio, "audio/wav")},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["transcript"], "A mocked interview answer.")
        self.assertEqual(response.json()["response_id"], "RESP-0123456789AB")
        self.assertEqual(response.json()["provider"], "faster-whisper")


if __name__ == "__main__":
    unittest.main(verbosity=2)
