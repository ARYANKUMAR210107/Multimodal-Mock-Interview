"""
=============================================================================
Comprehensive End-to-End Test Suite for All FastAPI Endpoints & ML Pipelines
Intelligent Multimodal Mock Interview Assessment System (faster-whisper API)
=============================================================================
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import soundfile as sf
import numpy as np

# Ensure project paths
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from fastapi.testclient import TestClient
from backend import main as api


def create_test_wav_bytes(duration=1.0, sr=16000):
    """Generates a small valid WAV in-memory byte buffer."""
    import io
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


class FullBackendEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        cls.sample_wav_bytes = create_test_wav_bytes(duration=1.5)
        cls.sample_wav_path = os.path.join(str(BASE_DIR), "reports", "test_candidate_sample.wav")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Root & Health Check
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_get_root(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("version", data)
        self.assertIn("transcription_configured", data)
        self.assertEqual(data.get("transcription_provider"), "faster-whisper")
        self.assertIsInstance(data["endpoints"], list)
        self.assertIn("/api/audio/transcribe", data["endpoints"])
        self.assertIn("/api/interview/multimodal-report", data["endpoints"])

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Reference Questions Endpoint
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_get_questions(self):
        res = self.client.get("/api/v1/questions")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("questions", data)
        self.assertGreater(len(data["questions"]), 0)
        first = data["questions"][0]
        self.assertIn("question", first)
        self.assertIn("reference_answer", first)

        # Test limit query param
        res_limit = self.client.get("/api/v1/questions?limit=5")
        self.assertEqual(res_limit.status_code, 200)
        self.assertLessEqual(len(res_limit.json()["questions"]), 5)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Employability Prediction Endpoint
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_predict_employability_valid(self):
        payload = {
            "manner_of_speaking": 4,
            "mental_alertness": 5,
            "self_confidence": 4,
            "ability_to_present_ideas": 3,
            "communication_skills": 4
        }
        res = self.client.post("/api/v1/predict/employability", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["prediction"], ["Employable", "LessEmployable"])
        self.assertIn("probability_employable", data)
        self.assertIn("probability_less_employable", data)
        self.assertAlmostEqual(data["probability_employable"] + data["probability_less_employable"], 1.0, places=2)
        self.assertIn("soft_skill_mean", data)
        self.assertIn("recommendations", data)
        self.assertIsInstance(data["recommendations"], list)

    def test_04_predict_employability_validation(self):
        # Invalid rating > 5
        res = self.client.post("/api/v1/predict/employability", json={
            "manner_of_speaking": 6,
            "mental_alertness": 4,
            "self_confidence": 4,
            "ability_to_present_ideas": 4,
            "communication_skills": 4
        })
        self.assertEqual(res.status_code, 422)

        # Invalid rating < 1
        res2 = self.client.post("/api/v1/predict/employability", json={
            "manner_of_speaking": 0,
            "mental_alertness": 4,
            "self_confidence": 4,
            "ability_to_present_ideas": 4,
            "communication_skills": 4
        })
        self.assertEqual(res2.status_code, 422)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Text Assessment Endpoint
    # ─────────────────────────────────────────────────────────────────────────
    def test_05_assess_text_valid(self):
        payload = {
            "question": "What is object oriented programming?",
            "candidate_answer": "Object oriented programming is a paradigm based on objects containing data and code.",
            "reference_answer": "Object-oriented programming is a programming paradigm based on the concept of objects."
        }
        res = self.client.post("/api/v1/assess/text", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("relevance_score", data)
        self.assertIn("question_alignment_score", data)
        self.assertIn("keyword_overlap", data)
        self.assertIn("structure_score", data)
        self.assertIn("quality_score", data)
        self.assertIn("performance_tier", data)
        self.assertIn("feedback", data)

    def test_06_assess_text_validation(self):
        # Answer too short (< 3 words)
        res = self.client.post("/api/v1/assess/text", json={
            "question": "What is Python?",
            "candidate_answer": "a language",
            "reference_answer": "Python is a language."
        })
        self.assertEqual(res.status_code, 400)
        self.assertIn("minimum 3 words", res.json()["detail"])

        # Empty question
        res2 = self.client.post("/api/v1/assess/text", json={
            "question": "   ",
            "candidate_answer": "Python is a dynamic programming language.",
            "reference_answer": "Python is a language."
        })
        self.assertEqual(res2.status_code, 400)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Audio Assessment Endpoint (/api/v1/assess/audio)
    # ─────────────────────────────────────────────────────────────────────────
    def test_07_assess_audio_valid(self):
        res = self.client.post(
            "/api/v1/assess/audio",
            files={"file": ("test.wav", self.sample_wav_bytes, "audio/wav")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("audio_metrics", data)
        self.assertIn("speech_feedback", data)
        metrics = data["audio_metrics"]
        self.assertIn("duration_sec", metrics)
        self.assertIn("pause_ratio", metrics)
        self.assertIn("mean_energy_rms", metrics)

    def test_08_assess_audio_validation(self):
        # Empty audio
        res_empty = self.client.post(
            "/api/v1/assess/audio",
            files={"file": ("empty.wav", b"", "audio/wav")}
        )
        self.assertEqual(res_empty.status_code, 400)

        # Unsupported extension
        res_fmt = self.client.post(
            "/api/v1/assess/audio",
            files={"file": ("test.txt", b"plain text", "text/plain")}
        )
        self.assertEqual(res_fmt.status_code, 415)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Multimodal Assessment Endpoint (/api/v1/assess/multimodal)
    # ─────────────────────────────────────────────────────────────────────────
    def test_09_assess_multimodal_valid_with_audio(self):
        res = self.client.post(
            "/api/v1/assess/multimodal",
            data={
                "question": "What is encapsulation in OOP?",
                "reference_answer": "Encapsulation bundles data and methods that operate on that data within a single class.",
                "candidate_answer": "Encapsulation restricts direct access to some of an object's components and prevents unauthorized modification.",
                "manner_of_speaking": 4,
                "mental_alertness": 4,
                "self_confidence": 4,
                "ability_to_present_ideas": 4,
                "communication_skills": 4
            },
            files={"audio_file": ("answer.wav", self.sample_wav_bytes, "audio/wav")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["overall_status"], "Evaluation Complete")
        self.assertIn("employability_result", data)
        self.assertIn("text_evaluation", data)
        self.assertIn("speech_acoustic_evaluation", data)
        self.assertIsNotNone(data["speech_acoustic_evaluation"])

    def test_10_assess_multimodal_valid_without_audio(self):
        res = self.client.post(
            "/api/v1/assess/multimodal",
            data={
                "question": "What is polymorphism?",
                "candidate_answer": "Polymorphism allows methods to do different things based on the object acting upon them.",
                "manner_of_speaking": 4,
                "mental_alertness": 4,
                "self_confidence": 4,
                "ability_to_present_ideas": 4,
                "communication_skills": 4
            }
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsNone(data["speech_acoustic_evaluation"])

    def test_11_assess_multimodal_validation(self):
        # Short candidate answer
        res = self.client.post(
            "/api/v1/assess/multimodal",
            data={
                "question": "What is encapsulation?",
                "candidate_answer": "data hiding",
                "manner_of_speaking": 4
            }
        )
        self.assertEqual(res.status_code, 400)

        # Invalid ratings out of range
        res2 = self.client.post(
            "/api/v1/assess/multimodal",
            data={
                "question": "What is encapsulation?",
                "candidate_answer": "Encapsulation hides internal object state.",
                "manner_of_speaking": 7
            }
        )
        self.assertEqual(res2.status_code, 422)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Audio Transcription with faster-whisper (/api/audio/transcribe)
    # ─────────────────────────────────────────────────────────────────────────
    def test_12_transcribe_audio_missing_whisper_service(self):
        """Verify 503 when Whisper model fails to load."""
        with patch.object(api.transcription_service, "_model", None):
            with patch.object(api.transcription_service, "transcribe", side_effect=RuntimeError("Model not loaded")):
                res = self.client.post(
                    "/api/audio/transcribe",
                    files={"file": ("test.wav", self.sample_wav_bytes, "audio/wav")}
                )
        self.assertEqual(res.status_code, 502)
        self.assertIn("Transcription failed", res.json()["detail"])

    def test_13_transcribe_audio_whisper_mocked_success(self):
        """Verify successful transcription with mocked Whisper model."""
        mock_result = {
            "transcript": "This is a transcribed test response via faster-whisper.",
            "detected_language": "en",
            "language_probability": 0.99,
        }
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result):
            res = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("test.wav", self.sample_wav_bytes, "audio/wav")}
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["transcript"], "This is a transcribed test response via faster-whisper.")
        self.assertEqual(data["provider"], "faster-whisper")
        self.assertEqual(data["model"], "faster-whisper-small")
        self.assertEqual(data["detected_language"], "en")

    def test_14_transcribe_audio_uses_small_model(self):
        """Verify the faster-whisper-small model is used."""
        mock_result = {
            "transcript": "Transcribed text.",
            "detected_language": "en",
            "language_probability": 0.95,
        }
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result) as mock_transcribe:
            res = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("test.wav", self.sample_wav_bytes, "audio/wav")}
            )
        self.assertEqual(res.status_code, 200)
        # Verify transcribe was called with the audio bytes
        call_args = mock_transcribe.call_args
        self.assertIsNotNone(call_args)
        self.assertEqual(len(call_args[0]), 2)  # audio_bytes, file_ext

    def test_15_transcribe_audio_empty_audio_400(self):
        """Verify 400 when empty audio is uploaded."""
        res = self.client.post(
            "/api/audio/transcribe",
            files={"file": ("test.wav", b"", "audio/wav")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_16_transcribe_audio_no_speech_422(self):
        """Verify 422 when Whisper detects no speech."""
        mock_result = {
            "transcript": "",
            "detected_language": "en",
            "language_probability": 0.5,
        }
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result):
            res = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("test.wav", self.sample_wav_bytes, "audio/wav")}
            )
        self.assertEqual(res.status_code, 422)
        self.assertIn("no speech detected", res.json()["detail"].lower())

    def test_17_transcribe_audio_detected_language(self):
        """Verify detected_language is returned in response."""
        mock_result = {
            "transcript": "Hello world.",
            "detected_language": "en",
            "language_probability": 0.98,
        }
        with patch.object(api.transcription_service, "transcribe", return_value=mock_result):
            res = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("test.wav", self.sample_wav_bytes, "audio/wav")}
            )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("detected_language", data)
        self.assertEqual(data["detected_language"], "en")
        self.assertIn("language_probability", data)

    def test_18_transcribe_audio_whisper_error_502(self):
        """Verify 502 when Whisper transcription raises an exception."""
        with patch.object(api.transcription_service, "transcribe", side_effect=Exception("Whisper model error")):
            res = self.client.post(
                "/api/audio/transcribe",
                files={"file": ("test.wav", self.sample_wav_bytes, "audio/wav")}
            )
        self.assertEqual(res.status_code, 502)
        self.assertIn("Transcription failed", res.json()["detail"])

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Multimodal Report Endpoint (/api/interview/multimodal-report)
    # ─────────────────────────────────────────────────────────────────────────
    def test_19_multimodal_report_success(self):
        res = self.client.post(
            "/api/interview/multimodal-report",
            data={
                "question": "What is an abstract class in Python?",
                "reference_answer": "An abstract class serves as a blueprint that cannot be directly instantiated and defines abstract methods.",
                "corrected_transcript": "Um an abstract class defines common structure and cannot be directly instantiated, like you know.",
            },
            files={"file": ("interview.wav", self.sample_wav_bytes, "audio/wav")}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["assessment_type"], "Independent Text and Audio Assessment")
        self.assertNotIn("combined_score", data)
        self.assertIn("interview_information", data)
        self.assertIn("text_assessment", data)
        self.assertIsInstance(data["text_assessment_score"], float)
        self.assertEqual(data["text_assessment_score_status"], "Available")
        self.assertEqual(data["overall_multimodal_assessment"]["status"], "Available")
        self.assertIn("voice_assessment", data)
        self.assertIn("feedback", data)
        voice = data["voice_assessment"]
        # Verify filler word count detected 'um', 'like', 'you know'
        self.assertGreaterEqual(voice["filler_word_count"], 1)
        self.assertGreater(voice["speaking_rate_wpm"], 0)

    def test_20_multimodal_report_validation(self):
        # Short transcript < 3 words
        res = self.client.post(
            "/api/interview/multimodal-report",
            data={
                "question": "What is recursion?",
                "reference_answer": "Recursion is a function calling itself.",
                "corrected_transcript": "a function",
            },
            files={"file": ("interview.wav", self.sample_wav_bytes, "audio/wav")}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("minimum 3 words", res.json()["detail"])

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Phase 1 Bug Fix Regression Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_bug1_final_report_generates_when_all_questions_assessed(self):
        """Final report should succeed when text_status and voice_status are completed."""
        # Create student with unique roll number
        import uuid as _uuid
        unique_suffix = _uuid.uuid4().hex[:8]
        student_res = self.client.post("/api/students", json={
            "full_name": f"Regression Test Student {unique_suffix}",
            "roll_number": f"REG-TEST-{unique_suffix}",
            "course": "B.Tech",
            "department": "CSE",
        })
        self.assertEqual(student_res.status_code, 201, student_res.text)
        student_id = student_res.json()["id"]

        # Create session with 1 question
        session_res = self.client.post("/api/sessions", json={
            "student_id": student_id, "domain": "Python", "difficulty": "Beginner",
            "mode": "text", "question_count": 1,
        })
        self.assertEqual(session_res.status_code, 201, session_res.text)
        session = session_res.json()
        session_id = session["id"]
        q_item = session["questions"][0]
        question = q_item["question"]

        # Save response + assess text (persist_assessment auto-completes mode status)
        self.client.put(f"/api/sessions/{session_id}/questions/1/response", json={
            "answer_text": question["reference_answer"] or "A test answer.",
            "response_mode": "text",
        })
        assess_res = self.client.post("/api/v1/assess/text", json={
            "question": question["question_text"],
            "candidate_answer": question["reference_answer"] or "A test answer.",
            "reference_answer": question["reference_answer"] or "A test answer.",
            "session_id": session_id, "position": 1,
        })
        self.assertEqual(assess_res.status_code, 200, assess_res.text)

        # Verify text_status is now completed (auto-advanced by persist_assessment)
        refreshed = self.client.get(f"/api/sessions/{session_id}").json()
        self.assertEqual(refreshed["text_status"], "completed")

        # Final report should now succeed
        report_res = self.client.post(f"/api/sessions/{session_id}/final-report")
        self.assertEqual(report_res.status_code, 200, report_res.text)
        report_data = report_res.json()
        self.assertEqual(report_data["report"]["assessment"]["completion_percentage"], 100.0)

    def test_bug6_voice_mode_status_syncs_when_all_assessed(self):
        """Voice mode_status should advance to completed when all voice responses are assessed."""
        import uuid as _uuid
        unique_suffix = _uuid.uuid4().hex[:8]
        student_res = self.client.post("/api/students", json={
            "full_name": f"Voice Sync Student {unique_suffix}",
            "roll_number": f"REG-VOICE-{unique_suffix}",
            "course": "B.Tech",
            "department": "CSE",
        })
        self.assertEqual(student_res.status_code, 201)
        student_id = student_res.json()["id"]

        session_res = self.client.post("/api/sessions", json={
            "student_id": student_id, "domain": "Python", "difficulty": "Beginner",
            "mode": "voice", "question_count": 1,
        })
        self.assertEqual(session_res.status_code, 201)
        session = session_res.json()
        session_id = session["id"]
        q_item = session["questions"][0]
        question = q_item["question"]

        # Persist voice assessment via multimodal-report
        self.client.put(f"/api/sessions/{session_id}/questions/1/response", json={
            "original_transcript": "This is a raw voice transcript for testing.",
            "corrected_transcript": "This is a reviewed voice transcript for testing.",
            "response_mode": "voice",
        })
        report_res = self.client.post("/api/interview/multimodal-report", data={
            "question": question["question_text"],
            "reference_answer": question["reference_answer"] or "",
            "corrected_transcript": "This is a reviewed voice transcript for testing.",
            "original_transcript": "This is a raw voice transcript for testing.",
            "session_id": session_id, "position": "1",
        }, files={"file": ("interview.wav", self.sample_wav_bytes, "audio/wav")})
        self.assertEqual(report_res.status_code, 200, report_res.text)

        # Verify voice_status is completed
        refreshed = self.client.get(f"/api/sessions/{session_id}").json()
        self.assertEqual(refreshed["voice_status"], "completed")

    def test_bug9_analytics_returns_correct_field_names(self):
        """Analytics endpoint should return fields expected by the frontend."""
        res = self.client.get("/api/analytics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        # Frontend expects these exact keys
        self.assertIn("total_students", data)
        self.assertIn("total_sessions", data)
        self.assertIn("completed_sessions", data)
        self.assertIn("in_progress_sessions", data)
        self.assertIn("average_assessment_score", data)
        # Legacy keys should still work
        self.assertIn("students", data)
        self.assertIn("sessions", data)
        self.assertIn("completed_sessions", data)

    def test_session_creation_validates_user_id_isolation(self):
        """Session creation should reject a user_id that doesn't match the student's user_id."""
        import uuid as _uuid
        unique_suffix = _uuid.uuid4().hex[:8]
        # Create a student WITHOUT a user (legacy student)
        student_res = self.client.post("/api/students", json={
            "full_name": f"Legacy Student {unique_suffix}",
            "roll_number": f"REG-LEGACY-{unique_suffix}",
            "course": "B.Tech",
            "department": "CSE",
        })
        self.assertEqual(student_res.status_code, 201)
        student_id = student_res.json()["id"]

        # Creating a session with a user_id that doesn't match should fail
        res = self.client.post("/api/sessions", json={
            "student_id": student_id, "user_id": "nonexistent-user-id",
            "domain": "Python", "difficulty": "Beginner",
            "mode": "text", "question_count": 1,
        })
        self.assertEqual(res.status_code, 403, res.text)


if __name__ == "__main__":
    unittest.main()
