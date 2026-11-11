"""End-to-end linkage audit using a temporary SQLite database."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import requests
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src import database
from src.test_multimodal_complete import make_wav
from backend import main as api


class RequestsResponseAdapter:
    def __init__(self, response):
        self._response = response
        self.ok = response.status_code < 400
        self.status_code = response.status_code
        self.content = response.content
        self.text = response.text

    def json(self):
        return self._response.json()

    def raise_for_status(self):
        self._response.raise_for_status()


class InterviewAssessmentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.main import app
        cls.app = app

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="interviewai-audit-")
        self.db_path = Path(self.temp_dir.name) / "audit.sqlite3"
        database.configure_database(f"sqlite:///{self.db_path.as_posix()}")
        database.initialize_database()
        self.client = TestClient(self.app)

    def tearDown(self):
        database.engine.dispose()
        self.temp_dir.cleanup()
        database.configure_database()

    def _client_request(self, method, url, **kwargs):
        parsed = urlparse(url)
        response = self.client.request(method, parsed.path, params=kwargs.get("params"),
                                       json=kwargs.get("json"), data=kwargs.get("data"),
                                       files=kwargs.get("files"))
        return RequestsResponseAdapter(response)

    def _client_get(self, url, **kwargs):
        return self._client_request("GET", url, **kwargs)

    def test_student_directory_profile_session_text_voice_report_and_isolation(self):
        # Student registration and directory lookup
        created_student = self.client.post("/api/auth/register", json={
            "email": "audit-one@example.test", "password": "AuditPass123!",
            "full_name": "Audit Candidate One", "roll_number": "AUDIT-001",
            "registration_number": "AUDIT-REG-001", "course": "B.Tech",
            "department": "Computer Science", "semester_year": "Final year",
            "skills": ["Python", "SQL"], "target_job_role": "ML Engineer",
            "preferred_domain": "Machine Learning",
        })
        self.assertEqual(created_student.status_code, 201, created_student.text)
        reg_data = created_student.json()
        student = reg_data["student"]
        self.assertTrue(student["id"])
        self.assertEqual(len(student["id"]), 24)
        self.assertEqual(self.client.get("/api/students", params={"search": student["id"]}).json()["students"][0]["id"], student["id"])
        self.assertEqual(self.client.get("/api/students", params={"search": "AUDIT-001"}).json()["students"][0]["id"], student["id"])
        profile = self.client.get(f"/api/students/{student['id']}").json()
        self.assertEqual(profile["full_name"], "Audit Candidate One")

        # AppTest verifies that the actual directory and profile pages read that API record.
        frontend = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
        with patch.object(requests, "get", side_effect=self._client_get), \
             patch.object(requests, "request", side_effect=self._client_request):
            directory = AppTest.from_file(str(frontend), default_timeout=40)
            directory.session_state["nav"] = "Student Directory"
            directory.session_state["is_authenticated"] = True
            directory.session_state["user"] = {"id": "test-user", "email": "test@example.test", "role": "student", "account_status": "active"}
            directory.session_state["student"] = {"id": student["id"], "full_name": "Audit Candidate One"}
            directory.run(timeout=40)
            self.assertFalse(directory.exception, [e.message for e in directory.exception])
            self.assertTrue(directory.dataframe)
            self.assertIn("View profile", [button.label for button in directory.button])
            next(button for button in directory.button if button.label == "View profile").click().run(timeout=40)
            self.assertIn("Audit Candidate One", [node.value for node in directory.subheader])
            profile_page = AppTest.from_file(str(frontend), default_timeout=40)
            profile_page.session_state["nav"] = "Student Profile"
            profile_page.session_state["is_authenticated"] = True
            profile_page.session_state["user"] = {"id": "test-user", "email": "test@example.test", "role": "student", "account_status": "active"}
            profile_page.session_state["student"] = {"id": student["id"], "full_name": "Audit Candidate One"}
            profile_page.session_state["profile_student_id"] = student["id"]
            profile_page.run(timeout=40)
            self.assertFalse(profile_page.exception, [e.message for e in profile_page.exception])
            self.assertIn("Audit Candidate One", [node.value for node in profile_page.subheader])
            next(button for button in profile_page.button if button.label == "Create New Assessment").click().run(timeout=40)
            self.assertEqual(profile_page.session_state["assessment_student_id"], student["id"])
            self.assertEqual(profile_page.session_state["nav"], "Create Interview")
            profile_page.selectbox(key="flow_assessment_domain").select("Machine Learning").run(timeout=40)
            submit_assessment = next(button for button in profile_page.button if button.label == "Create assessment and start")
            submit_assessment.click().run(timeout=40)
            self.assertFalse(profile_page.exception, [e.message for e in profile_page.exception])
            session_id = profile_page.session_state["active_interview_session_id"]
            # Creating the assessment reruns into Interview Workspace, where
            # selectbox[0] is the question position (an integer), not the
            # student selector that was rendered on the setup page.
            self.assertEqual(profile_page.session_state["nav"], "Active Workspace")
            self.assertEqual(profile_page.session_state["assessment_student_id"], student["id"])
            self.assertFalse(profile_page.exception, [e.message for e in profile_page.exception])

        # Create one mixed-mode session from the selected student's identifier.
        auth_headers = {"Authorization": f"Bearer {reg_data['access_token']}"}
        created_session = self.client.get(f"/api/sessions/{session_id}", headers=auth_headers)
        self.assertEqual(created_session.status_code, 200, created_session.text)
        session = created_session.json()
        self.assertEqual(session["title"], "Machine Learning Interview")
        self.assertEqual(session["student_id"], student["id"])
        self.assertEqual(len(session["questions"]), 5)
        initial_order = [(row["position"], row["question"]["id"]) for row in session["questions"]]
        active = self.client.get("/api/sessions", params={"status": "in_progress", "student_id": student["id"]}, headers=auth_headers).json()["sessions"]
        self.assertTrue(any(row["id"] == session_id for row in active))

        # Real text scoring persists every answer and assessment to its assigned question.
        for item in session["questions"]:
            question = item["question"]
            answer = question["reference_answer"]
            evaluated = self.client.post("/api/v1/assess/text", json={
                "question": question["question_text"], "candidate_answer": answer,
                "reference_answer": answer, "session_id": session_id,
                "position": item["position"],
            })
            self.assertEqual(evaluated.status_code, 200, evaluated.text)
            self.assertIsInstance(evaluated.json()["quality_score"], float)
        text_done = self.client.post(f"/api/sessions/{session_id}/modes/text/complete")
        self.assertEqual(text_done.status_code, 200, text_done.text)

        # Whisper transcription is mocked; uploaded synthetic WAVs are decoded
        # and analyzed by the actual Librosa paths.
        wav_bytes, _ = make_wav(duration=2.5, sample_rate=16000, frequency=180)
        mock_transcript = "The model learns from labeled training examples and evaluates generalization."
        with patch.object(api.transcription_service, "transcribe", return_value={
            "transcript": mock_transcript,
            "detected_language": "en",
            "language_probability": 0.99,
        }):
            for item in session["questions"]:
                question = item["question"]
                position = item["position"]
                response_id = "RESP-" + hashlib.sha256(f"{session_id}:{position}".encode()).hexdigest()[:12].upper()
                transcript_response = self.client.post("/api/audio/transcribe", data={"response_id": response_id},
                    files={"file": (f"question-{position}.wav", wav_bytes, "audio/wav")})
                self.assertEqual(transcript_response.status_code, 200, transcript_response.text)
                original = transcript_response.json()["transcript"]
                corrected = original + " I would validate the model on unseen data."
                report_response = self.client.post("/api/interview/multimodal-report", data={
                    "question": question["question_text"], "reference_answer": question["reference_answer"],
                    "corrected_transcript": corrected, "original_transcript": original,
                    "response_id": response_id, "session_id": session_id,
                    "position": str(position),
                }, files={"file": (f"question-{position}.wav", wav_bytes, "audio/wav")})
                self.assertEqual(report_response.status_code, 200, report_response.text)
                metrics = report_response.json()["voice_assessment"]
                self.assertGreater(metrics["duration_sec"], 2.0)
                self.assertEqual(len(metrics["mfcc_mean"]), 13)
        voice_done = self.client.post(f"/api/sessions/{session_id}/modes/voice/complete")
        self.assertEqual(voice_done.status_code, 200, voice_done.text)

        # The session returns the same positions/IDs after changing modes and after a DB reopen.
        after_voice = self.client.get(f"/api/sessions/{session_id}").json()
        self.assertEqual(initial_order, [(row["position"], row["question"]["id"]) for row in after_voice["questions"]])
        self.assertEqual(after_voice["text_status"], "completed")
        self.assertEqual(after_voice["voice_status"], "completed")
        for item in after_voice["questions"]:
            self.assertEqual(item["text_response"]["answer_text"], item["question"]["reference_answer"])
            self.assertEqual(item["voice_response"]["original_transcript"], "The model learns from labeled training examples and evaluates generalization.")
            self.assertNotEqual(item["voice_response"]["original_transcript"], item["voice_response"]["corrected_transcript"])
            self.assertGreater(item["voice_response"]["audio_measurements"]["duration_sec"], 2.0)

        with database.db_session() as db:
            text_responses = db.query(database.InterviewResponse).filter_by(session_id=session_id, response_mode="text").all()
            voice_responses = db.query(database.InterviewResponse).filter_by(session_id=session_id, response_mode="voice").all()
            self.assertEqual(len(text_responses), 5)
            self.assertEqual(len(voice_responses), 5)
            for response in text_responses + voice_responses:
                self.assertEqual(response.student_id, student["id"])
                self.assertEqual(response.session_id, session_id)
                self.assertEqual(response.session_question.session_id, session_id)
                self.assertEqual(response.question_id, response.session_question.question_id)
            self.assertEqual(sum(bool(r.text_assessment) for r in text_responses), 5)
            self.assertEqual(sum(bool(r.audio_assessment) for r in voice_responses), 5)

        final = self.client.post(f"/api/sessions/{session_id}/final-report")
        self.assertEqual(final.status_code, 200, final.text)
        report = final.json()["report"]
        self.assertEqual(report["student"]["student_id"], student["id"])
        self.assertEqual(report["assessment"]["assessment_id"], session_id)
        self.assertEqual(report["assessment"]["completion_percentage"], 100.0)
        self.assertEqual(len(report["question_results"]), 5)
        self.assertIsNone(report["multimodal_summary"]["combined_numeric_score"])
        self.assertEqual(self.client.get(f"/api/sessions/{session_id}/reports", headers=auth_headers).json()["reports"][0]["student_id"], student["id"])

        # Restarting the SQLAlchemy engine preserves session, transcripts, and report.
        database.engine.dispose()
        database.configure_database(f"sqlite:///{self.db_path.as_posix()}")
        database.initialize_database()
        restored = self.client.get(f"/api/students/{student['id']}").json()
        self.assertEqual(restored["sessions"][0]["id"], session_id)
        self.assertEqual(restored["sessions"][0]["reports"][0]["version"], 1)

        # A second student cannot see the first student's sessions or report rows.
        second = self.client.post("/api/students", json={"full_name": "Audit Candidate Two", "roll_number": "AUDIT-002"})
        self.assertEqual(second.status_code, 201, second.text)
        second_id = second.json()["id"]
        self.assertEqual(self.client.get(f"/api/students/{second_id}").json()["sessions"], [])
        self.assertEqual(self.client.get("/api/sessions", params={"student_id": second_id}, headers=auth_headers).json()["sessions"], [])
        self.assertEqual(self.client.get(f"/api/sessions/{session_id}/reports", headers=auth_headers).json()["reports"][0]["student_id"], student["id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
