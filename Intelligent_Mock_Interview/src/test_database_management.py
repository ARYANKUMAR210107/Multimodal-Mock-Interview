"""Integration checks for persistent interview management APIs."""
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src import database


class PersistentManagementAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.main import app
        cls.app = app

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="interviewai-db-")
        self.db_path = Path(self.temp_dir.name) / "test.sqlite3"
        database.configure_database(f"sqlite:///{self.db_path.as_posix()}")
        database.initialize_database()
        self.client = TestClient(self.app)

    def tearDown(self):
        database.engine.dispose()
        self.temp_dir.cleanup()
        database.configure_database()

    def create_student(self):
        response = self.client.post("/api/students", json={
            "full_name": "Example Candidate", "roll_number": "R-001",
            "course": "Computer Science", "department": "Engineering",
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_student_conflicts_are_reported(self):
        self.create_student()
        duplicate = self.client.post("/api/students", json={"full_name": "Second", "roll_number": "R-001"})
        self.assertEqual(duplicate.status_code, 409)
        unnamed_duplicate = self.client.post("/api/students", json={"full_name": "Candidate Without IDs"})
        self.assertEqual(unnamed_duplicate.status_code, 201)
        accidental_duplicate = self.client.post("/api/students", json={"full_name": "Candidate Without IDs"})
        self.assertEqual(accidental_duplicate.status_code, 409)

    def test_session_response_assessment_and_restart_persistence(self):
        student = self.create_student()
        session_response = self.client.post("/api/sessions", json={
            "student_id": student["id"], "domain": "Python", "difficulty": "Beginner",
            "mode": "text", "question_count": 1,
        })
        self.assertEqual(session_response.status_code, 201, session_response.text)
        session = session_response.json()
        question = session["questions"][0]["question"]
        saved = self.client.put(f"/api/sessions/{session['id']}/questions/1/response", json={
            "answer_text": "A mutable list can change while an immutable tuple cannot.",
            "response_mode": "text",
        })
        self.assertEqual(saved.status_code, 200, saved.text)
        assessed = self.client.post("/api/v1/assess/text", json={
            "question": question["question_text"],
            "reference_answer": question["reference_answer"] or "A Python answer.",
            "candidate_answer": "A mutable list can change while an immutable tuple cannot.",
            "session_id": session["id"], "position": 1,
        })
        self.assertEqual(assessed.status_code, 200, assessed.text)
        self.assertGreaterEqual(assessed.json()["quality_score"], 0)
        database.engine.dispose()
        database.configure_database(f"sqlite:///{self.db_path.as_posix()}")
        database.initialize_database()
        restored = self.client.get(f"/api/sessions/{session['id']}")
        self.assertEqual(restored.status_code, 200, restored.text)
        item = restored.json()["questions"][0]
        self.assertEqual(item["status"], "assessed")
        self.assertEqual(item["response"]["answer_text"], "A mutable list can change while an immutable tuple cannot.")
        self.assertIsNotNone(item["response"]["text_assessment"]["quality_score"])

    def test_question_bank_seed_has_authorship_and_domain_availability(self):
        response = self.client.get("/api/questions")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["questions"])
        authored = [q for q in data["questions"] if "Authored" in q["provenance"]]
        self.assertGreaterEqual(len(authored), 30)
        self.assertTrue(all(q["reference_answer"] for q in authored))

    def test_voice_report_keeps_transcript_and_audio_measurements_separate(self):
        from backend.management_api import persist_assessment

        # Register a user to get an auth token and linked student profile
        reg = self.client.post("/api/auth/register", json={
            "email": "voice@example.test", "password": "VoicePass123!",
            "full_name": "Voice Tester", "roll_number": "V-001",
            "course": "Computer Science", "department": "Engineering",
        })
        self.assertEqual(reg.status_code, 201, reg.text)
        token = reg.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}

        student = reg.json()["student"]

        created = self.client.post("/api/sessions", json={
            "student_id": student["id"], "domain": "Python", "difficulty": "Any",
            "mode": "voice", "question_count": 1,
        }, headers=auth_headers)
        self.assertEqual(created.status_code, 201, created.text)
        session_id = created.json()["id"]
        report = {
            "interview_information": {
                "question": "A question", "original_transcript": "Raw words",
                "corrected_transcript": "Reviewed words",
            },
            "text_assessment": {"relevance_score": .4, "question_alignment_score": .5,
                                "keyword_overlap": .3, "structure_score": .6,
                                "quality_score": .45, "strengths": [], "improvements": []},
            "voice_assessment": {"duration_sec": 4.2, "mean_pitch_hz": 130.0, "mfcc_mean": [0.1, 0.2]},
        }
        persist_assessment(session_id, 1, "voice", report)
        restored = self.client.get(f"/api/sessions/{session_id}").json()
        response = restored["questions"][0]["response"]
        self.assertEqual(response["original_transcript"], "Raw words")
        self.assertEqual(response["corrected_transcript"], "Reviewed words")
        self.assertFalse(response["audio_retained"])
        self.assertEqual(response["audio_measurements"]["mean_pitch_hz"], 130.0)
        finished = self.client.post(f"/api/sessions/{session_id}/modes/voice/complete")
        self.assertEqual(finished.status_code, 200, finished.text)
        final = self.client.post(f"/api/sessions/{session_id}/final-report")
        self.assertEqual(final.status_code, 200, final.text)
        reports = self.client.get(f"/api/sessions/{session_id}/reports", headers=auth_headers).json()["reports"]
        self.assertEqual(len(reports), 1)

    def test_full_five_question_student_assessment_and_isolation(self):
        from backend.management_api import persist_assessment
        first_student = self.create_student()
        created = self.client.post("/api/sessions", json={
            "student_id": first_student["id"], "title": "ML Practice Assessment",
            "domain": "Machine Learning", "difficulty": "Any", "mode": "mixed",
            "question_count": 5,
        })
        self.assertEqual(created.status_code, 201, created.text)
        session = created.json()
        original_order = [(item["position"], item["question"]["id"]) for item in session["questions"]]

        for item in session["questions"]:
            question = item["question"]
            answer = question["reference_answer"] or "A model learns useful patterns from representative examples."
            evaluated = self.client.post("/api/v1/assess/text", json={
                "question": question["question_text"], "candidate_answer": answer,
                "reference_answer": question["reference_answer"] or "",
                "session_id": session["id"], "position": item["position"],
            })
            self.assertEqual(evaluated.status_code, 200, evaluated.text)
        text_done = self.client.post(f"/api/sessions/{session['id']}/modes/text/complete")
        self.assertEqual(text_done.status_code, 200, text_done.text)

        reloaded = self.client.get(f"/api/sessions/{session['id']}").json()
        self.assertEqual(original_order, [(item["position"], item["question"]["id"]) for item in reloaded["questions"]])
        for item in reloaded["questions"]:
            persist_assessment(session["id"], item["position"], "voice", {
                "interview_information": {"original_transcript": "Raw transcript", "corrected_transcript": "Reviewed transcript"},
                "text_assessment": {},
                "voice_assessment": {"duration_sec": 5.0, "mean_pitch_hz": 142.0, "mean_energy_rms": .1},
            })
        voice_done = self.client.post(f"/api/sessions/{session['id']}/modes/voice/complete")
        self.assertEqual(voice_done.status_code, 200, voice_done.text)
        final = self.client.post(f"/api/sessions/{session['id']}/final-report")
        self.assertEqual(final.status_code, 200, final.text)
        report = final.json()["report"]
        self.assertEqual(report["student"]["student_id"], first_student["id"])
        self.assertEqual(len(report["question_results"]), 5)
        self.assertEqual(report["assessment"]["completion_percentage"], 100.0)
        self.assertIsNone(report["multimodal_summary"]["combined_numeric_score"])

        second = self.client.post("/api/students", json={"full_name": "Another Candidate", "roll_number": "R-002"})
        self.assertEqual(second.status_code, 201, second.text)
        second_profile = self.client.get(f"/api/students/{second.json()['id']}").json()
        self.assertEqual(second_profile["sessions"], [])
        database.engine.dispose()
        database.configure_database(f"sqlite:///{self.db_path.as_posix()}")
        database.initialize_database()
        first_profile = self.client.get(f"/api/students/{first_student['id']}").json()
        self.assertEqual(first_profile["sessions"][0]["reports"][0]["version"], 1)
        second_profile_after_restart = self.client.get(f"/api/students/{second.json()['id']}").json()
        self.assertEqual(second_profile_after_restart["sessions"], [])


if __name__ == "__main__":
    unittest.main()
