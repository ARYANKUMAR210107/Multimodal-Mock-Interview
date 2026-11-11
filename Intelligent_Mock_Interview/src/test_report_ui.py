"""UI regressions for multimodal score availability and stale reports."""
import hashlib
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT = Path(__file__).resolve().parents[1]
APP = PROJECT / "frontend" / "app.py"


def context_for(question, digest, transcript):
    raw = "\0".join([question or "", digest or "", transcript or ""])
    return hashlib.sha256(raw.encode()).hexdigest()


class MultimodalReportUiTests(unittest.TestCase):
    def make_app(self, report, digest="audio-digest", transcript="A clear answer with useful details."):
        question = "Explain supervised learning."
        app = AppTest.from_file(str(APP), default_timeout=30)
        app.session_state["nav"] = "Multimodal Report"
        app.session_state["voice_question"] = question
        app.session_state["voice_audio_digest"] = digest
        app.session_state["voice_transcript_editor"] = transcript
        app.session_state["voice_report"] = report
        app.session_state["voice_report_context"] = context_for(question, digest, transcript)
        return app.run(timeout=30)

    def valid_report(self):
        return {
            "response_id": "RESP-0123456789AB",
            "interview_information": {
                "question": "Explain supervised learning.",
                "reference_answer": "Models learn from labeled examples.",
                "corrected_transcript": "A clear answer with useful details.",
            },
            "text_assessment_score": 0.465,
            "text_assessment_score_status": "Available",
            "text_assessment_score_formula": "0.50 * reference + 0.20 * question + 0.15 * Jaccard + 0.15 * structure",
            "text_assessment_score_limitation": "Heuristic practice score.",
            "text_assessment": {
                "quality_score": 0.465,
                "reference_similarity": 0.5,
                "question_alignment_score": 0.4,
                "keyword_overlap": 0.3,
                "structure_score": 0.6,
                "strengths": [],
                "improvements": [],
            },
            "voice_analysis_status": "Available",
            "voice_assessment": {
                "duration_sec": 2.5,
                "speaking_duration_sec": 2.0,
                "pause_ratio": 0.2,
                "pause_duration_sec": 0.5,
                "pause_count": 1,
                "speaking_rate_wpm": 48.0,
                "mean_pitch_hz": 150.0,
                "pitch_std_hz": 18.0,
                "mean_energy_rms": 0.1,
                "mfcc_mean": [0.1] * 13,
            },
            "overall_multimodal_assessment": {
                "status": "Available",
                "explanation": "Text score and voice measurements are both available and reported separately.",
            },
            "feedback": {"strengths": [], "areas_for_improvement": []},
        }

    def test_score_displays_and_invalid_or_stale_data_never_becomes_zero(self):
        app = self.make_app(self.valid_report())
        self.assertFalse(app.exception, [error.message for error in app.exception])
        score_metrics = [metric for metric in app.metric if metric.label == "Text-only practice score"]
        self.assertEqual(score_metrics[0].value, "46.5%")
        self.assertFalse(any("Not calculated" in node.value for node in app.markdown))

        invalid_report = self.valid_report()
        invalid_report["text_assessment_score"] = "0.465"
        app.session_state["voice_report"] = invalid_report
        app.run(timeout=30)
        score_metrics = [metric for metric in app.metric if metric.label == "Text-only practice score"]
        self.assertEqual(score_metrics[0].value, "Not available")
        self.assertFalse(app.exception, [error.message for error in app.exception])

        app.session_state["voice_report_context"] = "stale-context"
        app.run(timeout=30)
        self.assertTrue(app.warning)
        self.assertFalse(any(metric.label == "Text-only practice score" for metric in app.metric))
        self.assertFalse(app.exception, [error.message for error in app.exception])


if __name__ == "__main__":
    unittest.main()
