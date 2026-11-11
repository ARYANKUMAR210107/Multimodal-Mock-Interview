"""Frontend API client tests for transcription (mocked, no FastAPI required)."""
import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(os.path.join(BASE_DIR, "src")))

from frontend.services.api_client import APIClient


class TestTranscribeAudio(unittest.TestCase):
    """Mocked tests for APIClient.transcribe_audio — no FastAPI, no real Gemini calls."""

    def setUp(self):
        self.client = APIClient(base_url="http://127.0.0.1:8000")
        self.wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

    def _mock_post(self, status_code=200, json_body=None, raw_text=None, side_effect=None):
        """Helper to mock requests.post via the APIClient.request method."""
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.ok = 200 <= status_code < 300
        if json_body is not None:
            mock_resp.json.return_value = json_body
        else:
            mock_resp.json.side_effect = ValueError("No JSON")
        if raw_text is not None:
            mock_resp.text = raw_text
        else:
            mock_resp.text = str(json_body) if json_body else ""
        mock_post = MagicMock(return_value=mock_resp)
        if side_effect:
            mock_post.side_effect = side_effect
        return mock_post

    def test_transcribe_audio_success(self):
        """Successful transcription returns (True, transcript_dict)."""
        mock_post = self._mock_post(
            status_code=200,
            json_body={"transcript": "Hello, this is a test transcript.", "model": "gemini-3.5-flash-lite", "provider": "gemini"},
        )
        with patch.object(self.client, "post", mock_post):
            ok, result = self.client.transcribe_audio("test.wav", self.wav_bytes, "audio/wav")
        self.assertTrue(ok)
        self.assertEqual(result["transcript"], "Hello, this is a test transcript.")

    def test_transcribe_audio_non_json_success(self):
        """Successful transcription with non-JSON body returns text as transcript."""
        mock_post = self._mock_post(
            status_code=200,
            json_body=None,
            raw_text="Plain text transcript",
        )
        mock_post.return_value.json.side_effect = ValueError("No JSON")
        with patch.object(self.client, "post", mock_post):
            ok, result = self.client.transcribe_audio("test.wav", self.wav_bytes, "audio/wav")
        self.assertTrue(ok)
        self.assertEqual(result["transcript"], "Plain text transcript")

    def test_transcribe_audio_quota_exhausted(self):
        """429 quota error returns (False, error_message) with quota mention."""
        mock_post = self._mock_post(
            status_code=429,
            json_body={"detail": "Gemini API rate limit or quota exceeded."},
        )
        with patch.object(self.client, "post", mock_post):
            ok, result = self.client.transcribe_audio("test.wav", self.wav_bytes, "audio/wav")
        self.assertFalse(ok)
        self.assertIn("quota", result.lower())

    def test_transcribe_audio_invalid_api_key(self):
        """401 invalid key error returns (False, error_message) with key mention."""
        mock_post = self._mock_post(
            status_code=401,
            json_body={"detail": "Invalid Gemini API key or permission denied."},
        )
        with patch.object(self.client, "post", mock_post):
            ok, result = self.client.transcribe_audio("test.wav", self.wav_bytes, "audio/wav")
        self.assertFalse(ok)
        self.assertIn("key", result.lower())

    def test_transcribe_audio_not_configured(self):
        """503 not configured returns (False, error_message) with config mention."""
        mock_post = self._mock_post(
            status_code=503,
            json_body={"detail": "Google Gemini transcription is not configured."},
        )
        with patch.object(self.client, "post", mock_post):
            ok, result = self.client.transcribe_audio("test.wav", self.wav_bytes, "audio/wav")
        self.assertFalse(ok)
        self.assertIn("not configured", result.lower())

    def test_transcribe_audio_timeout(self):
        """Timeout returns (False, timeout message)."""
        import requests
        with patch.object(self.client, "post", side_effect=requests.exceptions.Timeout("timed out")):
            ok, result = self.client.transcribe_audio("test.wav", self.wav_bytes, "audio/wav")
        self.assertFalse(ok)
        self.assertIn("timed out", result.lower())

    def test_transcribe_audio_network_error(self):
        """Network error returns (False, network error message)."""
        import requests
        with patch.object(self.client, "post", side_effect=requests.exceptions.ConnectionError("conn refused")):
            ok, result = self.client.transcribe_audio("test.wav", self.wav_bytes, "audio/wav")
        self.assertFalse(ok)
        self.assertIn("Network", result)

    def test_transcribe_audio_webm_mime_detection(self):
        """WebM magic bytes are auto-detected and filename adjusted."""
        webm_bytes = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        mock_post = self._mock_post(
            status_code=200,
            json_body={"transcript": "webm transcript"},
        )
        with patch.object(self.client, "post", mock_post) as mock_p:
            ok, result = self.client.transcribe_audio("recording.wav", webm_bytes, "audio/wav")
        self.assertTrue(ok)
        # Verify filename was changed to .webm and mime was set to audio/webm
        call_args = mock_p.call_args
        files = call_args.kwargs.get("files")
        self.assertIsNotNone(files)
        filename, _, mime = files["file"]
        self.assertTrue(filename.endswith(".webm"))
        self.assertEqual(mime, "audio/webm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
