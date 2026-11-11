"""
InterviewAI Frontend API Client
================================
Centralized HTTP client for communicating with the FastAPI backend.
Handles request timeouts, error mapping, transparent data parsing,
and authentication token management.
"""
from __future__ import annotations

import os
import requests
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

API_BASE_URL = os.getenv("MOCK_INTERVIEW_API_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TIMEOUT = 30.0
UPLOAD_TIMEOUT = 90.0


class APIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"

    def _auth_headers(self) -> Dict[str, str]:
        """Build Authorization header from Streamlit session state."""
        token = st.session_state.get("access_token")
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = self._url(path)
        kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
        # Inject auth headers unless caller provided their own
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].update(self._auth_headers())
        try:
            return requests.request(method, url, **kwargs)
        except requests.exceptions.RequestException:
            res = requests.Response()
            res.status_code = 503
            res._content = b'{"detail": "Backend server is currently offline."}'
            return res

    def get(self, path: str, **kwargs) -> requests.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> requests.Response:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> requests.Response:
        return self.request("DELETE", path, **kwargs)

    # ── Authentication ────────────────────────────────────────────────────────

    def register(self, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        """Register a new user + student profile. Returns (success, data_or_error)."""
        res = self.post("/api/auth/register", json=payload)
        if res.status_code in (200, 201):
            data = res.json()
            st.session_state["access_token"] = data.get("access_token")
            st.session_state["token_type"] = data.get("token_type", "bearer")
            st.session_state["user"] = data.get("user")
            st.session_state["student"] = data.get("student")
            st.session_state["is_authenticated"] = True
            st.session_state["profile_student_id"] = data.get("student", {}).get("id")
            st.session_state["assessment_student_id"] = data.get("student", {}).get("id")
            return True, data
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    def login(self, email: str, password: str) -> Tuple[bool, Any]:
        """Authenticate user and store session token."""
        res = self.post("/api/auth/login", json={"email": email, "password": password})
        if res.ok:
            data = res.json()
            st.session_state["access_token"] = data.get("access_token")
            st.session_state["token_type"] = data.get("token_type", "bearer")
            st.session_state["user"] = data.get("user")
            st.session_state["is_authenticated"] = True
            student = data.get("student") or data.get("user", {}).get("student")
            if not student:
                me = self.get("/api/auth/me")
                if me.ok:
                    me_data = me.json()
                    data["user"] = me_data.get("user", data.get("user"))
                    st.session_state["user"] = data.get("user")
                    student = me_data.get("student")
                    data["student"] = student
            if student:
                st.session_state["student"] = student
                st.session_state["profile_student_id"] = student.get("id")
                st.session_state["assessment_student_id"] = student.get("id")
            return True, data
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    def logout(self) -> Tuple[bool, Any]:
        """Invalidate the current session and clear local auth state."""
        token = st.session_state.get("access_token")
        if token:
            res = self.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
        else:
            res = type("Resp", (), {"ok": True})()  # mock ok response
        # Clear local state regardless of backend result
        for key in ("access_token", "token_type", "user", "student", "is_authenticated",
                    "profile_student_id", "assessment_student_id", "active_interview_session_id",
                    "nav"):
            st.session_state.pop(key, None)
        st.session_state["is_authenticated"] = False
        st.session_state["nav"] = "Dashboard"
        return True, {"message": "Logged out"}

    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Fetch the authenticated user's profile."""
        res = self.get("/api/auth/me")
        return res.json() if res.ok else None

    def change_password(self, current_password: str, new_password: str) -> Tuple[bool, Any]:
        res = self.post("/api/auth/change-password", json={
            "current_password": current_password,
            "new_password": new_password,
        })
        if res.ok:
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    # ── System Health ────────────────────────────────────────────────────────
    def get_system_health(self) -> Dict[str, Any]:
        try:
            res = self.get("/", timeout=4.0)
            if res.ok:
                data = res.json()
                # Backend returns `transcription_configured` and `transcription_status`
                gemini_ok = bool(
                    data.get("transcription_configured")
                    or data.get("transcription_status") == "configured"
                    or data.get("gemini_configured")  # legacy fallback
                    or data.get("configured")          # legacy fallback
                )
                return {
                    "online": True,
                    "gemini_configured": gemini_ok,
                    "provider": data.get("transcription_provider", data.get("provider", "gemini")),
                    "transcription_model": data.get("transcription_model", ""),
                    "details": data,
                }
        except Exception as e:
            return {"online": False, "gemini_configured": False, "error": str(e)}
        return {"online": False, "gemini_configured": False}

    # ── Students ─────────────────────────────────────────────────────────────
    def get_students(self, search: str = "", limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        params = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        res = self.get("/api/students", params=params)
        return res.json().get("students", []) if res.ok else []

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        res = self.get(f"/api/students/{student_id}")
        return res.json() if res.ok else None

    def create_student(self, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        res = self.post("/api/students", json=payload)
        if res.status_code in (200, 201):
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    def update_student(self, student_id: str, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        res = self.put(f"/api/students/{student_id}", json=payload)
        if res.ok:
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    # ── Question Bank ────────────────────────────────────────────────────────
    def get_questions(self, domain: str = "", difficulty: str = "", search: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        params = {"limit": limit}
        if domain and domain != "All Domains":
            params["domain"] = domain
        if difficulty and difficulty != "All Difficulties":
            params["difficulty"] = difficulty
        if search:
            params["search"] = search
        res = self.get("/api/questions", params=params)
        return res.json().get("questions", []) if res.ok else []

    def get_domains(self) -> List[str]:
        res = self.get("/api/questions/domains")
        if res.ok:
            return res.json().get("domains", [])
        return [
            "Python", "C++", "Java", "Data Structures & Algorithms",
            "DBMS", "Operating Systems", "Computer Networks", "Machine Learning",
            "Artificial Intelligence", "Web Development", "Software Engineering", "HR & Behavioral"
        ]

    def create_question(self, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        res = self.post("/api/questions", json=payload)
        if res.status_code in (200, 201):
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    # ── Interview Sessions ───────────────────────────────────────────────────
    def create_session(self, payload: Dict[str, Any]) -> Tuple[bool, Any]:
        res = self.post("/api/sessions", json=payload)
        if res.status_code in (200, 201):
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    def get_session(self, session_id: str) -> Tuple[bool, Any]:
        res = self.get(f"/api/sessions/{session_id}")
        if res.ok:
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except (ValueError, AttributeError):
            err = res.text
        return False, f"[HTTP {res.status_code}] {err}"

    def get_sessions(self, status: str = "", student_id: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        params = {"limit": limit}
        if status:
            params["status"] = status
        if student_id:
            params["student_id"] = student_id
        res = self.get("/api/sessions", params=params)
        return res.json().get("sessions", []) if res.ok else []

    def save_response_draft(self, session_id: str, position: int, payload: Dict[str, Any]) -> bool:
        res = self.put(f"/api/sessions/{session_id}/questions/{position}/response", json=payload)
        return res.ok

    def complete_session_mode(self, session_id: str, mode: str) -> bool:
        res = self.post(f"/api/sessions/{session_id}/modes/{mode}/complete")
        return res.ok

    def get_final_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        res = self.get(f"/api/sessions/{session_id}/reports/final")
        return res.json() if res.ok else None

    def create_final_report(self, session_id: str) -> Tuple[bool, Any]:
        res = self.post(f"/api/sessions/{session_id}/reports/final")
        if res.ok:
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    # ── Text Assessment ──────────────────────────────────────────────────────
    def assess_text(self, question: str, candidate_answer: str, reference_answer: str = "",
                    session_id: str = "", position: Optional[int] = None) -> Tuple[bool, Any]:
        payload = {
            "question": question,
            "candidate_answer": candidate_answer,
            "reference_answer": reference_answer or question,
        }
        if session_id:
            payload["session_id"] = session_id
        if position is not None:
            payload["position"] = position
        res = self.post("/api/v1/assess/text", json=payload)
        if res.ok:
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    # ── Audio Transcription (Gemini) ─────────────────────────────────────────
    def transcribe_audio(self, filename: str, audio_bytes: bytes, mime_type: Optional[str] = None) -> Tuple[bool, Any]:
        # Auto-detect real MIME from magic bytes if default or empty
        if not mime_type or mime_type == "audio/wav":
            if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
                mime_type = "audio/webm"
                if not filename.endswith(".webm"):
                    filename = os.path.splitext(filename)[0] + ".webm"
            elif audio_bytes.startswith(b"RIFF"):
                mime_type = "audio/wav"
            elif audio_bytes.startswith(b"ID3") or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF):
                mime_type = "audio/mp3"
            elif len(audio_bytes) > 12 and audio_bytes[4:8] == b"ftyp":
                mime_type = "audio/mp4"

        files = {"file": (filename, audio_bytes, mime_type or "audio/wav")}
        try:
            res = self.post("/api/audio/transcribe", files=files, timeout=UPLOAD_TIMEOUT)
            if res.ok:
                try:
                    return True, res.json()
                except ValueError:
                    return True, {"transcript": res.text.strip() or ""}
            try:
                err = res.json().get("detail", res.text)
            except (ValueError, AttributeError):
                err = res.text
            if res.status_code == 503:
                return False, f"Transcription service unavailable: {err}"
            if res.status_code == 422:
                return False, f"Could not transcribe audio: {err}"
            if res.status_code == 415:
                return False, f"Unsupported audio format: {err}"
            return False, err
        except requests.exceptions.Timeout:
            return False, "Finalizing transcript timed out. Please retry with a shorter recording or check CPU load."
        except requests.exceptions.RequestException as e:
            return False, f"Could not reach transcription service: {str(e)}"
        except Exception as e:
            return False, f"Unexpected transcription error: {str(e)}"

    # ── Multimodal Question Assessment ───────────────────────────────────────
    def assess_multimodal(self, question: str, corrected_transcript: str,
                          audio_bytes: bytes, audio_filename: str = "recording.wav",
                          reference_answer: str = "", original_transcript: str = "",
                          session_id: str = "", position: Optional[int] = None) -> Tuple[bool, Any]:
        data = {
            "question": question,
            "corrected_transcript": corrected_transcript,
            "reference_answer": reference_answer,
            "original_transcript": original_transcript,
        }
        if session_id:
            data["session_id"] = session_id
        if position is not None:
            data["position"] = str(position)

        mime = "audio/wav"
        if audio_filename.endswith(".mp3"): mime = "audio/mpeg"
        elif audio_filename.endswith(".m4a"): mime = "audio/mp4"
        elif audio_filename.endswith(".webm"): mime = "audio/webm"

        files = {"file": (audio_filename, audio_bytes, mime)}
        try:
            res = self.post("/api/interview/multimodal-report", data=data, files=files, timeout=UPLOAD_TIMEOUT)
            if res.ok:
                return True, res.json()
            try:
                err = res.json().get("detail", res.text)
            except Exception:
                err = res.text
            return False, err
        except Exception as e:
            return False, f"Assessment request failed: {str(e)}"

    # ── Employability Prediction ─────────────────────────────────────────────
    def predict_employability(self, scores: Dict[str, int]) -> Tuple[bool, Any]:
        payload = {
            "manner_of_speaking": scores.get("manner_of_speaking", 3),
            "mental_alertness": scores.get("mental_alertness", 3),
            "self_confidence": scores.get("self_confidence", 3),
            "ability_to_present_ideas": scores.get("ability_to_present_ideas", 3),
            "communication_skills": scores.get("communication_skills", 3),
        }
        res = self.post("/api/v1/predict/employability", json=payload)
        if res.ok:
            return True, res.json()
        try:
            err = res.json().get("detail", res.text)
        except Exception:
            err = res.text
        return False, err

    # ── Analytics & Metadata ─────────────────────────────────────────────────
    def get_analytics(self) -> Dict[str, Any]:
        res = self.get("/api/analytics")
        return res.json() if res.ok else {}

    def get_model_meta(self) -> Dict[str, Any]:
        res = self.get("/api/v1/models/meta")
        return res.json() if res.ok else {}


client = APIClient()
