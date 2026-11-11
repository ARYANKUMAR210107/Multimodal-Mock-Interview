# InterviewAI Repair Audit (Phase 1)

**Date:** 2026-11-11  
**Workspace:** `C:\Users\Asus\OneDrive\Desktop\intelligent_supervised\Intelligent_Mock_Interview`  
**Mode:** Read-only inspection before edits; DB backup created at `data/backups/interviewai_20261111_203242.sqlite3*`. Git tree already has extensive local modifications/untracked files (see root status) — no destructive reset performed.

## 1. Actual Architecture / Components

- **Frontend:** Streamlit app (`frontend/app.py`) with modular pages in `frontend/pages/`, shared UI in `frontend/components/ui.py`, HTTP client in `frontend/services/api_client.py`, CSS in `frontend/styles/theme.css` (+ legacy `frontend/theme.css`). Sidebar is role-based (admin/student) and currently exposes >6 items; login gate redirects unauthenticated users to `Login`.
- **Backend:** Single FastAPI app (`backend/main.py`, v1.0.0) with CORS `*`, request logging middleware, startup model/resource loading (Gradient Boosting employability, TF-IDF text assessor, Librosa `CandidateAudioAnalyzer`, cached parquet questions), transcription service import (`backend/transcription_service.py`), and mounted persistent router `backend/management_api.py` (`/api`). SQLite initialized via `src/database.initialize_database()` (create_all + additive migrations/seeding).
- **Database:** SQLAlchemy/SQLite `data/interviewai.sqlite3` (WAL present). Tables include `users`, `user_sessions`, `students`, `question_bank`, `interview_sessions`, `session_questions`, `interview_responses`, `text_assessments`, `audio_assessments`, `assessment_reports`, `alembic_version`. Counts observed: users 24, students 143, sessions 116, responses 96. Backup files exist (`interviewai.sqlite3.backup_20261110`, new timestamped backup).
- **Auth:** `src/auth.py` uses bcrypt hashing, token generation, `authenticate_user`, `create_user_session` (24h), token validation. `backend/management_api.py` provides `/api/auth/register`, `/api/auth/login`, `/api/auth/logout`, `/api/auth/me`, `/api/auth/change-password`, role helpers. Registration creates linked `User` + `Student`.
- **Voice/Transcription:** `backend/transcription_service.py` implements singleton `faster-whisper` (`small`, CPU, `int8` from env defaults), decodes via Librosa to 16k mono numpy, transcribes with `beam_size=5`, `vad_filter=True`. `backend/main.py` also contains Gemini/`google-genai` imports and related audio/transcription helpers/endpoints (`/api/audio/transcribe` present) — **implementation overlap/risk of provider confusion** (must confirm primary path and align with Whisper requirement). No WebSocket chunk-transcription endpoint found.
- **Audio analysis:** `src/audio_feature_extractor.py` + Librosa extraction in `backend/main.py` (`_analyze_audio_bytes`: duration, pauses, RMS, YIN pitch stats, MFCC means, WPM/fillers from transcript). Multimodal/report endpoints persist results via `persist_assessment`.
- **NLP/ML:** `src/text_pipeline.py` TF-IDF/Jaccard/structure heuristic scorer (`InterviewTextAssessor`, `calculate_text_assessment_score`), trained employability artifacts in `models/*.joblib` + `employability_model_meta.json`, interview TF-IDF vectorizer artifact. Existing tests/scripts in `src/test_*.py` (not rerun in this audit except syntax compile checks).

## 2. Root Causes Identified (Pre-fix)

### Session loading: "Could not load interview session"
- Frontend `interview_workspace.py` calls `client.get_session(session_id)` and shows this error for any non-OK response (403/404/network/401 mapped by client).
- `GET /api/sessions/{id}` (`management_api.py`) loads session then, **if Authorization Bearer token is present**, validates token and enforces ownership (`student.user_id == session.student_id`) returning 403 "You can only access your own sessions." If header missing/invalid-format, ownership check is skipped (security gap) but expired/invalid token path in helper behavior and client auth state can still break UX.
- `GET /api/sessions` (list) requires auth via `_get_current_user` (401 if missing/invalid) and filters to current student's sessions for non-admins.
- **Login response shape mismatch:** `login_user` returns `TokenResponse(access_token, expires_in, user=_user_dict(user))` with no `student`. `APIClient.login` stores token/user but cannot set `student`, `profile_student_id`, `assessment_student_id` unless nested in `user` — causing downstream session creation/selection issues. Registration does return `student` and sets IDs; `/api/auth/me` returns both.
- Existing DB shows many sessions/students with `user_id NULL` (legacy) — ownership checks against authenticated users may deny access to legacy records unless admin/migration logic handles linkage.

### Voice recording/transcription reliability
- Current workspace voice UI uses Streamlit `st.audio_input`/file upload + manual "Transcribe" button calling HTTP transcription; no live progressive transcript during speaking, no pause/resume, no chunk streaming/WebSocket, no clear mic/recording state machine in browser.
- `backend/main.py` contains Gemini-era transcription code paths/imports alongside faster-whisper service; `/api/audio/transcribe` behavior and model/provider reporting must be verified/cleaned to ensure faster-whisper is primary and errors/states are clear.
- No incremental chunk endpoint (`/ws/...`) exists; faster-whisper has no native token streaming so chunked completed-segment transcription with overlap/dedup is required by spec.

### UI/auth/navigation gaps vs requested UX
- Login/register UI is functional but not the requested simple professional centered card layout/messages.
- Sidebar/navigation does not match requested student flow (Dashboard, Start Interview, My Assessments, My Reports, My Profile, Logout) and contains duplicate/legacy destinations.
- Dashboard/welcome and metrics may not be strictly scoped to logged-in student’s real completed assessments.
- Data isolation/role separation needs hardening across all session/report/question endpoints (not only selective checks).

## 3. Safe Pre-change Checks
- `py -m py_compile` passed for core backend/frontend modules checked.
- SQLite backup copied to `data/backups/interviewai_20261111_203242.sqlite3*`.
- No DB tables dropped/reset; no secrets printed; `.env` not modified.

## 4. Immediate Fix Plan (priority order)
1. Fix auth/session APIs and client state: return student on login/me-consistent flows, enforce/clarify ownership checks, resolve "Could not load interview session" for valid owners, preserve legacy records safely.
2. Implement faster-whisper chunk WebSocket transcription + browser recorder component (progressive transcript, pause/resume, final full-file pass, retry/clear states) and integrate into interview workspace with question-scoped persistence.
3. Redesign login/registration and simplify app navigation/UI to requested structure while preserving existing capabilities and ML/reporting.
4. Add backward-compatible DB/audio persistence changes only with backups/migrations as needed; keep originals and transcripts durable.
5. Run targeted and end-to-end tests (including real audio transcription) and produce final delivery report.
