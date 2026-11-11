# InterviewAI — Bug, Anomaly & Error Report

**Audit Date:** 2026-11-10  
**Inspection Mode:** Static Code Analysis, Terminal Diagnostics & Runtime Probing  

---

## 1. High-Priority Issues

### BUG-01: Module Resolution Failure on Standard CLI Execution
- **Path:** Root execution scripts, `uvicorn`, and tests in `src/`.
- **Evidence:** Running `uvicorn backend.main:app` or `python src/test_database_management.py` directly from project root fails with `ModuleNotFoundError: No module named 'backend'` or `No module named 'src'`.
- **Root Cause:** Python's module resolution path (`sys.path`) does not automatically include `Intelligent_Mock_Interview` unless `PYTHONPATH` is explicitly set in the shell environment.
- **Severity:** High (Impacts local developer onboarding and CI/CD pipelines).
- **Status:** **FAIL (Workaround Verified: Setting `$env:PYTHONPATH` resolves it)**.
- **Recommendation:** Add a top-level `setup.py` / `pyproject.toml` with editable install (`pip install -e .`), or ensure startup scripts load `sys.path.insert(0, ...)` uniformly before module imports.

### BUG-02: Gemini Transcription Quota Dependency
- **Path:** `backend/main.py:270` (`_transcribe_audio_with_gemini`)
- **Evidence:** Live calls to `/api/audio/transcribe` depend entirely on active Google AI Studio billing/quota. When free-tier rate limits or key quotas are exhausted, endpoint returns HTTP 429/502.
- **Severity:** High (Core voice workflow blocks if external API quota runs out).
- **Status:** **PARTIALLY WORKING / NOT VERIFIED (Live external call blocked by API quota; error handling verified)**.
- **Recommendation:** Implement a local fallback automatic speech recognition (ASR) engine (e.g., OpenAI Whisper tiny/base via `faster-whisper` or PyTorch) for offline mock interviews when external Gemini API is unreachable.

---

## 2. Medium-Priority Issues

### BUG-03: Acoustic Feature Contract Discrepancy
- **Path:** `src/audio_feature_extractor.py:205` vs `src/test_multimodal_complete.py:44`
- **Evidence:** `CandidateAudioAnalyzer.analyze_audio_file` still generates and returns heuristic keys `'estimated_fluency_score'` and `'estimated_prosody_score'`. While `backend/main.py:256` explicitly pops these keys before returning the JSON response, the unit test for `CandidateAudioAnalyzer` asserts they should not be emitted by the extractor at all.
- **Severity:** Medium (Causes 1 unit test failure out of 55).
- **Status:** **FAIL**.
- **Recommendation:** Remove the unvalidated heuristic fluency/prosody calculation from `CandidateAudioAnalyzer` in `src/audio_feature_extractor.py` and document descriptive raw features only (pause ratio, pitch std, RMS energy).

### BUG-04: Streamlit UI Deprecation Notices
- **Path:** `frontend/pages/dashboard.py`, `frontend/pages/students.py`, `frontend/pages/interview_workspace.py`
- **Evidence:** Streamlit runtime logs emit repeated warnings: `Please replace use_container_width with width. use_container_width will be removed after 2025-12-31. For use_container_width=True, use width='stretch'`.
- **Severity:** Medium (Code deprecation warning in modern Streamlit versions).
- **Status:** **PARTIALLY WORKING**.
- **Recommendation:** Refactor frontend components to replace `use_container_width=True` with `width="stretch"` and `use_container_width=False` with `width="content"`.

---

## 3. Low-Priority & Minor Polish Observations

### BUG-05: Missing Port Conflict Detection in Streamlit Client
- **Path:** `frontend/services/api_client.py:12`
- **Evidence:** Default API base URL is hardcoded to `http://127.0.0.1:8000`. If backend runs on an alternate port due to local port collisions (such as port 8080 or 8001), the frontend will fail to connect unless `MOCK_INTERVIEW_API_URL` environment variable is explicitly configured.
- **Severity:** Low.
- **Status:** **PASS (Environment variable supported, default port 8000 works)**.
- **Recommendation:** Add a connection tester in `frontend/pages/system_status.py` allowing administrators to modify or re-probe the backend host/port dynamically.

### BUG-06: Missing Headless Chrome Remote Debug Port
- **Path:** Desktop / System Environment
- **Evidence:** Live Chrome DevTools subagent was unable to connect to an existing Chrome browser session because Chrome was not started with `--remote-debugging-port=9222`.
- **Severity:** Low.
- **Status:** **BLOCKED (Live browser automation required remote debugging flag; full HTTP/unittest suites passed)**.
- **Recommendation:** Provide an automated Playwright or Selenium headless script configured with bundled Chromium for standalone headless UI testing.

