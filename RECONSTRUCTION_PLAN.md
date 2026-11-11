# InterviewAI — System Reconstruction & Modernization Roadmap

**Goal:** Transform InterviewAI from an experimental prototype into a production-grade enterprise interview assessment platform with high resilience, clean architecture, automated CI/CD, and offline multimodal capabilities.

---

## Phase 1: Environment & Packaging Stabilization (Immediate)

1. **Standard Python Package Structure:**
   - Create a clean root `pyproject.toml` or `setup.py` defining `interviewai` as an installable package.
   - Eliminate all fragile relative `sys.path.insert(0, ...)` statements across backend, frontend, and test files.
   - Allow simple local execution: `pip install -e .` followed by `uvicorn backend.main:app` from any directory.
2. **Unified Configuration Management:**
   - Consolidate all configuration (API host, ports, SQLite path, Gemini key, audio thresholds) into a single Pydantic `BaseSettings` class (`backend/core/config.py`).
   - Support `.env` loading with environment precedence and automatic validation.
3. **Resolve Test Suite Discrepancy:**
   - Update `CandidateAudioAnalyzer` in `src/audio_feature_extractor.py` to remove legacy unvalidated fluency/prosody scores, aligning it 100% with `test_multimodal_complete.py`.

---

## Phase 2: Speech & Multimodal Resilience (Short-Term)

1. **Dual-Engine Speech Transcription (Online / Offline Fallback):**
   - **Primary Engine:** Google Gemini 3.5 API for high-accuracy cloud transcription when online and quota is active.
   - **Offline Fallback Engine:** Local Whisper ASR (via `faster-whisper` or `whisper.cpp`) running quantized models (e.g., `tiny.en` or `base.en`).
   - Automatically fall back to local transcription if Gemini returns HTTP 429, 401, 502, or network timeout.
2. **Audio Preprocessing & Noise Suppression:**
   - Integrate WebRTC Voice Activity Detection (VAD) or RNNoise for background hum suppression before Librosa acoustic extraction.
   - Provide visual audio waveform inspection in the frontend workspace.

---

## Phase 3: UI/UX & Design System Modernization (Mid-Term)

1. **Modern Dark Theme Consolidation:**
   - Standardize all pages on Deep Obsidian Navy (`#0B0F19`) background with Electric Cyan/Blue (`#06B6D4` / `#3B82F6`) accents.
   - Replace all deprecated Streamlit parameters (`use_container_width=True` → `width="stretch"`).
2. **Interactive Audio Visualizer:**
   - Implement real-time microphone recording widget with HTML5 AudioContext canvas waveform.
   - Instant audio playback and word-level transcript synchronization.
3. **Enhanced Candidate & Session Workflows:**
   - Candidate Resume PDF upload and automatic skill parsing.
   - Session timer with auto-submit on countdown expiry.
   - Comprehensive multi-candidate comparison views in Cohort Analytics.

---

## Phase 4: Enterprise Production Readiness (Long-Term)

1. **Database Scalability:**
   - Provide PostgreSQL database connection driver configuration alongside SQLite.
   - Integrate Alembic database migration scripts for seamless schema evolution.
2. **Containerization & CI/CD:**
   - Multi-stage `Dockerfile` creating an optimized runtime image with pre-downloaded ML weights and Librosa dependencies.
   - `docker-compose.yml` orchestrating FastAPI backend, Streamlit frontend, and optional local PostgreSQL/Redis services.
   - GitHub Actions workflow running linting (`ruff`), type-checking (`mypy`), and test suite (`pytest`) on every pull request.
3. **Security & Authentication:**
   - Introduce JWT-based authentication for interviewers and administrators.
   - Role-Based Access Control (RBAC): Candidate portal (interview taker) vs Administrator portal (evaluator/reports).
