# InterviewAI — API Endpoint Inventory

**Total Endpoints Identified:** 21  
**Protocol:** REST over HTTP / JSON / Multipart Form-Data  
**Base URL:** `http://127.0.0.1:8000`  
**Authentication:** Public / Session ID Based (No Bearer Token / Role-Based Access Enforced)  

---

## 1. System & Health Endpoints

### 1.1 `GET /`
- **Description:** System greeting, status, and supported endpoint index.
- **Request Schema:** None.
- **Response Schema:**
  ```json
  {
    "status": "online",
    "system": "Intelligent Multimodal Mock Interview Assessment System",
    "version": "1.0.0",
    "transcription_provider": "gemini",
    "transcription_configured": true,
    "transcription_status": "configured",
    "transcription_model": "gemini-3.5-transcribe",
    "endpoints": ["..."]
  }
  ```
- **DB Interaction:** None.
- **Test Result:** **PASS** (HTTP 200).

### 1.2 `GET /health`
- **Description:** Real-time health status and Gemini configuration probe.
- **Request Schema:** None.
- **Response Schema:**
  ```json
  {
    "status": "online",
    "version": "1.0.0",
    "transcription_provider": "gemini",
    "transcription_configured": true,
    "transcription_model": "gemini-3.5-transcribe"
  }
  ```
- **DB Interaction:** None.
- **Test Result:** **PASS** (HTTP 200).

---

## 2. AI / ML Inference Endpoints

### 2.1 `POST /api/v1/predict/employability`
- **Description:** Soft-skill student employability classification using Gradient Boosting.
- **Request Schema:**
  ```json
  {
    "manner_of_speaking": 4,
    "mental_alertness": 5,
    "self_confidence": 4,
    "ability_to_present_ideas": 4,
    "communication_skills": 5
  }
  ```
  *(All integer fields between 1 and 5)*
- **Response Schema:**
  ```json
  {
    "prediction": "Employable",
    "probability_employable": 0.5338,
    "probability_less_employable": 0.4662,
    "confidence_score": 53.4,
    "soft_skill_mean": 4.4,
    "soft_skill_min": 4,
    "recommendations": ["..."]
  }
  ```
- **DB Interaction:** None.
- **Error Handling:** 422 Unprocessable Entity on ratings `< 1` or `> 5`.
- **Test Result:** **PASS** (HTTP 200, verified live on port 8000).

### 2.2 `POST /api/v1/assess/text`
- **Description:** NLP technical answer evaluation against reference answer and question.
- **Request Schema:**
  ```json
  {
    "question": "What is polymorphism?",
    "candidate_answer": "Polymorphism allows objects to take on multiple forms...",
    "reference_answer": "Polymorphism in OOP is...",
    "session_id": "optional-uuid",
    "position": 1
  }
  ```
- **Response Schema:**
  ```json
  {
    "relevance_score": 0.42,
    "question_alignment_score": 0.31,
    "keyword_overlap": 0.28,
    "structure_score": 0.50,
    "quality_score": 0.37,
    "performance_tier": "Proficient / Strong Answer",
    "feedback": ["..."]
  }
  ```
- **DB Interaction:** If `session_id` provided, automatically updates `InterviewResponse` and `TextAssessment` in SQLite.
- **Error Handling:** 400 on empty question or `< 3` word answer.
- **Test Result:** **PASS** (HTTP 200).

### 2.3 `POST /api/v1/assess/audio`
- **Description:** Librosa acoustic feature extraction on raw audio file.
- **Request Schema:** Multipart form-data with `file` (WAV, MP3, M4A, WebM <= 25MB).
- **Response Schema:**
  ```json
  {
    "audio_metrics": {
      "duration_sec": 4.0,
      "pause_ratio": 0.12,
      "mean_pitch_hz": 160.15,
      "pitch_std_hz": 0.34,
      "mean_energy_rms": 0.24,
      "mfcc_mean": [...]
    },
    "speech_feedback": ["..."]
  }
  ```
- **DB Interaction:** None.
- **Error Handling:** 415 on invalid container/MIME type; 413 on `> 25MB`.
- **Test Result:** **PASS** (HTTP 200).

### 2.4 `POST /api/v1/assess/multimodal`
- **Description:** Combined text assessment, soft skill inference, and optional audio analysis.
- **Request Schema:** Multipart form-data (`question`, `candidate_answer`, soft skills 1-5, optional `audio_file`).
- **Response Schema:**
  ```json
  {
    "overall_status": "Evaluation Complete",
    "employability_result": {"prediction": "Employable", "employability_probability": 0.53},
    "text_evaluation": {...},
    "speech_acoustic_evaluation": {...}
  }
  ```
- **DB Interaction:** None.
- **Test Result:** **PASS** (HTTP 200).

### 2.5 `POST /api/audio/transcribe`
- **Description:** Transcribes candidate audio using Google Gemini 3.5 API.
- **Request Schema:** Multipart form-data (`file`, optional `response_id`).
- **Response Schema:**
  ```json
  {
    "transcript": "...",
    "model": "gemini-3.5-transcribe",
    "provider": "gemini",
    "response_id": "RESP-..."
  }
  ```
- **DB Interaction:** None.
- **Error Handling:** 503 if no API key; 401 if invalid key; 429 if quota exceeded; 415/422 if audio unparseable.
- **Test Result:** **NOT VERIFIED (Requires External Gemini Quota)** / Mocked tests **PASS**.

### 2.6 `POST /api/interview/multimodal-report`
- **Description:** Generates comprehensive multimodal report for an individual question response.
- **Request Schema:** Multipart form-data (`question`, `corrected_transcript`, `file`, `reference_answer`, `session_id`, `position`).
- **Response Schema:** Typed `MultimodalReportResponse` containing independent text and audio scores.
- **DB Interaction:** Persists audio recording to `data/audio_recordings/` and saves assessment into database.
- **Test Result:** **PASS** (HTTP 200 in end-to-end audit).

---

## 3. Persistent Database & Management APIs

### 3.1 Students
- `GET /api/students` — Lists students with optional `?search=` filter. (**PASS**)
- `POST /api/students` — Creates student record with duplicate checks. (**PASS**)
- `GET /api/students/{student_id}` — Retrieves student profile and all associated sessions. (**PASS**)
- `PUT /api/students/{student_id}` — Updates student record. (**PASS**)
- `GET /api/export/students.csv` — Exports student directory as CSV attachment. (**PASS**)

### 3.2 Question Bank
- `GET /api/questions` & `GET /api/v1/questions` — Lists questions by domain/difficulty. (**PASS**)
- `POST /api/questions` — Creates a custom question in bank. (**PASS**)
- `PUT /api/questions/{question_id}` — Updates question content/reference answer. (**PASS**)
- `DELETE /api/questions/{question_id}` — Deletes unassigned question (protects assigned ones with 409). (**PASS**)

### 3.3 Sessions & Assessment Lifecycle
- `POST /api/sessions` — Initializes interview session for a student with random question selection. (**PASS**)
- `GET /api/sessions` — Lists interview sessions with student/status filtering. (**PASS**)
- `GET /api/sessions/{session_id}` — Retrieves detailed session state, questions, and responses. (**PASS**)
- `PUT /api/sessions/{session_id}/position` — Updates active question cursor. (**PASS**)
- `PUT /api/sessions/{session_id}/questions/{position}/response` — Saves draft text/voice answer. (**PASS**)
- `PUT /api/sessions/{session_id}/questions/{position}/status` — Sets question status (`pending`, `answered`, `skipped`, `assessed`). (**PASS**)
- `POST /api/sessions/{session_id}/modes/{mode}/complete` — Marks `text` or `voice` stage as completed. (**PASS**)
- `POST /api/sessions/{session_id}/complete` — Finalizes overall session. (**PASS**)

### 3.4 Reports & Analytics
- `POST /api/sessions/{session_id}/reports/final` — Generates immutable versioned student report. (**PASS**)
- `GET /api/sessions/{session_id}/reports` — Lists reports for a session. (**PASS**)
- `GET /api/reports/{report_id}` — Fetches single assessment report. (**PASS**)
- `GET /api/dashboard/summary` & `GET /api/analytics` — Aggregated KPI metrics for dashboard. (**PASS**)
- `GET /api/audio/recordings/{filename}` — Serves candidate audio recording file. (**PASS**)

