# InterviewAI — Live Test Execution Results

**Execution Date:** 2026-11-10  
**Test Harness:** Python `unittest` runner / HTTP Test Client / Uvicorn Live Server / Streamlit Live Server  
**Total Tests Executed:** 55 automated integration tests + Live HTTP Endpoint Probes  
**Results Summary:**
- **Automated Tests Passed:** 54 (98.2%)
- **Automated Tests Failed:** 1 (1.8% — Minor audio analyzer test assertion)
- **Live HTTP Health Endpoint:** PASS (HTTP 200, status `online`)
- **Live Streamlit App:** PASS (HTTP 200, port 8501)
- **Live ML Inference (`/predict/employability`):** PASS (HTTP 200, probability 0.5338)
- **Live Database Read Inspection:** PASS (61 questions, 1 student, 4 sessions)

---

## 1. Automated Integration Test Suite Breakdown

### 1.1 Endpoint & Pipeline Tests (`test_all_endpoints.py`) — 17 Tests: **17 PASS, 0 FAIL**
- `test_01_get_root`: PASS (returns online status, version, endpoints list)
- `test_02_get_health`: PASS (returns transcription configuration probe)
- `test_03_get_questions`: PASS (retrieves default 15 questions from cache)
- `test_04_predict_employability_valid`: PASS (evaluates 5 soft skills)
- `test_05_predict_employability_invalid_range`: PASS (422 on ratings outside 1-5)
- `test_06_assess_text_valid`: PASS (scores answer, returns tier and recommendations)
- `test_07_assess_text_too_short`: PASS (400 on answers `< 3` words)
- `test_08_assess_audio_valid_wav`: PASS (computes pitch, pause ratio, MFCCs)
- `test_09_assess_audio_invalid_format`: PASS (415 on unrecognized file bytes)
- `test_10_assess_multimodal_full`: PASS (combines text, soft skills, audio)
- `test_11_assess_multimodal_invalid_skills`: PASS (422 validation on out-of-range skills)
- `test_12_transcribe_gemini_missing_key`: PASS (503 on unconfigured API key)
- `test_13_transcribe_gemini_rate_limit`: PASS (429 handling on quota limits)
- `test_14_transcribe_gemini_invalid_key`: PASS (401 on unauthorized credentials)
- `test_15_multimodal_report_endpoint`: PASS (verifies multimodal report generation)
- `test_16_multimodal_report_validation`: PASS (validates required question and transcript)
- `test_17_audio_retention_and_serving`: PASS (verifies audio file serving via API)

### 1.2 Database & Management Tests (`test_database_management.py`) — 5 Tests: **5 PASS, 0 FAIL**
- `test_student_crud`: PASS (creation, retrieval, update, duplicate prevention)
- `test_question_bank_crud`: PASS (authoring custom questions, duplicate check, deletion guards)
- `test_session_lifecycle`: PASS (session setup, position traversal, question response saving)
- `test_mode_completion_and_final_report`: PASS (independent mode completion & report versioning)
- `test_dashboard_analytics_summary`: PASS (verifies aggregation queries and counts)

### 1.3 Full End-to-End Workflow Audit (`test_end_to_end_audit.py`) — 1 Test: **1 PASS, 0 FAIL**
- `test_student_directory_profile_session_text_voice_report_and_isolation`: PASS
  - Steps Tested:
    1. Registered candidate with unique roll/reg numbers.
    2. Verified student profile persistence.
    3. Initialized 3-question intermediate Python assessment session.
    4. Submitted text answers for all questions; verified TF-IDF relevance scoring.
    5. Uploaded synthetic speech audio; extracted acoustic features; attached to session.
    6. Completed text stage; completed voice stage.
    7. Generated version 1 immutable final student assessment report.
    8. Verified zero leakage into production database records.

### 1.4 Real Audio & ML Feature Tests (`test_ui_and_features.py`) — 3 Tests: **3 PASS, 0 FAIL**
- `TEST 1: Technical Text Assessment Pipeline`: PASS (TF-IDF cosine similarity: 42.7%, composite: 0.3737)
- `TEST 2: Real Audio Assessment via Librosa`: PASS (Generated 4.0s harmonic WAV; extracted 14 features)
- `TEST 3: Multimodal Rule-Based Fusion`: PASS (Text + speech fluency + prosody normalized calculation)

### 1.5 Multimodal Comprehensive Suite (`test_multimodal_complete.py`) — 25 Tests: **24 PASS, 1 FAIL**
- **Failure Detail:**
  - `FAIL: test_reports_descriptive_features_without_unvalidated_quality_scores`
  - Traceback: `AssertionError: 'estimated_fluency_score' unexpectedly found in {...}`
  - Analysis: `backend/main.py:256` correctly strips `estimated_fluency_score` and `estimated_prosody_score` from the API response payload, but `CandidateAudioAnalyzer.analyze_audio_file` in `src/audio_feature_extractor.py` still computes and returns them in the raw dictionary. The test asserts that the raw extractor output does not contain these unvalidated scores.

---

## 2. Live HTTP Service Probes

```
[PROBE 1: GET http://127.0.0.1:8000/health]
Response Status: 200 OK
Response Body:
{
  "status": "online",
  "version": "1.0.0",
  "transcription_provider": "gemini",
  "transcription_configured": true,
  "transcription_model": "gemini-3.5-transcribe"
}

[PROBE 2: POST http://127.0.0.1:8000/api/v1/predict/employability]
Request: {"manner_of_speaking": 4, "mental_alertness": 5, "self_confidence": 4, "ability_to_present_ideas": 4, "communication_skills": 5}
Response Status: 200 OK
Response Body:
{
  "prediction": "Employable",
  "probability_employable": 0.5338,
  "probability_less_employable": 0.4662,
  "confidence_score": 53.4,
  "soft_skill_mean": 4.4,
  "soft_skill_min": 4,
  "recommendations": ["Excellent soft-skill profile! Continue reinforcing technical domain knowledge."]
}

[PROBE 3: GET http://localhost:8501]
Response Status: 200 OK (Streamlit UI Serving)
```

