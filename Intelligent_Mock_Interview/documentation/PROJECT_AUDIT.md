# InterviewAI Product and Functional Audit

**Audit date:** 6 November 2026  
**Scope:** Current Streamlit/FastAPI/SQLAlchemy implementation and existing
SQLite records. This audit is read-only; it does not change application data.

## Executive summary

The project already has a persistent student/session/question/response model,
shared ordered question assignments, a transparent text heuristic, separate
audio measurements, a saved final-report route, Gemini request error mapping,
and mocked integration tests. The current problem is primarily workflow
fragmentation and several persistence/state gaps, not a missing application
foundation.

The current worktree has 19 sidebar destinations. The active interview is split
between separate Text and Voice pages, with a second report page and a separate
response-level Multimodal Report page. Student lookup, profile, registration,
assessment creation, active sessions, history, and reports also have separate
destinations. This makes the same persistent workflow look like several
unconnected products.

## Findings and root causes

| Priority | Finding | Source and root cause | Consequence |
|---|---|---|---|
| P0 | Original recordings are not retained. | `src/database.py` defines `InterviewResponse.audio_file_name` and `audio_retained`, but no audio bytes or durable path. `backend/main.py` writes temporary audio for analysis and deletes it in `_analyze_audio_bytes`; report data sets `audio_file_retained` to false. | A student cannot retrieve the original recording after navigation or restart, despite the requested shared-session workflow. |
| P0 | Voice draft data is not persistent until analysis completes. | `frontend/app.py::voice_interview` keeps bytes and transcripts in Streamlit session state. The API calls `persist_assessment` only after the full multimodal report is generated. | A restart or navigation before assessment loses the audio and transcript draft. |
| P1 | Navigation and actions are duplicated. | `frontend/app.py::NAV_GROUPS` exposes 19 pages. `main()` maps both `Text Interview` and `Interview Workspace` to the same renderer. It also exposes `Question Bank`, `Active Sessions`, `Interview History`, `Assessment Reports`, `Multimodal Report`, and `Independent Text Practice` separately. The dashboard repeats module links alongside Start/Register/Existing Student actions. | Users see parallel routes into the same workflow and must infer which pages are authoritative. |
| P1 | Text and voice are separate pages, not two views of one workspace. | `frontend/app.py::interview_workspace` and `voice_interview` are separate renderers and the sidebar links between them. Both retrieve the same `InterviewSession` when linked, but there are no Text/Voice tabs or unified progress controls. | Session continuity exists in storage, but the UI does not communicate it clearly. |
| P1 | Text drafts require an explicit save or evaluate action. | `frontend/app.py::interview_workspace` stores the response only when Save answer or Evaluate answer is pressed; changing questions or navigating away before either call leaves the draft only in widget state. | Answers can be lost during navigation or reruns before saving. |
| P1 | Missing reference answers are silently replaced by the question. | `backend/main.py::assess_interview_text` sets `ref = req.reference_answer.strip() if req.reference_answer else q`. The multimodal route similarly scores against `reference or question`. | A question may be presented as a reference answer; resulting similarity and Jaccard values look meaningful but do not compare against a reference. |
| P1 | Expected concepts do not contribute to the score. | `QuestionBank.expected_concepts` is exposed in `backend/management_api.py`; the UI displays concepts, but `src/text_pipeline.py::InterviewTextAssessor.assess_answer` only computes TF-IDF reference/question similarity, Jaccard reference overlap, and structure. | The rubric can imply concept-aware assessment although concept presence does not affect the score. |
| P1 | A changed voice question can retain the previous original transcript. | `src/voice_session.py::reset_for_question_change` clears recording, edited transcript, and report state but omits `voice_original_transcript`. Switching audio source in `frontend/app.py::voice_interview` also does not clear it. | The next report can associate an earlier raw transcript with the new question/recording. |
| P1 | Repeated Gemini requests are not deduplicated. | `backend/main.py::transcribe_audio` validates `response_id` but uses it for logging only; it does not cache or reject a completed/in-flight request. The frontend has no in-flight submission guard. | Double clicks or reruns may send the same recording to Gemini multiple times. Saved responses are upserted, but API work and possible usage are repeated. |
| P1 | Repeated final-report generation creates another version each time. | `backend/management_api.py::create_final_report` always uses `max(version)+1` after completion, with no request idempotency or “open latest” guard. | Accidental double submission creates duplicate-looking report versions. |
| P2 | Final report omits a skipped-question count. | `backend/management_api.py::_make_final_report` reports total and answered counts but does not return `skipped_questions`; `frontend/app.py::assessment_reports_page` cannot display it. | The requested completion summary is incomplete when a student skips questions. |
| P2 | Profile labels can collide for students with the same name/roll label. | `student_profile`, `student_directory`, and session selectors build dictionaries keyed by formatted display strings. Duplicate labels overwrite earlier IDs. | Some records can become inaccessible through the selector or resolve to the wrong ID. |
| P2 | Legacy/dead renderer remains in source. | `frontend/app.py::create_interview` is not the renderer selected in `main()`; `Text Interview` is an alias for `interview_workspace`. The older Multimodal Report duplicates the report shown on Voice Interview. | More code paths must be maintained and page names do not map one-to-one to behavior. |
| P2 | Browser/Gemini failures are not fully verified end to end. | `backend/main.py` maps quota, auth, timeout, network, and upstream errors; SDK/network work runs in a thread pool. `src/test_voice_interview_api.py` mocks the SDK. No browser device or real Gemini request has been exercised in this workspace. | The application has useful error handling, but physical microphone capture, provider availability, quota, and actual account behavior remain unverified. |
| P2 | Independent practice can coexist with a stale active-session pointer. | Selecting “Independent practice” in `voice_interview` does not clear `active_interview_session_id`, which is later used to render workflow progress/actions. The independent report request omits `session_id`, so it is not persisted to that session. | The UI can show an unrelated active assessment context beside an unlinked practice response. |

## Areas that passed inspection

- `InterviewSession` belongs to one student; each `SessionQuestion` belongs to
  one session and one question, with a stable unique position.
- `InterviewResponse` stores `student_id`, `session_id`,
  `session_question_id`, `question_id`, and `response_mode`. A unique constraint
  on question assignment plus mode turns saves into updates rather than
  duplicate response rows.
- Mixed-mode sessions use one saved question assignment list and preserve its
  order across API reloads. The existing five-question test exercises this.
- Text score components are finite values in `[0, 1]`. The documented formula
  is `0.50 × reference TF-IDF cosine + 0.20 × question TF-IDF cosine + 0.15 ×
  reference Jaccard + 0.15 × structure heuristic`. It is a practice heuristic,
  not human-calibrated grading. Empty answers under three words are rejected by
  the API; an empty vector comparison itself returns zero.
- The report generator reads persisted text assessments and voice measurements
  and does not compute a combined numeric text/audio score.
- The new transcription/report routes cap files at 25 MiB, check extension,
  signature, and MIME, clean temporary files, and avoid blocking the event loop
  for synchronous Gemini/Librosa work.
- Startup database changes are additive: `create_all` plus explicit additive
  missing-column migrations; no table or record deletion was observed.

## Read-only database integrity check

The existing `data/interviewai.sqlite3` was opened read-only. It currently has
1 student, 4 interview sessions, 61 questions, 20 session-question assignments,
2 responses, 2 text assessments, 0 audio assessments, and 0 saved final
reports. No student/session/question mismatch was found in responses, no
student mismatch was found in reports, and `PRAGMA foreign_key_check` returned
zero violations. This check did not reveal or export student-identifying data.

## Work plan

1. Simplify the sidebar to Dashboard, Students, New Assessment, Assessment
   History, and Model Analytics; move question management into assessment
   management and system status into a secondary settings area.
2. Make the active assessment a single workspace with Text and Voice tabs,
   backend-persisted question progress, and reliable navigation/resume actions.
3. Make saved response state explicit and isolate it by session/question/mode;
   clear all voice-derived state when its recording or question changes.
4. Preserve original audio in a durable, access-controlled local storage
   reference and retain separate original/corrected transcripts. Add a safe
   additive schema migration and prove existing rows remain intact.
5. Reject or explicitly mark missing reference answers instead of silently
   using the question as the benchmark; add known-input score tests and improve
   feedback transparency.
6. Make transcription/report submissions idempotent, show clear loading/error
   states, and prevent duplicate final-report versions on repeated requests.
7. Add skipped counts and ensure final reports expose separate available text
   and voice findings from saved records.
8. Run automated endpoint, database, score, audio, and Streamlit tests plus an
   isolated restart/persistence workflow. Real Gemini and physical microphone
   checks remain conditional on provider/browser access.
