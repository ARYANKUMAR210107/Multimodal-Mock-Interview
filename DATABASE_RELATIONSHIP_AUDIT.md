# InterviewAI — Database Relationship & Schema Audit

**Database Engine:** SQLite 3  
**ORM:** SQLAlchemy 2.x Declarative  
**Connection Settings:** Foreign Keys ON (`PRAGMA foreign_keys=ON`), WAL Mode (`PRAGMA journal_mode=WAL`), 30s timeout  
**Production DB File:** `data/interviewai.sqlite3`  

---

## 1. Relational Entity-Relationship Diagram

```mermaid
erDiagram
    STUDENT ||--o{ INTERVIEW_SESSION : "has many"
    STUDENT ||--o{ INTERVIEW_RESPONSE : "author of"
    STUDENT ||--o{ ASSESSMENT_REPORT : "subject of"
    QUESTION_BANK ||--o{ SESSION_QUESTION : "referenced by"
    QUESTION_BANK ||--o{ INTERVIEW_RESPONSE : "answered for"
    INTERVIEW_SESSION ||--|{ SESSION_QUESTION : "contains (1..N)"
    INTERVIEW_SESSION ||--o{ INTERVIEW_RESPONSE : "tracks"
    INTERVIEW_SESSION ||--o{ ASSESSMENT_REPORT : "generates (1..N versions)"
    SESSION_QUESTION ||--o{ INTERVIEW_RESPONSE : "has responses (text/voice)"
    INTERVIEW_RESPONSE ||--o| TEXT_ASSESSMENT : "assessed by (1:1)"
    INTERVIEW_RESPONSE ||--o| AUDIO_ASSESSMENT : "measured by (1:1)"

    STUDENT {
        string id PK "24-char hex"
        string full_name "Student name"
        string roll_number UK "Unique roll number"
        string registration_number UK "Unique reg number"
        string course
        string department
        string semester_year
        text skills_json "JSON array of skills"
        string email
        string target_job_role
        string preferred_domain
        datetime created_at
        datetime updated_at
    }

    QUESTION_BANK {
        int id PK "Autoincrement"
        string question_code UK "e.g. Q-CUSTOM-..."
        string source_id UK "Catalog provenance"
        string domain "Indexed"
        string topic "Indexed"
        string difficulty "Beginner|Intermediate|Advanced"
        text question_text "Indexed with domain"
        text reference_answer
        text expected_concepts_json "JSON array"
        string provenance
        datetime created_at
    }

    INTERVIEW_SESSION {
        string id PK "32-char UUID hex"
        string student_id FK "students.id (RESTRICT)"
        string title
        string domain
        string difficulty
        string mode "text|voice|mixed"
        int question_count
        int time_limit_minutes
        int current_position
        string status "in_progress|ready_for_report|completed"
        string text_status "pending|in_progress|completed|not_required"
        string voice_status "pending|in_progress|completed|not_required"
        datetime created_at
        datetime updated_at
        datetime completed_at
    }

    SESSION_QUESTION {
        int id PK "Autoincrement"
        string session_id FK "interview_sessions.id (CASCADE)"
        int question_id FK "question_bank.id (RESTRICT)"
        int position "Question order 1..N"
        string status "pending|answered|skipped|assessed"
        bool marked_for_review
        datetime updated_at
    }

    INTERVIEW_RESPONSE {
        string id PK "36-char UUID"
        string student_id FK "students.id (RESTRICT)"
        string session_id FK "interview_sessions.id (CASCADE)"
        int session_question_id FK "session_questions.id (CASCADE)"
        int question_id FK "question_bank.id (RESTRICT)"
        string response_mode "text|voice"
        text answer_text
        text original_transcript
        text corrected_transcript
        string audio_file_name
        bool audio_retained
        datetime created_at
        datetime updated_at
    }

    TEXT_ASSESSMENT {
        int id PK "Autoincrement"
        string response_id FK "interview_responses.id (CASCADE, UNIQUE)"
        float relevance_score
        float question_alignment_score
        float keyword_overlap
        float structure_score
        float quality_score
        text feedback_json "Strengths & improvements"
        datetime created_at
    }

    AUDIO_ASSESSMENT {
        int id PK "Autoincrement"
        string response_id FK "interview_responses.id (CASCADE, UNIQUE)"
        text measurements_json "Pitch, pause, MFCCs"
        datetime created_at
    }

    ASSESSMENT_REPORT {
        int id PK "Autoincrement"
        string session_id FK "interview_sessions.id (CASCADE)"
        string student_id FK "students.id (RESTRICT)"
        int version "Sequential version (1, 2, ...)"
        text report_json "Complete snapshot"
        datetime created_at
    }
```

---

## 2. Table Inspection & Schema Validation

### 2.1 Constraint Verification
1. **Foreign Key Integrity:**
   - Deleting an `InterviewSession` automatically cascades to `SessionQuestion`, `InterviewResponse`, `TextAssessment`, `AudioAssessment`, and `AssessmentReport`.
   - Deleting a `Student` is **restricted** (`ondelete="RESTRICT"`) if they have active sessions or responses.
   - Deleting a `QuestionBank` question is **restricted** if it is referenced in an active session question or response.
2. **Uniqueness Constraints:**
   - `students.roll_number` & `students.registration_number` are strictly enforced unique.
   - `question_bank(domain, question_text)` ensures zero duplicate questions within a domain.
   - `session_questions(session_id, position)` prevents order collision.
   - `interview_responses(session_question_id, response_mode)` enforces exactly one text and/or one voice response per question.
   - `assessment_reports(session_id, version)` enforces strict report version uniqueness.

### 2.2 Shared Session Workflow (Text & Voice)
- **Finding:** The architecture correctly allows a single session to handle both text and voice modes:
  - Each `SessionQuestion` can link up to two `InterviewResponse` records: one for `text` and one for `voice`.
  - The session tracks `text_status` and `voice_status` independently (`pending` → `in_progress` → `completed`).
  - Only when all required modes are completed can a final report be compiled.

### 2.3 Live Production DB State
- `students`: 1 record (`Aryan kumar`).
- `question_bank`: 61 curated questions across 12 domains.
- `interview_sessions`: 4 sessions.
- `session_questions`: 20 assigned questions.
- `interview_responses`: 2 recorded answers.
- `text_assessments`: 2 scored responses.
- `audio_assessments`: 0 recorded acoustic sessions.
- `assessment_reports`: 0 finalized reports.

