# InterviewAI Database Relationship Audit

**Generated:** 2026-11-10  
**Database:** SQLite (`data/interviewai.sqlite3`)  
**ORM:** SQLAlchemy 2.x  
**Migrations:** Alembic

## Entity Relationship Diagram (Text)

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│     User     │      │   Student    │      │ UserSession  │
├──────────────┤      ├──────────────┤      ├──────────────┤
│ id (PK)      │      │ id (PK)      │      │ id (PK)      │
│ email (UQ)   │◄─────┤ user_id (FK) │      │ token (UQ)   │
│ password_hash│      │ full_name    │      │ user_id (FK) │
│ role         │      │ roll_number  │      │ expires_at   │
│ account_status│     │ registration │      │ created_at   │
│ created_at   │      │   _number    │      └──────────────┘
│ updated_at   │      │ course       │
└──────────────┘      │ department   │
                      │ semester_year│
                      │ skills_json  │
                      │ email (UQ)   │
                      │ target_job_role│
                      │ preferred_domain│
                      │ created_at   │
                      │ updated_at   │
                      └──────┬───────┘
                             │
                      ┌──────▼──────┐      ┌──────────────┐
                      │InterviewSession│   │InterviewResponse│
                      ├──────────────┤      ├──────────────┤
                      │ id (PK)      │      │ id (PK)      │
                      │ student_id (FK)│   │ session_id (FK)│
                      │ title        │      │ session_question_id (FK)│
                      │ domain       │      │ question_id (FK)│
                      │ difficulty   │      │ response_mode  │
                      │ mode         │      │ answer_text    │
                      │ question_count│     │ original_transcript│
                      │ current_position│ │ corrected_transcript│
                      │ status       │      │ audio_file_name│
                      │ text_status  │      │ audio_retained │
                      │ voice_status │      └──────┬───────┘
                      │ created_at   │           │
                      │ updated_at   │           │
                      │ completed_at │      ┌────▼────────┐
                      └──────┬───────┘      │ TextAssessment │
                             │              ├──────────────┤
                      ┌──────▼──────┐      │ id (PK)      │
                      │ SessionQuestion│   │ response_id (FK)│
                      ├──────────────┤      │ relevance_score│
                      │ id (PK)      │      │ question_alignment_score│
                      │ session_id (FK)│   │ keyword_overlap│
                      │ question_id (FK)│   │ structure_score│
                      │ position     │      │ quality_score  │
                      └──────┬───────┘      │ feedback_json  │
                             │              └──────────────┘
                      ┌──────▼──────┐
                      │ QuestionBank│
                      ├──────────────┤
                      │ id (PK)      │
                      │ question_code│
                      │ source_id    │
                      │ domain       │
                      │ topic        │
                      │ difficulty   │
                      │ question_text│
                      │ reference_answer│
                      │ expected_concepts_json│
                      │ provenance   │
                      │ created_at   │
                      └──────────────┘

                      ┌──────────────┐
                      │AudioAssessment│
                      ├──────────────┤
                      │ id (PK)      │
                      │ response_id (FK)│
                      │ measurements_json│
                      └──────────────┘

                      ┌──────────────┐
                      │AssessmentReport│
                      ├──────────────┤
                      │ id (PK)      │
                      │ session_id (FK)│
                      │ student_id (FK)│
                      │ version      │
                      │ report_json  │
                      └──────────────┘
```

## Tables and Columns

### users
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| email | VARCHAR(254) | NO | — | Unique, indexed |
| password_hash | VARCHAR(255) | NO | — | bcrypt hash |
| role | VARCHAR(20) | NO | 'student' | 'student' or 'admin' |
| account_status | VARCHAR(20) | NO | 'active' | 'active', 'suspended', etc. |
| created_at | DATETIME | NO | utcnow | |
| updated_at | DATETIME | NO | utcnow | Auto-updated |

Indexes: `ix_users_email` (unique), `ix_users_role`

### user_sessions
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(36) | NO | uuid4 | Primary key |
| token | VARCHAR(255) | NO | — | Unique, indexed |
| user_id | VARCHAR(24) | NO | — | FK → users.id (CASCADE) |
| expires_at | DATETIME | NO | — | Session expiration |
| created_at | DATETIME | NO | utcnow | |

Indexes: `ix_user_sessions_user_id`, `ix_user_sessions_expires_at`

### students
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| user_id | VARCHAR(24) | YES | — | FK → users.id (SET NULL) |
| full_name | VARCHAR(160) | NO | — | |
| roll_number | VARCHAR(80) | YES | — | Unique |
| registration_number | VARCHAR(80) | YES | — | Unique |
| course | VARCHAR(120) | NO | '' | |
| department | VARCHAR(120) | NO | '' | |
| semester_year | VARCHAR(80) | NO | '' | |
| skills_json | TEXT | NO | '[]' | JSON array |
| email | VARCHAR(254) | YES | — | Unique |
| target_job_role | VARCHAR(160) | NO | '' | |
| preferred_domain | VARCHAR(100) | NO | '' | |
| created_at | DATETIME | NO | utcnow | |
| updated_at | DATETIME | NO | utcnow | Auto-updated |

Indexes: `ix_students_full_name`, `ix_students_user_id`
Unique: `uq_students_roll_number`, `uq_students_registration_number`, `uq_students_email`

### interview_sessions
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| student_id | VARCHAR(24) | NO | — | FK → students.id |
| title | VARCHAR(240) | NO | — | |
| domain | VARCHAR(100) | NO | — | |
| difficulty | VARCHAR(20) | NO | 'Intermediate' | |
| mode | VARCHAR(10) | NO | 'text' | 'text', 'voice', 'mixed' |
| question_count | INTEGER | NO | 5 | |
| time_limit_minutes | INTEGER | YES | — | |
| current_position | INTEGER | NO | 1 | |
| status | VARCHAR(24) | NO | 'in_progress' | 'in_progress', 'ready_for_report', 'completed' |
| text_status | VARCHAR(24) | NO | 'pending' | 'pending', 'in_progress', 'completed', 'not_required' |
| voice_status | VARCHAR(24) | NO | 'pending' | 'pending', 'in_progress', 'completed', 'not_required' |
| created_at | DATETIME | NO | utcnow | |
| updated_at | DATETIME | NO | utcnow | Auto-updated |
| completed_at | DATETIME | YES | — | |

Indexes: `ix_sessions_student_status`, `ix_sessions_created_at`

### session_questions
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| session_id | VARCHAR(24) | NO | — | FK → interview_sessions.id |
| question_id | INTEGER | NO | — | FK → question_bank.id |
| position | INTEGER | NO | — | 1-based order |

### interview_responses
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| session_id | VARCHAR(24) | NO | — | FK → interview_sessions.id |
| session_question_id | VARCHAR(24) | NO | — | FK → session_questions.id |
| question_id | INTEGER | NO | — | FK → question_bank.id |
| student_id | VARCHAR(24) | NO | — | FK → students.id |
| response_mode | VARCHAR(10) | NO | — | 'text' or 'voice' |
| answer_text | TEXT | YES | — | |
| original_transcript | TEXT | YES | — | Raw Gemini transcript |
| corrected_transcript | TEXT | YES | — | User-edited transcript |
| audio_file_name | VARCHAR(255) | YES | — | |
| audio_retained | BOOLEAN | NO | False | |

### text_assessments
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| response_id | VARCHAR(24) | NO | — | FK → interview_responses.id |
| relevance_score | FLOAT | YES | — | 0-1 |
| question_alignment_score | FLOAT | YES | — | 0-1 |
| keyword_overlap | FLOAT | YES | — | 0-1 |
| structure_score | FLOAT | YES | — | 0-1 |
| quality_score | FLOAT | YES | — | 0-1 |
| feedback_json | TEXT | YES | — | JSON strengths/improvements |

### audio_assessments
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| response_id | VARCHAR(24) | NO | — | FK → interview_responses.id |
| measurements_json | TEXT | YES | — | JSON Librosa metrics |

### assessment_reports
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | VARCHAR(24) | NO | uuid hex | Primary key |
| session_id | VARCHAR(24) | NO | — | FK → interview_sessions.id |
| student_id | VARCHAR(24) | NO | — | FK → students.id |
| version | INTEGER | NO | 1 | Auto-incremented per session |
| report_json | TEXT | NO | — | JSON report |

Indexes: `uq_assessment_report_session_version` (unique on session_id + version)

### question_bank
| Column | Type | Nullable | Default | Notes |
|---|---|---|---|---|
| id | INTEGER | NO | autoincrement | Primary key |
| question_code | VARCHAR(24) | NO | — | Unique |
| source_id | VARCHAR(180) | YES | — | Unique |
| domain | VARCHAR(100) | NO | — | |
| topic | VARCHAR(120) | NO | 'General' | |
| difficulty | VARCHAR(20) | NO | 'Intermediate' | |
| question_text | TEXT | NO | — | |
| reference_answer | TEXT | YES | — | |
| expected_concepts_json | TEXT | NO | '[]' | JSON array |
| provenance | VARCHAR(240) | NO | — | |
| created_at | DATETIME | NO | utcnow | |

Indexes: `ix_questions_domain_difficulty`, `ix_questions_topic`
Unique: `uq_questions_domain_text`, `uq_questions_source_id`

## Foreign Key Constraints

| Child Table | Column | Parent Table | Parent Column | On Delete |
|---|---|---|---|---|
| students | user_id | users | id | SET NULL |
| user_sessions | user_id | users | id | CASCADE |
| interview_sessions | student_id | students | id | (default) |
| session_questions | session_id | interview_sessions | id | (default) |
| session_questions | question_id | question_bank | id | (default) |
| interview_responses | session_id | interview_sessions | id | (default) |
| interview_responses | session_question_id | session_questions | id | (default) |
| interview_responses | question_id | question_bank | id | (default) |
| interview_responses | student_id | students | id | (default) |
| text_assessments | response_id | interview_responses | id | (default) |
| audio_assessments | response_id | interview_responses | id | (default) |
| assessment_reports | session_id | interview_sessions | id | (default) |
| assessment_reports | student_id | students | id | (default) |

## Data Isolation Rules

- Each `User` has at most one `Student` profile (1:1 via `user_id`)
- Each `Student` can have multiple `InterviewSession` records (1:N)
- Each `InterviewSession` has multiple `SessionQuestion` records (1:N)
- Each `SessionQuestion` can have multiple `InterviewResponse` records (1:N, one per mode)
- Each `InterviewResponse` can have one `TextAssessment` and one `AudioAssessment` (1:1)
- Each `InterviewSession` can have multiple `AssessmentReport` records (1:N, versioned)
- All queries filter by `student_id` or `user_id` to enforce data isolation

## Migration Notes

- Schema changes are additive only (no table/column deletion)
- Alembic migration `805f05af28ba` adds `users`, `user_sessions` tables and `user_id` column on `students`
- Existing data is preserved during migration
- `initialize_database()` handles additive column migration via `ALTER TABLE`
- Database backup: `data/interviewai.sqlite3.backup_20261110`
