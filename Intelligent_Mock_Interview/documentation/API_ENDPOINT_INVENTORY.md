# InterviewAI API Endpoint Inventory

**Generated:** 2026-11-10  
**Base URL:** `http://127.0.0.1:8000`  
**Auth:** Bearer token (session-based, 24-hour TTL)  
**Total Endpoints:** ~35

## Authentication Endpoints

| Method | Path | Description | Auth Required | Status Codes |
|---|---|---|---|---|
| POST | `/api/auth/register` | Register user account + student profile | No | 201, 409 |
| POST | `/api/auth/login` | Authenticate and return session token | No | 200, 401 |
| POST | `/api/auth/logout` | Invalidate session token | Yes | 200, 401 |
| GET | `/api/auth/me` | Get current user profile | Yes | 200, 401 |
| POST | `/api/auth/change-password` | Change authenticated user's password | Yes | 200, 401 |

## Student Management Endpoints

| Method | Path | Description | Auth Required | Status Codes |
|---|---|---|---|---|
| GET | `/api/students` | List students (search by `search` param) | No | 200 |
| POST | `/api/students` | Create student profile (legacy, no auth) | No | 201, 409 |
| GET | `/api/students/{id}` | Get student details with sessions & reports | No | 200, 404 |

## Session Management Endpoints

| Method | Path | Description | Auth Required | Status Codes |
|---|---|---|---|---|
| POST | `/api/sessions` | Create interview session | No | 201, 403, 404 |
| GET | `/api/sessions` | List sessions (filters: `student_id`, `user_id`, `status`, `limit`) | **Yes** — returns only the authenticated user's sessions | 200 |
| GET | `/api/sessions/{id}` | Get session details with questions & responses | No (optional; 403 if authenticated but not owner) | 200, 403, 404 |
| PUT | `/api/sessions/{id}/position` | Set current question position | No | 200, 422 |
| PUT | `/api/sessions/{id}/questions/{position}/response` | Save answer draft (text or voice) | No | 200, 404 |
| POST | `/api/sessions/{id}/modes/text/complete` | Mark text mode as completed | No | 200, 409 |
| POST | `/api/sessions/{id}/modes/voice/complete` | Mark voice mode as completed | No | 200, 409 |
| POST | `/api/sessions/{id}/final-report` | Generate final assessment report | No | 200, 409 |
| GET | `/api/sessions/{id}/reports` | List session reports | **Yes** — returns only the authenticated user's session reports | 200 |

## Assessment Endpoints

| Method | Path | Description | Auth Required | Status Codes |
|---|---|---|---|---|
| POST | `/api/v1/assess/text` | Evaluate text response against reference answer | No | 200, 400 |
| POST | `/api/interview/multimodal-report` | Voice assessment with Librosa audio analysis | No | 200, 400 |
| POST | `/api/audio/transcribe` | Gemini speech transcription (audio file upload) | No | 200, 401, 409, 413, 415, 422, 429 |
| POST | `/api/v1/predict/employability` | Employability prediction from soft-skill ratings | No | 200 |

## Question Bank Endpoints

| Method | Path | Description | Auth Required | Status Codes |
|---|---|---|---|---|
| GET | `/api/v1/questions` | List interview questions (filters: `domain`, `difficulty`, `limit`) | No | 200 |

## System Endpoints

| Method | Path | Description | Auth Required | Status Codes |
|---|---|---|---|---|
| GET | `/` | Health check | No | 200 |
| GET | `/api/health` | Health check (alias) | No | 200 |
| GET | `/api/analytics` | Dashboard analytics & statistics | No | 200 |
| GET | `/api/dashboard/summary` | Dashboard summary (alias for analytics) | No | 200 |
| GET | `/api/sessions/export/csv` | Export students as CSV | No | 200 |

## Request/Response Schemas

### Auth Register Request
```json
{
  "email": "string (required)",
  "password": "string (required, min 8)",
  "full_name": "string (required)",
  "roll_number": "string (optional)",
  "registration_number": "string (optional)",
  "course": "string (optional)",
  "department": "string (optional)",
  "semester_year": "string (optional)",
  "skills": ["string"],
  "target_job_role": "string (optional)",
  "preferred_domain": "string (optional)"
}
```

### Auth Login Request
```json
{
  "email": "string (required)",
  "password": "string (required)"
}
```

### Session Create Request
```json
{
  "student_id": "string (required)",
  "user_id": "string (optional)",
  "domain": "string (required)",
  "difficulty": "Any|Beginner|Intermediate|Advanced (default: Any)",
  "mode": "text|voice|mixed (default: text)",
  "question_count": "integer 1-50 (default: 5)",
  "time_limit_minutes": "integer 1-300 (optional)",
  "title": "string (optional)"
}
```

### Save Response Request
```json
{
  "answer_text": "string (optional)",
  "original_transcript": "string (optional)",
  "corrected_transcript": "string (optional)",
  "response_mode": "text|voice (default: text)"
}
```

### Text Assessment Request
```json
{
  "question": "string (required)",
  "candidate_answer": "string (required)",
  "reference_answer": "string (optional)",
  "session_id": "string (optional)",
  "position": "integer (optional)"
}
```

### Multimodal Report Request (multipart/form-data)
| Field | Type | Description |
|---|---|---|
| question | string | Question text |
| reference_answer | string | Reference answer |
| corrected_transcript | string | User-edited transcript (min 3 words) |
| original_transcript | string | Raw Gemini transcript |
| session_id | string | Session ID |
| position | string | Question position |
| response_id | string | Transcription response ID |
| file | file | Audio file (WAV, MP3, M4A, WebM, max 25MB) |

### Change Password Request
```json
{
  "current_password": "string (required)",
  "new_password": "string (required, min 8)"
}
```

## Error Response Format

```json
{
  "detail": "Human-readable error message"
}
```

## HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 201 | Created |
| 400 | Bad request (validation error) |
| 401 | Unauthorized (invalid/expired token, bad credentials) |
| 403 | Forbidden (insufficient permissions, cross-user access) |
| 404 | Not found |
| 409 | Conflict (duplicate, incomplete workflow) |
| 413 | Payload too large (audio > 25MB) |
| 415 | Unsupported media type (invalid audio format) |
| 422 | Unprocessable entity (validation or business rule) |
| 429 | Rate limit exceeded (Gemini quota) |
| 503 | Service unavailable (backend offline, Gemini not configured) |

## Data Isolation

- Authenticated users can only access their own sessions (403 if cross-user)
- Session listing supports `user_id` filter for explicit scoping
- Student profiles are linked to users via `user_id` foreign key
- Legacy students (without `user_id`) are accessible without auth

## Rate Limiting & Quotas

- Gemini transcription: subject to Google Gemini API quota (429 returned on exhaustion)
- Audio file size: maximum 25MB per upload
- Session question count: 1-50 questions per session
