# Production Hardening & Deployment Guide

**InterviewAI** — Multi-User Mock Interview Assessment Platform  
**Version:** 1.0.0  
**Updated:** 2026-11-10

---

## 1. Overview

This guide covers production deployment hardening for InterviewAI. The platform has been upgraded from a local prototype to a secure multi-user system with:

- **Authentication**: bcrypt password hashing, session tokens (24-hour TTL)
- **Authorization**: role-based access (student/admin), per-user data isolation
- **Protected endpoints**: `GET /api/sessions` and `GET /api/sessions/{id}/reports` require authentication and ownership checks
- **Cross-user isolation**: 403 Forbidden on unauthorized access to another user's sessions/reports

---

## 2. Pre-Deployment Checklist

### 2.1 Environment Variables

Create a `.env` file at the project root (never commit it):

```env
# Database (use PostgreSQL in production)
INTERVIEWAI_DATABASE_URL=postgresql+psycopg2://user:password@host:5432/interviewai

# Gemini API (required for transcription)
GEMINI_API_KEY=your_gemini_api_key_here

# Frontend backend URL
MOCK_INTERVIEW_API_URL=https://api.yourdomain.com

# Secret key for session signing (production: use a strong random value)
SESSION_SECRET_KEY=change-this-to-a-strong-random-secret

# CORS allowed origins (production: list specific domains, not *)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 2.2 Database Migration

```powershell
# Create a new migration after model changes
python -m alembic revision --autogenerate -m "description"

# Apply migrations to production database
python -m alembic upgrade head

# Verify current revision
python -m alembic current
```

### 2.3 Database Backup

Before any production deploy, back up the existing SQLite database:

```powershell
Copy-Item data\interviewai.sqlite3 data\interviewai.sqlite3.pre-deploy-backup
```

---

## 3. Server Deployment

### 3.1 FastAPI Backend (Production)

Use a production ASGI server with multiple workers:

```powershell
# Install production server
pip install uvicorn[standard]

# Run with multiple workers (adjust --workers to CPU count)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3.2 Reverse Proxy (nginx)

Place nginx in front of FastAPI for TLS termination, rate limiting, and CORS:

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # CORS headers
    add_header Access-Control-Allow-Origin "https://yourdomain.com" always;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Authorization, Content-Type" always;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3.3 Streamlit Frontend

```powershell
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

For production, serve Streamlit behind nginx as well, or use Streamlit's built-in production options.

---

## 4. Security Hardening

### 4.1 CORS Configuration

The development backend uses `allow_origins=["*"]`. For production, restrict to specific domains:

```python
# backend/main.py — update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### 4.2 Rate Limiting

Apply rate limiting at the reverse proxy (nginx `limit_req`) or via FastAPI middleware. Critical endpoints:

- `POST /api/auth/login` — brute-force protection
- `POST /api/auth/register` — prevent account enumeration/spam
- `POST /api/audio/transcribe` — Gemini quota protection

### 4.3 Secrets Management

- Never commit `.env` or API keys to version control
- Use environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault)
- Rotate `GEMINI_API_KEY` periodically
- Use a strong `SESSION_SECRET_KEY` (minimum 32 random characters)

### 4.4 HTTPS Everywhere

- Enforce HTTPS in production via nginx TLS termination
- Set `Secure` and `HttpOnly` flags on session cookies
- Enable HSTS header

---

## 5. Authentication & Authorization

### 5.1 Endpoint Protection Summary

| Endpoint | Auth Required | Access Control |
|---|---|---|
| `POST /api/auth/register` | No | Public |
| `POST /api/auth/login` | No | Public |
| `POST /api/auth/logout` | Yes | Owner only |
| `GET /api/auth/me` | Yes | Owner only |
| `POST /api/auth/change-password` | Yes | Owner only |
| `GET /api/sessions` | Yes | User's own sessions only |
| `GET /api/sessions/{id}` | No (optional) | Owner only if authenticated |
| `GET /api/sessions/{id}/reports` | Yes | Owner or admin |
| `POST /api/sessions` | No | Legacy (student_id-based) |
| `GET /api/students/{id}` | No | Legacy (public directory) |

### 5.2 Admin Access

Admin users (`role = "admin"`) can:
- List all sessions via `GET /api/sessions`
- Filter sessions by `user_id` parameter
- Access any student's profile

Regular students (`role = "student"`) can only access their own sessions and reports.

---

## 6. Database Configuration

### 6.1 SQLite (Development Only)

Default configuration uses SQLite at `data/interviewai.sqlite3`. Not recommended for production due to concurrency limitations.

### 6.2 PostgreSQL (Production Recommended)

```env
INTERVIEWAI_DATABASE_URL=postgresql+psycopg2://user:password@host:5432/interviewai
```

SQLAlchemy models are DB-agnostic; the same Alembic migrations work with PostgreSQL.

---

## 7. Monitoring & Logging

### 7.1 Application Logging

InterviewAI logs structured events to stdout/stderr:

- `audio_validation_started/completed/failed`
- `gemini_upload_started/completed`
- `gemini_transcription_started/completed`
- `gemini_client_error`, `gemini_server_error`, `gemini_timeout`
- `request_completed` (with method, path, status_code, elapsed_ms)

Forward these logs to your logging infrastructure (ELK, CloudWatch, etc.).

### 7.2 Health Check

```powershell
curl https://api.yourdomain.com/health
```

Returns `{"status": "online", ...}` when the service is healthy.

---

## 8. Known Limitations

- No email verification or password reset flow yet
- Session tokens are stored in-memory (SQLite `user_sessions` table); restart invalidates all sessions
- Legacy student records without `user_id` remain accessible without auth (backward compatibility)
- Gemini transcription requires a valid API key with available quota (429 on exhaustion)

---

## 9. Rollback Plan

1. Restore database from pre-deploy backup
2. Revert code to previous stable commit
3. Restart services
4. Verify `/health` endpoint returns online