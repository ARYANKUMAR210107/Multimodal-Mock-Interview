# Intelligent Multimodal Mock Interview Assessment System (InterviewAI)

A complete, professional, multi-user online interview practice and assessment platform.
Students from different colleges can independently register, log in, practice interviews,
receive AI-based assessments, and access their personal results.

## Features

- **Multi-user authentication**: Secure registration, login, logout with bcrypt password hashing
- **Role-based access**: Student and Admin dashboards with separate navigation
- **Persistent interview workflow**: Registration → Dashboard → Create Interview → Text Answer → Voice Recording → Gemini Transcription → Evaluation → Final Report → History
- **Data isolation**: Each student sees only their own profile, sessions, and reports
- **Resume incomplete interviews**: Saved progress persists across navigation and browser refresh
- **Gemini speech transcription**: Verbose transcription with editable transcript, retry, and proper error handling
- **Text evaluation**: TF-IDF similarity scoring against reference answers
- **Audio analysis**: Librosa acoustic feature extraction (pitch, RMS, MFCC, pause ratio, WPM, filler words)
- **Employability prediction**: Gradient Boosting classifier for soft-skill assessment
- **Professional UI**: Dark theme with Electric Cyan/Electric Blue accents, responsive design

## Tech Stack

- **Backend**: FastAPI (Python 3.13), SQLAlchemy ORM, SQLite (development) / PostgreSQL (production)
- **Frontend**: Streamlit with custom CSS design system
- **AI/ML**: Google Gemini 3.5 Transcribe, Librosa audio analysis, Scikit-Learn
- **Migrations**: Alembic
- **Authentication**: bcrypt password hashing, session tokens (24-hour TTL)

## Project Structure

```
Intelligent_Mock_Interview/
├── alembic/                      ← Database migration scripts
├── alembic.ini                   ← Alembic configuration
├── backend/
│   ├── main.py                   ← FastAPI application entry point
│   └── management_api.py         ← Persistent API endpoints + auth routes
├── frontend/
│   ├── app.py                    ← Streamlit entrypoint & router
│   ├── pages/
│   │   ├── auth_page.py          ← Login & registration UI
│   │   ├── dashboard.py          ← Student/Admin dashboard
│   │   ├── students.py           ← Directory, registration, profile
│   │   ├── new_assessment.py     ← Interview creation wizard
│   │   ├── interview_workspace.py│ ← Active interview (text + voice)
│   │   ├── assessment_history.py ← Past assessments & reports
│   │   ├── analytics.py          ← Cohort analytics
│   │   ├── question_bank.py      ← Question management
│   │   ├── system_status.py      ← System health diagnostics
│   │   └── styles/theme.css      ← Design system (navy/cyan/purple)
│   └── services/
│       └── api_client.py         ← HTTP client with auth token management
├── src/
│   ├── database.py               ← SQLAlchemy models & session management
│   ├── auth.py                   ← Password hashing, session tokens, authorization
│   ├── test_all_endpoints.py     ← FastAPI endpoint tests (23 tests)
│   ├── test_database_management.py ← Database CRUD tests (5 tests)
│   ├── test_multimodal_complete.py ← Audio/text assessment tests (13 tests)
│   └── test_end_to_end_audit.py  ← Full E2E workflow test (1 test)
├── data/
│   ├── interviewai.sqlite3       ← SQLite database (development)
│   └── question_bank_seed.json   ← Seed questions
├── models/                       ← Trained ML models (.joblib)
├── dataset/                      ← Datasets (employability, interview questions)
├── documentation/                ← Technical audits and guides
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

## Quick Start

### Prerequisites

- Python 3.13
- Virtual environment at `c:\Users\Asus\OneDrive\Desktop\intelligent_supervised\venv\`

### 1. Activate the virtual environment

```powershell
cd c:\Users\Asus\OneDrive\Desktop\intelligent_supervised
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
cd Intelligent_Mock_Interview
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` (at the project root) and set your Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Do not overwrite the existing `.env` if one exists.

### 4. Start the backend (FastAPI)

```powershell
cd Intelligent_Mock_Interview
..\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

### 5. Start the frontend (Streamlit)

```powershell
cd Intelligent_Mock_Interview
..\venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

The app opens at: `http://localhost:8501`

## Authentication Flow

1. **Register**: Create an account at `http://localhost:8501` (Sign Up tab)
2. **Login**: Sign in with email and password
3. **Dashboard**: View personal statistics and recent assessments
4. **Start Interview**: Create a new interview session
5. **My Assessments**: View past sessions and resume incomplete ones
6. **My Reports**: View generated final reports
7. **My Profile**: Manage your student profile and change password

## Database Migrations

Alembic is configured for schema migrations:

```powershell
# Create a new migration
..\venv\Scripts\python.exe -m alembic revision --autogenerate -m "description"

# Apply migrations
..\venv\Scripts\python.exe -m alembic upgrade head

# Check current revision
..\venv\Scripts\python.exe -m alembic current
```

The existing SQLite database is preserved. Migrations are additive only.

## Running Tests

```powershell
cd Intelligent_Mock_Interview
..\venv\Scripts\python.exe -m unittest src.test_all_endpoints src.test_database_management src.test_multimodal_complete src.test_end_to_end_audit
```

**Results: 43/43 tests passing (100%)**

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register user + student profile
- `POST /api/auth/login` - Authenticate and get session token
- `POST /api/auth/logout` - Invalidate session
- `GET /api/auth/me` - Get current user profile
- `POST /api/auth/change-password` - Change password

### Students & Sessions
- `POST /api/students` - Create student profile (legacy)
- `GET /api/students` - List students
- `GET /api/students/{id}` - Get student details
- `POST /api/sessions` - Create interview session
- `GET /api/sessions` - List sessions (supports `student_id`, `user_id` filters)
- `GET /api/sessions/{id}` - Get session details
- `PUT /api/sessions/{id}/questions/{position}/response` - Save answer draft

### Assessment
- `POST /api/v1/assess/text` - Evaluate text response
- `POST /api/interview/multimodal-report` - Voice assessment with audio analysis
- `POST /api/audio/transcribe` - Gemini speech transcription
- `POST /api/sessions/{id}/final-report` - Generate final report
- `GET /api/sessions/{id}/reports` - List session reports

### System
- `GET /health` - Health check
- `GET /api/analytics` - Dashboard analytics
- `GET /api/v1/questions` - Interview questions

## Production Deployment

### Database
For production, use PostgreSQL instead of SQLite:

```env
INTERVIEWAI_DATABASE_URL=postgresql+psycopg2://user:password@host:port/dbname
```

### Security
- Set `GEMINI_API_KEY` in environment (never in source code)
- Use HTTPS in production
- Configure proper CORS origins (not `*`)
- Set up a reverse proxy (nginx) with rate limiting
- Use a production WSGI/ASGI server (gunicorn, uvicorn with workers)

### Environment Variables
| Variable | Description | Default |
|---|---|---|
| `INTERVIEWAI_DATABASE_URL` | Database connection string | `sqlite:///data/interviewai.sqlite3` |
| `GEMINI_API_KEY` | Google Gemini API key | (required for transcription) |
| `MOCK_INTERVIEW_API_URL` | Backend URL for frontend | `http://127.0.0.1:8000` |

## Known Issues

- Gemini transcription requires a valid API key with available quota
- No email verification or password reset flow yet

## Security Summary

- All session endpoints (`GET /api/sessions`, `GET /api/sessions/{id}/reports`) are protected by authorization
- Students can only access their own sessions and reports (403 for cross-user access)
- Admin users can list all sessions and filter by `user_id`
- Session tokens use `secrets.token_urlsafe(32)` with 24-hour TTL
- Passwords are hashed with bcrypt
- Legacy student records without a linked `user_id` remain supported for backward compatibility

## License

Academic project - B.Tech Final Year
