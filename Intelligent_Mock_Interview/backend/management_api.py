"""Persistent student, question-bank, interview-session and report APIs."""
from __future__ import annotations

import csv
import io
import json
import math
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response, status, Header
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from src.database import (
    AssessmentReport, AudioAssessment, InterviewResponse, InterviewSession,
    QuestionBank, SessionQuestion, Student, TextAssessment, User, UserSession,
    db_session, utcnow,
)
from src.auth import (
    authenticate_user, create_user_session, create_session_token,
    get_current_user_from_token, hash_password, require_role, verify_password,
)

router = APIRouter(prefix="/api", tags=["persistent interview management"])


def _json_safe(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            return str(value)
    return str(value)


class StudentInput(BaseModel):
    full_name: str = Field(min_length=1, max_length=160)
    roll_number: Optional[str] = Field(default=None, max_length=80)
    registration_number: Optional[str] = Field(default=None, max_length=80)
    course: str = Field(default="", max_length=120)
    department: str = Field(default="", max_length=120)
    semester_year: str = Field(default="", max_length=80)
    skills: list[str] = Field(default_factory=list, max_length=40)
    email: Optional[str] = Field(default=None, max_length=254)
    target_job_role: str = Field(default="", max_length=160)
    preferred_domain: str = Field(default="", max_length=100)


class QuestionInput(BaseModel):
    domain: str = Field(min_length=1, max_length=100)
    topic: str = Field(default="General", max_length=120)
    difficulty: str = Field(default="Intermediate", pattern="^(Beginner|Intermediate|Advanced)$")
    question_text: str = Field(min_length=8, max_length=5000)
    reference_answer: Optional[str] = None
    expected_concepts: list[str] = Field(default_factory=list)
    provenance: str = Field(default="Added by an InterviewAI administrator", max_length=240)


class SessionInput(BaseModel):
    student_id: str
    user_id: Optional[str] = None
    domain: str
    difficulty: str = Field(default="Any", pattern="^(Any|Beginner|Intermediate|Advanced)$")
    mode: str = Field(default="text", pattern="^(text|voice|mixed)$")
    question_count: int = Field(default=5, ge=1, le=50)
    time_limit_minutes: Optional[int] = Field(default=None, ge=1, le=300)
    title: Optional[str] = None


class ResponseInput(BaseModel):
    answer_text: Optional[str] = None
    original_transcript: Optional[str] = None
    corrected_transcript: Optional[str] = None
    response_mode: str = "text"


# ── Authentication Models ────────────────────────────────────────────────────

class UserRegisterInput(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=160)
    roll_number: Optional[str] = Field(default=None, max_length=80)
    registration_number: Optional[str] = Field(default=None, max_length=80)
    course: str = Field(default="", max_length=120)
    department: str = Field(default="", max_length=120)
    semester_year: str = Field(default="", max_length=80)
    skills: list[str] = Field(default_factory=list, max_length=40)
    target_job_role: str = Field(default="", max_length=160)
    preferred_domain: str = Field(default="", max_length=100)


class UserLoginInput(BaseModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class ChangePasswordInput(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


def _get_current_user(db, authorization: Optional[str] = None) -> User:
    """Extract and validate the current user from the Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Use: Bearer <token>",
        )
    token = authorization.split(" ", 1)[1]
    user = get_current_user_from_token(db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token. Please log in again.",
        )
    return user


def _user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "account_status": user.account_status,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _student_dict_with_user(student: Student) -> dict:
    data = {name: getattr(student, name) for name in (
        "id", "user_id", "full_name", "roll_number", "registration_number",
        "course", "department", "semester_year", "skills_json", "email",
        "target_job_role", "preferred_domain", "created_at", "updated_at",
    )}
    if data.get("created_at"):
        data["created_at"] = data["created_at"].isoformat()
    if data.get("updated_at"):
        data["updated_at"] = data["updated_at"].isoformat()
    try:
        import json as _json
        data["skills"] = _json.loads(data.get("skills_json") or "[]")
    except Exception:
        data["skills"] = []
    return data


def _student_dict(student: Student):
    data = {name: getattr(student, name) for name in (
        "id", "full_name", "roll_number", "registration_number", "course",
        "department", "semester_year", "email", "target_job_role",
        "preferred_domain", "created_at", "updated_at")}
    try:
        data["skills"] = json.loads(student.skills_json or "[]")
    except (TypeError, json.JSONDecodeError):
        data["skills"] = []
    return data


def _question_dict(question: QuestionBank):
    return {
        "id": question.id, "question_code": question.question_code,
        "domain": question.domain, "topic": question.topic,
        "difficulty": question.difficulty, "question_text": question.question_text,
        "reference_answer": question.reference_answer,
        "expected_concepts": question.expected_concepts,
        "provenance": question.provenance,
    }


def _response_dict(response: Optional[InterviewResponse]):
    if not response:
        return None
    text_assessment = response.text_assessment
    return {
        "id": response.id, "response_mode": response.response_mode,
        "answer_text": response.answer_text,
        "original_transcript": response.original_transcript,
        "corrected_transcript": response.corrected_transcript,
        "audio_retained": response.audio_retained,
        "audio_file_name": response.audio_file_name,
        "audio_url": f"/api/audio/recordings/{response.audio_file_name}" if response.audio_file_name else None,
        "text_assessment": ({
            "relevance_score": text_assessment.relevance_score,
            "question_alignment_score": text_assessment.question_alignment_score,
            "keyword_overlap": text_assessment.keyword_overlap,
            "structure_score": text_assessment.structure_score,
            "quality_score": text_assessment.quality_score,
            "feedback": json.loads(text_assessment.feedback_json or "{}"),
        } if text_assessment else None),
        "audio_measurements": (json.loads(response.audio_assessment.measurements_json or "{}")
                               if response.audio_assessment else None),
    }


def _session_dict(session: InterviewSession):
    questions = []
    for item in session.questions:
        response_by_mode = {r.response_mode: r for r in item.responses}
        questions.append({
            "session_question_id": item.id, "position": item.position,
            "status": item.status, "marked_for_review": item.marked_for_review,
            "question": _question_dict(item.question),
            "response": _response_dict(max(item.responses, key=lambda r: r.updated_at) if item.responses else None),
            "text_response": _response_dict(response_by_mode.get("text")),
            "voice_response": _response_dict(response_by_mode.get("voice")),
        })
    assessed = sum(1 for item in session.questions if item.status == "assessed")
    return {
        "id": session.id, "student_id": session.student_id,
        "student": _student_dict(session.student), "title": session.title,
        "domain": session.domain, "difficulty": session.difficulty,
        "mode": session.mode, "question_count": session.question_count,
        "time_limit_minutes": session.time_limit_minutes,
        "current_position": session.current_position, "status": session.status,
        "text_status": session.text_status, "voice_status": session.voice_status,
        "created_at": session.created_at, "updated_at": session.updated_at,
        "completed_at": session.completed_at,
        "progress": {"assessed": assessed, "total": session.question_count},
        "questions": questions,
        "reports": [{"id": r.id, "version": r.version, "created_at": r.created_at}
                    for r in sorted(session.reports, key=lambda report: report.version)],
    }


@router.get("/students")
def list_students(search: str = ""):
    with db_session() as db:
        stmt = select(Student).order_by(Student.full_name)
        if search.strip():
            term = f"%{search.strip()}%"
            stmt = stmt.where((Student.id.ilike(term)) | (Student.full_name.ilike(term)) | (Student.roll_number.ilike(term)) | (Student.registration_number.ilike(term)))
        return {"students": [_student_dict(s) for s in db.scalars(stmt).all()]}


@router.post("/students", status_code=201)
def create_student(payload: StudentInput):
    data = payload.model_dump()
    skills = data.pop("skills")
    data["skills_json"] = json.dumps([x.strip() for x in skills if x.strip()])
    data = {k: (v.strip() if isinstance(v, str) else v) for k, v in data.items()}
    with db_session() as db:
        duplicate_name = db.scalar(select(Student).where(func.lower(Student.full_name) == data["full_name"].casefold()).limit(1))
        has_identifier = bool(data.get("roll_number") or data.get("registration_number") or data.get("email"))
        if duplicate_name and not has_identifier:
            raise HTTPException(409, "A student with this name already exists. Search the directory or add a roll number, registration number, or email to distinguish the records.")
        student = Student(id=uuid.uuid4().hex[:24], **data)
        db.add(student)
        try:
            db.flush()
        except IntegrityError as exc:
            raise HTTPException(409, "Roll number or registration number is already in use.") from exc
        return _student_dict(student)


@router.get("/students/{student_id}")
def get_student(student_id: str):
    with db_session() as db:
        student = db.get(Student, student_id)
        if not student:
            raise HTTPException(404, "Student not found.")
        sessions = db.scalars(select(InterviewSession).where(InterviewSession.student_id == student_id).options(
            selectinload(InterviewSession.student), selectinload(InterviewSession.questions).selectinload(SessionQuestion.question),
            selectinload(InterviewSession.questions).selectinload(SessionQuestion.responses).selectinload(InterviewResponse.text_assessment),
            selectinload(InterviewSession.questions).selectinload(SessionQuestion.responses).selectinload(InterviewResponse.audio_assessment),
        ).order_by(InterviewSession.created_at.desc())).all()
        return {**_student_dict(student), "sessions": [_session_dict(s) for s in sessions]}


@router.put("/students/{student_id}")
def update_student(student_id: str, payload: StudentInput):
    with db_session() as db:
        student = db.get(Student, student_id)
        if not student:
            raise HTTPException(404, "Student not found.")
        for key, value in payload.model_dump().items():
            if key == "skills":
                student.skills_json = json.dumps([x.strip() for x in value if x.strip()])
            else:
                setattr(student, key, value.strip() if isinstance(value, str) else value)
        try:
            db.flush()
        except IntegrityError as exc:
            raise HTTPException(409, "Roll number or registration number is already in use.") from exc
        return _student_dict(student)


@router.get("/questions")
def list_question_bank(domain: Optional[str] = None, difficulty: Optional[str] = None, search: str = ""):
    with db_session() as db:
        stmt = select(QuestionBank).order_by(QuestionBank.domain, QuestionBank.difficulty, QuestionBank.id)
        if domain:
            stmt = stmt.where(QuestionBank.domain == domain)
        if difficulty and difficulty != "Any":
            stmt = stmt.where(QuestionBank.difficulty == difficulty)
        if search.strip():
            stmt = stmt.where(QuestionBank.question_text.ilike(f"%{search.strip()}%"))
        questions = db.scalars(stmt).all()
        domains = db.execute(select(QuestionBank.domain, func.count(QuestionBank.id)).group_by(QuestionBank.domain)).all()
        return {"questions": [_question_dict(q) for q in questions],
                "domains": [{"domain": d, "count": n} for d, n in domains]}


@router.get("/questions/domains")
def get_question_domains():
    """Returns distinct question domains available in QuestionBank."""
    with db_session() as db:
        domains = db.scalars(select(QuestionBank.domain).distinct().order_by(QuestionBank.domain)).all()
        if not domains:
            domains = [
                "Python", "C++", "Java", "Data Structures & Algorithms",
                "DBMS", "Operating Systems", "Computer Networks", "Machine Learning",
                "Artificial Intelligence", "Web Development", "Software Engineering", "HR & Behavioral"
            ]
        return {"domains": list(domains)}


@router.post("/questions", status_code=201)
def create_question(payload: QuestionInput):
    data = payload.model_dump()
    concepts = data.pop("expected_concepts")
    question = QuestionBank(
        question_code=f"Q-CUSTOM-{uuid.uuid4().hex[:12].upper()}",
        expected_concepts_json=json.dumps(concepts), **data,
    )
    with db_session() as db:
        db.add(question)
        try:
            db.flush()
        except IntegrityError as exc:
            raise HTTPException(409, "This question already exists in the selected domain.") from exc
        return _question_dict(question)


@router.put("/questions/{question_id}")
def update_question(question_id: int, payload: QuestionInput):
    with db_session() as db:
        question = db.get(QuestionBank, question_id)
        if not question:
            raise HTTPException(404, "Question not found.")
        data = payload.model_dump()
        question.expected_concepts_json = json.dumps(data.pop("expected_concepts"))
        for key, value in data.items():
            setattr(question, key, value)
        return _question_dict(question)


@router.delete("/questions/{question_id}")
def delete_question(question_id: int):
    with db_session() as db:
        question = db.get(QuestionBank, question_id)
        if not question:
            raise HTTPException(404, "Question not found.")
        if question.session_questions or question.responses:
            raise HTTPException(409, "A question assigned to an interview cannot be deleted.")
        db.delete(question)
        return {"deleted": True}


@router.post("/sessions", status_code=201)
def create_session(payload: SessionInput):
    with db_session() as db:
        student = db.get(Student, payload.student_id)
        if not student:
            raise HTTPException(404, "Student not found.")
        if payload.user_id and student.user_id != payload.user_id:
            raise HTTPException(403, "This student does not belong to the authenticated user.")
        stmt = select(QuestionBank).where(QuestionBank.domain == payload.domain)
        if payload.difficulty != "Any":
            stmt = stmt.where(QuestionBank.difficulty == payload.difficulty)
        pool = db.scalars(stmt).all()
        if len(pool) < payload.question_count:
            raise HTTPException(422, f"Only {len(pool)} matching questions are available; choose a smaller count or another difficulty.")
        import random
        chosen = random.SystemRandom().sample(pool, payload.question_count)
        session = InterviewSession(
            id=uuid.uuid4().hex, student_id=student.id,
            title=payload.title or f"{payload.domain} {payload.mode.title()} Interview",
            domain=payload.domain, difficulty=payload.difficulty,
            mode=payload.mode if payload.mode in {"text", "voice", "mixed"} else "text",
            question_count=payload.question_count, time_limit_minutes=payload.time_limit_minutes,
            current_position=1, status="in_progress",
            text_status="not_required" if payload.mode == "voice" else "pending",
            voice_status="not_required" if payload.mode == "text" else "pending",
        )
        session.questions = [SessionQuestion(question_id=q.id, position=i) for i, q in enumerate(chosen, 1)]
        db.add(session)
        db.flush()
        return _session_dict(session)


def _get_session(db, session_id: str):
    session = db.scalar(select(InterviewSession).where(InterviewSession.id == session_id).options(
        selectinload(InterviewSession.student),
        selectinload(InterviewSession.questions).selectinload(SessionQuestion.question),
        selectinload(InterviewSession.questions).selectinload(SessionQuestion.responses).selectinload(InterviewResponse.text_assessment),
        selectinload(InterviewSession.questions).selectinload(SessionQuestion.responses).selectinload(InterviewResponse.audio_assessment),
    ))
    if not session:
        raise HTTPException(404, "Interview session not found.")
    return session


@router.get("/sessions")
def list_sessions(student_id: Optional[str] = None, user_id: Optional[str] = None, status: Optional[str] = None, limit: int = Query(100, ge=1, le=500), authorization: Optional[str] = Header(None)):
    with db_session() as db:
        user = _get_current_user(db, authorization)
        student = db.scalar(select(Student).where(Student.user_id == user.id))
        is_admin = user.role == "admin"

        stmt = select(InterviewSession).options(selectinload(InterviewSession.student)).order_by(InterviewSession.created_at.desc()).limit(limit)
        if not is_admin:
            if not student:
                return {"sessions": []}
            target_student_id = student_id if student_id else student.id
            if target_student_id != student.id:
                return {"sessions": []}
            stmt = stmt.where(InterviewSession.student_id == student.id)
        elif user_id:
            stmt = stmt.where(InterviewSession.student_id.in_(select(Student.id).where(Student.user_id == user_id)))
        elif student_id:
            stmt = stmt.where(InterviewSession.student_id == student_id)
        if status:
            if status == "in_progress":
                stmt = stmt.where(InterviewSession.status.in_(["in_progress", "ready_for_report"]))
            else:
                stmt = stmt.where(InterviewSession.status == status)
        sessions = db.scalars(stmt).all()
        return {"sessions": [{"id": s.id, "student_id": s.student_id,
                               "student": _student_dict(s.student) if s.student else {},
                               "student_name": s.student.full_name if s.student else "Unknown",
                               "title": s.title,
                               "domain": s.domain, "difficulty": s.difficulty, "mode": s.mode,
                               "question_count": s.question_count, "current_position": s.current_position,
                               "status": s.status, "text_status": s.text_status,
                               "voice_status": s.voice_status, "created_at": s.created_at,
                               "completed_at": s.completed_at} for s in sessions]}


@router.get("/sessions/{session_id}")
def get_session(session_id: str, authorization: Optional[str] = Header(None)):
    with db_session() as db:
        session = _get_session(db, session_id)
        # Enforce ownership when authenticated; allow legacy access only when no auth header is provided.
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ", 1)[1]
            user = get_current_user_from_token(db, token)
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session token. Please log in again.")
            if user.role != "admin":
                student = db.scalar(select(Student).where(Student.user_id == user.id))
                if not student or session.student_id != student.id:
                    raise HTTPException(403, "You can only access your own sessions.")
        return _session_dict(session)


@router.put("/sessions/{session_id}/position")
def set_session_position(session_id: str, position: int = Query(ge=1)):
    with db_session() as db:
        session = _get_session(db, session_id)
        if position > session.question_count:
            raise HTTPException(422, "Position is outside the session question range.")
        session.current_position = position
        return {"current_position": position}


@router.put("/sessions/{session_id}/questions/{position}/response")
def save_response(session_id: str, position: int, payload: ResponseInput):
    with db_session() as db:
        session = _get_session(db, session_id)
        item = next((q for q in session.questions if q.position == position), None)
        if not item:
            raise HTTPException(404, "Question position not found.")
        mode = payload.response_mode if payload.response_mode in {"text", "voice"} else "text"
        response = next((r for r in item.responses if r.response_mode == mode), None)
        if response is None:
            response = InterviewResponse(student_id=session.student_id, session_id=session.id,
                                         session_question_id=item.id, question_id=item.question_id,
                                         response_mode=mode)
            db.add(response)
        response.answer_text = payload.answer_text
        response.original_transcript = payload.original_transcript
        response.corrected_transcript = payload.corrected_transcript
        item.status = "answered" if (payload.answer_text or "").strip() or (payload.corrected_transcript or "").strip() else "pending"
        session.current_position = position
        if mode == "text" and session.text_status == "pending":
            session.text_status = "in_progress"
        if mode == "voice" and session.voice_status == "pending":
            session.voice_status = "in_progress"
        db.flush()
        return _response_dict(response)


@router.put("/sessions/{session_id}/questions/{position}/status")
def set_question_status(session_id: str, position: int, status: str = Query(pattern="^(pending|answered|skipped|assessed)$"), review: bool = False):
    with db_session() as db:
        session = _get_session(db, session_id)
        item = next((q for q in session.questions if q.position == position), None)
        if not item:
            raise HTTPException(404, "Question position not found.")
        item.status, item.marked_for_review = status, review
        return {"position": position, "status": status, "marked_for_review": review}


@router.post("/sessions/{session_id}/complete")
def complete_session(session_id: str):
    with db_session() as db:
        session = _get_session(db, session_id)
        if not all(getattr(session, f"{name}_status") in {"completed", "not_required"} for name in ("text", "voice")):
            raise HTTPException(409, "Complete the required assessment modes before completing this session.")
        if not session.reports:
            raise HTTPException(409, "Generate the final student report before closing this session.")
        session.status, session.completed_at = "completed", utcnow()
        return {"id": session.id, "status": session.status, "completed_at": session.completed_at}


@router.post("/sessions/{session_id}/modes/{mode}/complete")
def complete_assessment_mode(session_id: str, mode: str):
    if mode not in {"text", "voice"}:
        raise HTTPException(422, "Mode must be text or voice.")
    with db_session() as db:
        session = _get_session(db, session_id)
        mode_status = getattr(session, f"{mode}_status")
        if mode_status == "not_required":
            raise HTTPException(409, f"The {mode} mode is not part of this assessment.")
        for item in session.questions:
            response = next((r for r in item.responses if r.response_mode == mode), None)
            if not response:
                raise HTTPException(409, f"Question {item.position} has no saved {mode} response.")
            if mode == "text" and not response.text_assessment:
                raise HTTPException(409, f"Question {item.position} has not been text assessed.")
            if mode == "voice" and not response.audio_assessment:
                raise HTTPException(409, f"Question {item.position} has no saved voice analysis.")
        setattr(session, f"{mode}_status", "completed")
        if all(getattr(session, f"{name}_status") in {"completed", "not_required"} for name in ("text", "voice")):
            session.status = "ready_for_report"
        return {"session_id": session.id, "mode": mode, "status": "completed",
                "text_status": session.text_status, "voice_status": session.voice_status,
                "session_status": session.status}


def _make_final_report(session: InterviewSession, version: int) -> dict:
    results = []
    text_scores, recommendations = [], []
    text_available = voice_available = 0
    answered = 0
    completed_mode_responses = 0
    required_mode_count = int(session.text_status != "not_required") + int(session.voice_status != "not_required")
    for item in session.questions:
        by_mode = {r.response_mode: r for r in item.responses}
        text_response, voice_response = by_mode.get("text"), by_mode.get("voice")
        text_assessment = text_response.text_assessment if text_response else None
        voice_measurements = None
        if voice_response and voice_response.audio_assessment:
            try:
                voice_measurements = json.loads(voice_response.audio_assessment.measurements_json or "{}")
            except (TypeError, json.JSONDecodeError):
                voice_measurements = {}
        transcript = voice_response.corrected_transcript if voice_response else None
        original = voice_response.original_transcript if voice_response else None
        has_text = bool(text_response and (text_response.answer_text or "").strip())
        has_voice = bool(voice_response and ((transcript or "").strip() or voice_measurements))
        answered += int(has_text or has_voice)
        if session.text_status != "not_required":
            completed_mode_responses += int(has_text)
        if session.voice_status != "not_required":
            completed_mode_responses += int(has_voice)
        if text_assessment:
            text_available += 1
            if isinstance(text_assessment.quality_score, (int, float)) and math.isfinite(float(text_assessment.quality_score)):
                text_scores.append(float(text_assessment.quality_score))
            try:
                feedback = json.loads(text_assessment.feedback_json or "{}")
            except (TypeError, json.JSONDecodeError):
                feedback = {}
            recommendations.extend(feedback.get("improvements", []))
        else:
            feedback = {}
        if has_voice:
            voice_available += 1
        result = {
            "position": item.position,
            "question_id": item.question_id,
            "question": item.question.question_text,
            "text": {
                "answer": text_response.answer_text if text_response else None,
                "strengths": feedback.get("strengths", []),
                "improvement_areas": feedback.get("improvements", []),
                "assessment": ({
                    "score": text_assessment.quality_score,
                    "cosine_similarity": text_assessment.relevance_score,
                    "question_alignment": text_assessment.question_alignment_score,
                    "jaccard_similarity": text_assessment.keyword_overlap,
                    "structure_score": text_assessment.structure_score,
                    "feedback": feedback,
                } if text_assessment else None),
            },
            "voice": {
                "original_transcript": original,
                "corrected_transcript": transcript,
                "audio_file_retained": False,
                "measurements": voice_measurements,
            },
            "status": item.status,
            "observation": ("Text assessment and measured voice features are available." if text_assessment and has_voice
                            else "Text assessment is available; voice analysis is pending." if text_assessment
                            else "Voice analysis is available; text assessment is pending." if has_voice
                            else "No saved response is available for this question."),
        }
        results.append(result)
    total = len(results)
    text_state = session.text_status
    voice_state = session.voice_status
    modes_complete = all(value in {"completed", "not_required"} for value in (text_state, voice_state))
    return {
        "report_type": "Student Interview Assessment Report",
        "report_version": version,
        "student": {
            "student_id": session.student.id, "full_name": session.student.full_name,
            "course": session.student.course, "department": session.student.department,
            "semester_year": session.student.semester_year,
            "roll_number": session.student.roll_number,
            "skills": json.loads(session.student.skills_json or "[]"),
        },
        "assessment": {
            "assessment_id": session.id, "title": session.title,
            "domain": session.domain, "difficulty": session.difficulty,
            "mode": session.mode, "assessment_date": session.created_at.isoformat(),
            "total_questions": total, "answered_questions": answered,
            "completed_mode_responses": completed_mode_responses,
            "required_mode_responses": total * required_mode_count,
            "completion_percentage": round(completed_mode_responses * 100 / (total * required_mode_count), 1) if total and required_mode_count else 0,
            "text_status": text_state, "voice_status": voice_state,
            "interview_status": "completed" if modes_complete else "in_progress",
        },
        "text_assessment": {
            "status": text_state,
            "average_text_assessment_score": round(sum(text_scores) / len(text_scores), 4) if text_scores else None,
            "score_formula": "0.50 * TF-IDF reference cosine + 0.20 * TF-IDF question cosine + 0.15 * Jaccard keyword overlap + 0.15 * answer structure heuristic",
            "score_limitation": "Practice heuristic; not calibrated against human ratings or intended as a hiring decision.",
            "questions_assessed": text_available,
        },
        "voice_assessment": {
            "status": voice_state, "questions_with_voice_analysis": voice_available,
            "score_note": "Acoustic measurements are descriptive and are not converted into a voice quality score.",
        },
        "question_results": results,
        "multimodal_summary": {
            "combined_numeric_score": None,
            "explanation": "No validated common scale exists for combining the text heuristic with raw acoustic measurements. They are presented independently.",
            "observations": [item["observation"] for item in results],
            "recommendations": list(dict.fromkeys(recommendations)),
        },
        "final_conclusion": {
            "status": "completed" if modes_complete else "in_progress",
            "summary": f"{answered} of {total} questions have saved responses; text and voice findings remain separate.",
            "next_steps": list(dict.fromkeys(recommendations)) or ["Continue practicing clear explanations with examples and review each question's feedback."],
        },
    }


@router.get("/sessions/{session_id}/reports/final")
@router.post("/sessions/{session_id}/reports/final")
@router.post("/sessions/{session_id}/final-report")
def create_final_report(session_id: str):
    with db_session() as db:
        session = _get_session(db, session_id)
        if not all(getattr(session, f"{name}_status") in {"completed", "not_required"} for name in ("text", "voice")):
            raise HTTPException(409, "Complete every assessment mode in this session before generating the final report.")
        previous_version = db.scalar(select(func.max(AssessmentReport.version)).where(AssessmentReport.session_id == session_id)) or 0
        report = _make_final_report(session, previous_version + 1)
        record = AssessmentReport(session_id=session.id, student_id=session.student_id,
                                  version=previous_version + 1,
                                  report_json=json.dumps(_json_safe(report), ensure_ascii=False, allow_nan=False))
        db.add(record)
        session.status, session.completed_at = "completed", utcnow()
        db.flush()
        return {"id": record.id, "session_id": session.id, "student_id": session.student_id,
                "version": record.version, "created_at": record.created_at, "report": report}


@router.get("/sessions/{session_id}/reports")
def list_reports(session_id: str, authorization: Optional[str] = Header(None)):
    with db_session() as db:
        user = _get_current_user(db, authorization)
        session = _get_session(db, session_id)
        if user.role != "admin":
            student = db.scalar(select(Student).where(Student.user_id == user.id))
            if not student or session.student_id != student.id:
                raise HTTPException(403, "You can only access reports for your own sessions.")
        reports = db.scalars(select(AssessmentReport).where(AssessmentReport.session_id == session_id).order_by(AssessmentReport.created_at.desc())).all()
        return {"reports": [{"id": r.id, "version": r.version, "student_id": r.student_id,
                              "created_at": r.created_at, "report": json.loads(r.report_json)} for r in reports]}


@router.get("/reports/{report_id}")
def get_report(report_id: int):
    with db_session() as db:
        report = db.get(AssessmentReport, report_id)
        if not report:
            raise HTTPException(404, "Assessment report not found.")
        return {"id": report.id, "version": report.version, "student_id": report.student_id,
                "session_id": report.session_id, "created_at": report.created_at,
                "report": json.loads(report.report_json)}


@router.get("/dashboard/summary")
@router.get("/analytics")
def dashboard_summary():
    with db_session() as db:
        students = db.scalar(select(func.count(Student.id))) or 0
        questions = db.scalar(select(func.count(QuestionBank.id))) or 0
        sessions = db.scalar(select(func.count(InterviewSession.id))) or 0
        completed = db.scalar(select(func.count(InterviewSession.id)).where(InterviewSession.status == "completed")) or 0
        in_progress = db.scalar(select(func.count(InterviewSession.id)).where(InterviewSession.status == "in_progress")) or 0
        ready = db.scalar(select(func.count(InterviewSession.id)).where(InterviewSession.status == "ready_for_report")) or 0
        recent = db.scalars(select(InterviewSession).options(selectinload(InterviewSession.student)).order_by(InterviewSession.created_at.desc()).limit(8)).all()
        domains = db.execute(select(QuestionBank.domain, func.count(QuestionBank.id)).group_by(QuestionBank.domain).order_by(QuestionBank.domain)).all()
        # Average text quality score across all assessed responses
        avg_score = db.scalar(
            select(func.avg(TextAssessment.quality_score)).where(
                TextAssessment.quality_score.isnot(None)
            )
        )
        return {
            "students": students,
            "total_students": students,
            "sessions": sessions,
            "total_sessions": sessions,
            "completed_sessions": completed,
            "in_progress_sessions": in_progress,
            "ready_for_report_sessions": ready,
            "completion_rate": round(completed / sessions * 100, 1) if sessions else None,
            "average_assessment_score": round(float(avg_score), 4) if avg_score is not None else None,
            "domains": [{"domain": d, "count": n} for d, n in domains],
            "recent_sessions": [{"id": s.id, "student_name": s.student.full_name,
                                 "title": s.title, "domain": s.domain, "status": s.status,
                                 "created_at": s.created_at} for s in recent],
        }


@router.get("/export/students.csv")
def export_students():
    with db_session() as db:
        rows = db.scalars(select(Student).order_by(Student.full_name)).all()
        output = io.StringIO(newline="")
        fields = ["id", "full_name", "roll_number", "registration_number", "course", "department", "semester_year", "email", "target_job_role", "preferred_domain"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for student in rows:
            writer.writerow({key: getattr(student, key) for key in fields})
        return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=interviewai-students.csv"})


# ── Authentication Endpoints ─────────────────────────────────────────────────

@router.post("/auth/register", status_code=201)
def register_user(payload: UserRegisterInput):
    """Register a new user account and linked student profile."""
    email = payload.email.strip().lower()
    with db_session() as db:
        existing = db.scalar(select(User).where(User.email == email))
        if existing:
            raise HTTPException(409, "An account with this email already exists.")
        user = User(
            email=email,
            password_hash=hash_password(payload.password),
            role="student",
            account_status="active",
        )
        db.add(user)
        db.flush()
        student = Student(
            id=uuid.uuid4().hex[:24],
            user_id=user.id,
            full_name=payload.full_name.strip(),
            roll_number=payload.roll_number.strip() if payload.roll_number else None,
            registration_number=payload.registration_number.strip() if payload.registration_number else None,
            course=payload.course,
            department=payload.department,
            semester_year=payload.semester_year,
            skills_json=json.dumps([s.strip() for s in payload.skills if s.strip()]),
            email=email,
            target_job_role=payload.target_job_role.strip(),
            preferred_domain=payload.preferred_domain,
        )
        db.add(student)
        try:
            db.flush()
        except IntegrityError as exc:
            raise HTTPException(409, "Roll number or registration number is already in use.") from exc
        session = create_user_session(db, user)
        return {
            "user": _user_dict(user),
            "student": _student_dict_with_user(student),
            "access_token": session.token,
            "token_type": "bearer",
            "expires_in": 86400,
        }


@router.post("/auth/login")
def login_user(payload: UserLoginInput):
    """Authenticate a user and return a session token and linked student profile."""
    email = payload.email.strip().lower()
    with db_session() as db:
        user = authenticate_user(db, email, payload.password)
        if not user:
            raise HTTPException(401, "Invalid email or password.")
        session = create_user_session(db, user)
        student = db.scalar(select(Student).where(Student.user_id == user.id))
        return {
            "access_token": session.token,
            "token_type": "bearer",
            "expires_in": 86400,
            "user": _user_dict(user),
            "student": _student_dict_with_user(student) if student else None,
        }


@router.post("/auth/logout")
def logout_user(authorization: Optional[str] = Header(None)):
    """Invalidate the current session token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    with db_session() as db:
        session = db.scalar(select(UserSession).where(UserSession.token == token))
        if session:
            db.delete(session)
    return {"message": "Logged out successfully."}


@router.get("/auth/me")
def get_current_user_profile(authorization: Optional[str] = Header(None)):
    """Return the currently authenticated user's profile."""
    with db_session() as db:
        user = _get_current_user(db, authorization)
        student = db.scalar(select(Student).where(Student.user_id == user.id))
        return {
            "user": _user_dict(user),
            "student": _student_dict_with_user(student) if student else None,
        }


@router.post("/auth/change-password")
def change_user_password(payload: ChangePasswordInput, authorization: Optional[str] = Header(None)):
    """Change the authenticated user's password and invalidate other sessions."""
    with db_session() as db:
        user = _get_current_user(db, authorization)
        if not verify_password(payload.current_password, user.password_hash):
            raise HTTPException(401, "Current password is incorrect.")
        user.password_hash = hash_password(payload.new_password)
        # Invalidate other active sessions for security
        current_token = authorization.split(" ", 1)[1] if authorization and " " in authorization else None
        if current_token:
            other_sessions = db.scalars(
                select(UserSession).where(UserSession.user_id == user.id, UserSession.token != current_token)
            ).all()
            for s in other_sessions:
                db.delete(s)
        return {"message": "Password updated successfully. Other active sessions revoked."}


# ── Authorization Helpers for Protected Endpoints ────────────────────────────

def require_auth(authorization: Optional[str] = None) -> User:
    """Dependency-style auth check for endpoints that receive the header directly."""
    with db_session() as db:
        return _get_current_user(db, authorization)


def require_student(authorization: Optional[str] = None) -> User:
    """Require an authenticated student user."""
    user = require_auth(authorization)
    require_role(user, {"student", "admin"})
    return user


def require_admin(authorization: Optional[str] = None) -> User:
    """Require an authenticated admin user."""
    user = require_auth(authorization)
    require_role(user, {"admin"})
    return user


def persist_assessment(session_id: str, position: int, response_mode: str, report: dict) -> None:
    """Persist a completed API assessment if the caller supplied a session link."""
    with db_session() as db:
        session = _get_session(db, session_id)
        item = next((q for q in session.questions if q.position == position), None)
        if not item:
            raise HTTPException(404, "Question position not found in session.")
        mode = response_mode if response_mode in {"voice", "text"} else "text"
        response = next((r for r in item.responses if r.response_mode == mode), None)
        if response is None:
            response = InterviewResponse(student_id=session.student_id, session_id=session.id,
                                         session_question_id=item.id, question_id=item.question_id,
                                         response_mode=mode)
            db.add(response)
        information = report.get("interview_information", {})
        response.answer_text = information.get("candidate_answer") or response.answer_text
        response.original_transcript = information.get("original_transcript") or response.original_transcript
        response.corrected_transcript = information.get("corrected_transcript") or response.corrected_transcript
        text = report.get("text_assessment") or {}
        assessment = response.text_assessment or TextAssessment(response=response)
        for field in ("relevance_score", "question_alignment_score", "keyword_overlap", "structure_score", "quality_score"):
            value = text.get(field)
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0 <= float(value) <= 1
            setattr(assessment, field, float(value) if valid else None)
        assessment.feedback_json = json.dumps({"strengths": text.get("strengths", []), "improvements": text.get("improvements", [])})
        db.add(assessment)
        audio_name = report.get("audio_file_name")
        if audio_name:
            response.audio_file_name = audio_name
            response.audio_retained = True
        voice = report.get("voice_assessment")
        if isinstance(voice, dict):
            audio = response.audio_assessment or AudioAssessment(response=response)
            audio.measurements_json = json.dumps(_json_safe(voice), allow_nan=False)
            db.add(audio)
        item.status = "assessed"
        session.current_position = position
        # Advance mode status: pending -> in_progress -> completed when all questions in this mode are assessed
        stage_field = f"{mode}_status"
        current_stage = getattr(session, stage_field)
        if current_stage == "pending":
            setattr(session, stage_field, "in_progress")
        # Check if all questions in this mode now have assessments
        all_assessed = True
        for item_check in session.questions:
            resp = next((r for r in item_check.responses if r.response_mode == mode), None)
            if mode == "text":
                if not resp or not resp.text_assessment:
                    all_assessed = False
                    break
            else:  # voice
                if not resp or not resp.audio_assessment:
                    all_assessed = False
                    break
        if all_assessed and current_stage in ("pending", "in_progress"):
            setattr(session, stage_field, "completed")
        # Sync session-level status
        if all(
            getattr(session, f"{name}_status") in {"completed", "not_required"}
            for name in ("text", "voice")
        ):
            session.status = "ready_for_report"
        db.flush()
