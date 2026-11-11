"""SQLAlchemy models and SQLite lifecycle for durable InterviewAI records."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    select,
    text as sql_text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_DIR / "data" / "interviewai.sqlite3"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
DATABASE_URL = os.getenv("INTERVIEWAI_DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    # Keep naive UTC values for SQLite's DateTime columns without the
    # deprecated datetime.utcnow() constructor.
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_role", "role"),
    )

    id: Mapped[str] = mapped_column(String(24), primary_key=True, default=lambda: uuid.uuid4().hex[:24])
    email: Mapped[str] = mapped_column(String(254), nullable=False, index=True, unique=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="student")
    account_status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    student: Mapped[Optional["Student"]] = relationship(back_populates="user", uselist=False)


class UserSession(Base):
    __tablename__ = "user_sessions"
    __table_args__ = (
        Index("ix_user_sessions_user_id", "user_id"),
        Index("ix_user_sessions_expires_at", "expires_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    token: Mapped[str] = mapped_column(String(255), nullable=False, index=True, unique=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    user: Mapped[User] = relationship(backref="sessions")


class Student(Base):
    __tablename__ = "students"
    __table_args__ = (
        UniqueConstraint("roll_number", name="uq_students_roll_number"),
        UniqueConstraint("registration_number", name="uq_students_registration_number"),
        UniqueConstraint("email", name="uq_students_email"),
        Index("ix_students_full_name", "full_name"),
        Index("ix_students_user_id", "user_id"),
    )

    id: Mapped[str] = mapped_column(String(24), primary_key=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(24), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    roll_number: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    registration_number: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    course: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    department: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    semester_year: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    skills_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    email: Mapped[Optional[str]] = mapped_column(String(254), nullable=True)
    target_job_role: Mapped[str] = mapped_column(String(160), nullable=False, default="")
    preferred_domain: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    user: Mapped[Optional["User"]] = relationship(back_populates="student")
    sessions: Mapped[list["InterviewSession"]] = relationship(back_populates="student")
    responses: Mapped[list["InterviewResponse"]] = relationship(back_populates="student")


class QuestionBank(Base):
    __tablename__ = "question_bank"
    __table_args__ = (
        UniqueConstraint("domain", "question_text", name="uq_questions_domain_text"),
        UniqueConstraint("source_id", name="uq_questions_source_id"),
        Index("ix_questions_domain_difficulty", "domain", "difficulty"),
        Index("ix_questions_topic", "topic"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_code: Mapped[str] = mapped_column(String(24), unique=True, nullable=False)
    source_id: Mapped[Optional[str]] = mapped_column(String(180), nullable=True)
    domain: Mapped[str] = mapped_column(String(100), nullable=False)
    topic: Mapped[str] = mapped_column(String(120), nullable=False, default="General")
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False, default="Intermediate")
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected_concepts_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    provenance: Mapped[str] = mapped_column(String(240), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    session_questions: Mapped[list["SessionQuestion"]] = relationship(back_populates="question")
    responses: Mapped[list["InterviewResponse"]] = relationship(back_populates="question")

    @property
    def expected_concepts(self) -> list[str]:
        try:
            value = json.loads(self.expected_concepts_json or "[]")
            return value if isinstance(value, list) else []
        except (TypeError, json.JSONDecodeError):
            return []


class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    __table_args__ = (
        Index("ix_sessions_student_status", "student_id", "status"),
        Index("ix_sessions_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id", ondelete="RESTRICT"), nullable=False)
    title: Mapped[str] = mapped_column(String(180), nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False)
    mode: Mapped[str] = mapped_column(String(24), nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, nullable=False)
    time_limit_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    current_position: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="in_progress")
    text_status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    voice_status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    student: Mapped[Student] = relationship(back_populates="sessions")
    questions: Mapped[list["SessionQuestion"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="SessionQuestion.position"
    )
    responses: Mapped[list["InterviewResponse"]] = relationship(back_populates="session")
    reports: Mapped[list["AssessmentReport"]] = relationship(back_populates="session")


class SessionQuestion(Base):
    __tablename__ = "session_questions"
    __table_args__ = (
        UniqueConstraint("session_id", "position", name="uq_session_question_position"),
        UniqueConstraint("session_id", "question_id", name="uq_session_question_pair"),
        Index("ix_session_questions_status", "session_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False)
    question_id: Mapped[int] = mapped_column(ForeignKey("question_bank.id", ondelete="RESTRICT"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    marked_for_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    session: Mapped[InterviewSession] = relationship(back_populates="questions")
    question: Mapped[QuestionBank] = relationship(back_populates="session_questions")
    responses: Mapped[list["InterviewResponse"]] = relationship(back_populates="session_question")


class InterviewResponse(Base):
    __tablename__ = "interview_responses"
    __table_args__ = (
        UniqueConstraint("session_question_id", "response_mode", name="uq_response_question_mode"),
        Index("ix_responses_session_created", "session_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id", ondelete="RESTRICT"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False)
    session_question_id: Mapped[int] = mapped_column(ForeignKey("session_questions.id", ondelete="CASCADE"), nullable=False)
    question_id: Mapped[int] = mapped_column(ForeignKey("question_bank.id", ondelete="RESTRICT"), nullable=False)
    response_mode: Mapped[str] = mapped_column(String(12), nullable=False)
    answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    corrected_transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    audio_file_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    audio_retained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    student: Mapped[Student] = relationship(back_populates="responses")
    session: Mapped[InterviewSession] = relationship(back_populates="responses")
    session_question: Mapped[SessionQuestion] = relationship(back_populates="responses")
    question: Mapped[QuestionBank] = relationship(back_populates="responses")
    text_assessment: Mapped[Optional["TextAssessment"]] = relationship(back_populates="response", uselist=False)
    audio_assessment: Mapped[Optional["AudioAssessment"]] = relationship(back_populates="response", uselist=False)


class TextAssessment(Base):
    __tablename__ = "text_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("interview_responses.id", ondelete="CASCADE"), unique=True, nullable=False)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    question_alignment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    keyword_overlap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    structure_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feedback_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    response: Mapped[InterviewResponse] = relationship(back_populates="text_assessment")


class AudioAssessment(Base):
    __tablename__ = "audio_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    response_id: Mapped[str] = mapped_column(ForeignKey("interview_responses.id", ondelete="CASCADE"), unique=True, nullable=False)
    measurements_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    response: Mapped[InterviewResponse] = relationship(back_populates="audio_assessment")


class AssessmentReport(Base):
    __tablename__ = "assessment_reports"
    __table_args__ = (Index("ix_reports_session_created", "session_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False)
    student_id: Mapped[Optional[str]] = mapped_column(ForeignKey("students.id", ondelete="RESTRICT"), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    report_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    session: Mapped[InterviewSession] = relationship(back_populates="reports")


def _make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///"):
        db_path = database_url.removeprefix("sqlite:///")
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False, "timeout": 30} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


engine: Engine = _make_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def configure_database(database_url: Optional[str] = None) -> None:
    """Reconfigure the connection target; primarily useful for isolated tests."""
    global DATABASE_URL, engine, SessionLocal
    if engine:
        engine.dispose()
    DATABASE_URL = database_url or DEFAULT_DATABASE_URL
    engine = _make_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def db_session() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _insert_question(session, *, domain, topic, difficulty, question, reference, concepts, source_id, provenance):
    exists = session.scalar(select(QuestionBank.id).where(
        QuestionBank.domain == domain,
        QuestionBank.question_text == question.strip(),
    ))
    if exists:
        return False
    digest = hashlib.sha256(f"{domain}\0{question}".encode("utf-8")).hexdigest()[:12].upper()
    session.add(QuestionBank(
        question_code=f"Q-{_slug(domain)[:8].upper()}-{digest}",
        source_id=source_id,
        domain=domain,
        topic=(topic or "General")[:120],
        difficulty=difficulty if difficulty in {"Beginner", "Intermediate", "Advanced"} else "Intermediate",
        question_text=question.strip(),
        reference_answer=(reference or "").strip() or None,
        expected_concepts_json=json.dumps(concepts or [], ensure_ascii=False),
        provenance=provenance[:240],
    ))
    return True


def initialize_database(seed_questions: bool = True) -> None:
    """Create missing tables and add seed entries without deleting existing data."""
    Base.metadata.create_all(engine)
    # create_all preserves existing records but does not add columns to older
    # SQLite files. These additive changes keep legacy sessions and reports.
    additions = {
        "students": {
            "skills_json": "TEXT NOT NULL DEFAULT '[]'",
            "user_id": "VARCHAR(24) REFERENCES users(id)",
        },
        "interview_sessions": {
            "text_status": "VARCHAR(24) NOT NULL DEFAULT 'pending'",
            "voice_status": "VARCHAR(24) NOT NULL DEFAULT 'pending'",
        },
        "assessment_reports": {
            "student_id": "VARCHAR(24) REFERENCES students(id)",
            "version": "INTEGER NOT NULL DEFAULT 1",
        },
    }
    with engine.begin() as connection:
        inspector = inspect(connection)
        for table_name, columns in additions.items():
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for name, definition in columns.items():
                if name not in existing:
                    connection.execute(sql_text(f'ALTER TABLE "{table_name}" ADD COLUMN "{name}" {definition}'))
        connection.execute(sql_text(
            "UPDATE interview_sessions SET text_status = CASE "
            "WHEN mode = 'voice' THEN 'not_required' "
            "WHEN status = 'completed' THEN 'completed' ELSE text_status END"
        ))
        connection.execute(sql_text(
            "UPDATE interview_sessions SET voice_status = CASE "
            "WHEN mode = 'text' THEN 'not_required' "
            "WHEN status = 'completed' THEN 'completed' ELSE voice_status END"
        ))
    with engine.begin() as connection:
        # Preserve every legacy report while assigning deterministic versions
        # before enforcing one report version per assessment.
        report_rows = connection.execute(sql_text(
            "SELECT id, session_id FROM assessment_reports ORDER BY session_id, created_at, id"
        )).all()
        versions: dict[str, int] = {}
        for report_id, session_id in report_rows:
            versions[session_id] = versions.get(session_id, 0) + 1
            connection.execute(sql_text(
                "UPDATE assessment_reports SET version = :version WHERE id = :report_id"
            ), {"version": versions[session_id], "report_id": report_id})
        connection.execute(sql_text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_assessment_report_session_version "
            "ON assessment_reports(session_id, version)"
        ))
    if not seed_questions:
        return
    from src.interview_repository import CATEGORIZED_QUESTIONS

    aliases = {"DSA": "Data Structures and Algorithms", "OS": "Operating Systems"}
    with db_session() as session:
        for old_domain, questions in CATEGORIZED_QUESTIONS.items():
            domain = aliases.get(old_domain, old_domain)
            for item in questions:
                _insert_question(
                    session,
                    domain=domain,
                    topic="General",
                    difficulty="Intermediate",
                    question=item["question"],
                    reference=item.get("reference_answer"),
                    concepts=item.get("key_concepts", []),
                    source_id=f"existing-catalog:{domain}:{item.get('id', hashlib.sha1(item['question'].encode()).hexdigest()[:12])}",
                    provenance="Pre-existing InterviewAI project catalog; difficulty defaulted to Intermediate because the source omitted it.",
                )
        seed_path = PROJECT_DIR / "data" / "question_bank_seed.json"
        if seed_path.is_file():
            with seed_path.open(encoding="utf-8") as seed_file:
                authored = json.load(seed_file)
            for index, item in enumerate(authored):
                source_id = f"interviewai-authored-seed:{index + 1:04d}"
                _insert_question(
                    session,
                    domain=item["domain"],
                    topic=item.get("topic", "General"),
                    difficulty=item.get("difficulty", "Intermediate"),
                    question=item["question"],
                    reference=item.get("reference_answer"),
                    concepts=item.get("expected_concepts", []),
                    source_id=source_id,
                    provenance="Authored for the InterviewAI starter question bank.",
                )

