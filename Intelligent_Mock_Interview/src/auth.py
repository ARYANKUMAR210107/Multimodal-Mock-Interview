"""Authentication utilities: password hashing, session tokens, authorization."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import HTTPException, status
from sqlalchemy import select

from src.database import (
    InterviewSession,
    Student,
    User,
    UserSession,
    db_session,
    utcnow,
)


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_session_token() -> str:
    """Generate a secure random session token."""
    return secrets.token_urlsafe(32)


def authenticate_user(db, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password. Returns the User or None."""
    user = db.scalar(select(User).where(User.email == email.lower()))
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    if user.account_status != "active":
        return None
    return user


def create_user_session(db, user: User, ttl_hours: int = 24) -> UserSession:
    """Create a new session token for a user."""
    now = utcnow()
    expires_at = now + timedelta(hours=ttl_hours)
    session = UserSession(
        token=create_session_token(),
        user_id=user.id,
        expires_at=expires_at,
        created_at=now,
    )
    db.add(session)
    db.flush()
    return session


def get_current_user_from_token(db, token: str) -> Optional[User]:
    """Validate a session token and return the associated User."""
    now = utcnow()
    session = db.scalar(
        select(UserSession)
        .where(UserSession.token == token)
        .where(UserSession.expires_at > now)
    )
    if not session:
        return None
    user = db.scalar(select(User).where(User.id == session.user_id))
    if not user or user.account_status != "active":
        return None
    return user


def require_role(user: User, allowed_roles: set[str]) -> None:
    """Raise 403 if the user's role is not in allowed_roles."""
    if user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required role: {', '.join(sorted(allowed_roles))}.",
        )
