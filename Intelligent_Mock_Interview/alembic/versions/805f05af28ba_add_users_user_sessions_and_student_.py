"""Add users, user_sessions, and student user_id

Revision ID: 805f05af28ba
Revises: 
Create Date: 2026-11-10 17:06:26.874380

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '805f05af28ba'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Use batch_alter_table for SQLite compatibility
    with op.batch_alter_table('assessment_reports', schema=None) as batch_op:
        try:
            batch_op.drop_index(op.f('uq_assessment_report_session_version'))
        except Exception:
            pass

    with op.batch_alter_table('students', schema=None) as batch_op:
        try:
            batch_op.create_index('ix_students_user_id', ['user_id'], unique=False)
        except Exception:
            pass
        try:
            batch_op.create_unique_constraint('uq_students_email', ['email'])
        except Exception:
            pass

    with op.batch_alter_table('users', schema=None) as batch_op:
        try:
            batch_op.create_index('ix_users_role', ['role'], unique=False)
        except Exception:
            pass


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('users', schema=None) as batch_op:
        try:
            batch_op.drop_index('ix_users_role')
        except Exception:
            pass

    with op.batch_alter_table('students', schema=None) as batch_op:
        try:
            batch_op.drop_constraint('uq_students_email', type_='unique')
        except Exception:
            pass
        try:
            batch_op.drop_index('ix_students_user_id')
        except Exception:
            pass

    with op.batch_alter_table('assessment_reports', schema=None) as batch_op:
        try:
            batch_op.create_index(op.f('uq_assessment_report_session_version'), ['session_id', 'version'], unique=1)
        except Exception:
            pass
