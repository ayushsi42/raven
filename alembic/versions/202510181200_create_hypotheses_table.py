"""Create hypotheses table

Revision ID: 202510181200
Revises: None
Create Date: 2025-10-18
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "202510181200"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hypotheses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("workflow_id", sa.String(length=255), nullable=False),
        sa.Column("workflow_run_id", sa.String(length=255), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("validation", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workflow_id", name="uq_hypotheses_workflow_id"),
    )
    op.create_index(op.f("ix_hypotheses_user_id"), "hypotheses", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_hypotheses_user_id"), table_name="hypotheses")
    op.drop_table("hypotheses")
