"""Add durable shared video processing jobs.

Revision ID: 20261024_0002
Revises: 20261024_0001
Create Date: 2026-10-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261024_0002"
down_revision: Union[str, None] = "20261024_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("processing_jobs"):
        return

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("source_url", sa.String(), nullable=False),
        sa.Column(
            "requested_output_mode",
            sa.String(length=32),
            nullable=False,
            server_default="explanatory",
        ),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(length=64)),
        sa.Column("error_message", sa.String(length=255)),
        sa.Column("lease_owner", sa.String(length=64)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_processing_jobs_status",
        ),
        sa.CheckConstraint(
            "stage IN ('queued', 'processing', 'persisting', "
            "'preparing_explanations', 'complete', 'failed')",
            name="ck_processing_jobs_stage",
        ),
        sa.CheckConstraint(
            "requested_output_mode IN ('brief', 'explanatory', 'detailed')",
            name="ck_processing_jobs_output_mode",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_processing_jobs"),
        sa.UniqueConstraint("video_id", name="uq_processing_jobs_video_id"),
    )
    op.create_index("ix_processing_jobs_status", "processing_jobs", ["status"])
    op.create_index("ix_processing_jobs_stage", "processing_jobs", ["stage"])


def downgrade() -> None:
    raise RuntimeError(
        "The processing-jobs migration is non-destructive and cannot be downgraded automatically."
    )
