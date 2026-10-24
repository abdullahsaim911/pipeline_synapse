"""Track every explanation mode requested for a shared processing job.

Revision ID: 20261024_0003
Revises: 20261024_0002
Create Date: 2026-10-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261024_0003"
down_revision: Union[str, None] = "20261024_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("processing_job_requested_modes"):
        op.create_table(
            "processing_job_requested_modes",
            sa.Column("job_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("output_mode", sa.String(length=32), nullable=False),
            sa.Column(
                "requested_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.CheckConstraint(
                "output_mode IN ('brief', 'explanatory', 'detailed')",
                name="ck_processing_job_requested_modes_output_mode",
            ),
            sa.ForeignKeyConstraint(
                ["job_id"],
                ["processing_jobs.id"],
                name="fk_processing_job_requested_modes_job_id_processing_jobs",
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint(
                "job_id",
                "output_mode",
                name="pk_processing_job_requested_modes",
            ),
        )

    # Preserve the mode held by every Task 5A row. The NOT EXISTS predicate
    # also makes this safe for databases where the table was created manually.
    op.execute(sa.text("""
        INSERT INTO processing_job_requested_modes
            (job_id, output_mode, requested_at)
        SELECT processing_jobs.id,
               processing_jobs.requested_output_mode,
               processing_jobs.created_at
        FROM processing_jobs
        WHERE NOT EXISTS (
            SELECT 1
            FROM processing_job_requested_modes
            WHERE processing_job_requested_modes.job_id = processing_jobs.id
              AND processing_job_requested_modes.output_mode =
                  processing_jobs.requested_output_mode
        )
    """))


def downgrade() -> None:
    raise RuntimeError(
        "The requested-modes migration is non-destructive and cannot be downgraded automatically."
    )
