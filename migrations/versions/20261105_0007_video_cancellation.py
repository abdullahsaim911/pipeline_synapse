"""Add durable video cancellation and attempt fencing.

Revision ID: 20261105_0007
Revises: 20261031_0006
Create Date: 2026-11-05
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261105_0007"
down_revision: Union[str, None] = "20261031_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    active_count = op.get_bind().execute(sa.text(
        "SELECT COUNT(*) FROM processing_jobs p WHERE "
        "p.status IN ('queued', 'running', 'cancelling') OR EXISTS ("
        "SELECT 1 FROM explanation_preparations e WHERE e.video_id = p.video_id "
        "AND e.status IN ('pending', 'generating'))"
    )).scalar_one()
    if active_count > 1:
        raise RuntimeError(
            "Cannot enable single-video admission while multiple processing jobs are active"
        )
    processing_columns = {column["name"] for column in inspector.get_columns("processing_jobs")}
    with op.batch_alter_table("processing_jobs", recreate="always") as batch:
        if "generation_id" not in processing_columns:
            batch.add_column(sa.Column("generation_id", sa.String(length=64), nullable=True))
        if "active_slot" not in processing_columns:
            batch.add_column(sa.Column("active_slot", sa.Integer(), nullable=True))
        if "cancel_requested_at" not in processing_columns:
            batch.add_column(sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True))
        if "cancelled_at" not in processing_columns:
            batch.add_column(sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True))
        batch.drop_constraint("ck_processing_jobs_status", type_="check")
        batch.create_check_constraint(
            "ck_processing_jobs_status",
            "status IN ('queued', 'running', 'cancelling', 'cancelled', 'succeeded', 'failed')",
        )
        batch.drop_constraint("ck_processing_jobs_stage", type_="check")
        batch.create_check_constraint(
            "ck_processing_jobs_stage",
            "stage IN ('queued', 'processing', 'persisting', 'preparing_explanations', "
            "'cancelling', 'cancelled', 'complete', 'failed')",
        )
        batch.create_unique_constraint("uq_processing_jobs_active_slot", ["active_slot"])

    op.execute(sa.text(
        "UPDATE processing_jobs SET active_slot = 1 "
        "WHERE status IN ('queued', 'running', 'cancelling') OR EXISTS ("
        "SELECT 1 FROM explanation_preparations e "
        "WHERE e.video_id = processing_jobs.video_id "
        "AND e.status IN ('pending', 'generating'))"
    ))

    preparation_checks = {
        constraint["name"]
        for constraint in inspector.get_check_constraints("explanation_preparations")
        if constraint.get("name")
    }
    with op.batch_alter_table("explanation_preparations", recreate="always") as batch:
        if "ck_explanation_preparation_status" in preparation_checks:
            batch.drop_constraint("ck_explanation_preparation_status", type_="check")
        batch.create_check_constraint(
            "ck_explanation_preparation_status",
            "status IN ('pending', 'generating', 'cancelled', 'ready', 'failed')",
        )


def downgrade() -> None:
    raise RuntimeError(
        "The video-cancellation migration is non-destructive and cannot be downgraded automatically."
    )
