"""Add durable timing for deferred VLM provider recovery.

Revision ID: 20261110_0008
Revises: 20261105_0007
Create Date: 2026-11-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261110_0008"
down_revision: Union[str, None] = "20261105_0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("explanation_preparations"):
        return
    existing = {
        column["name"]
        for column in inspector.get_columns("explanation_preparations")
    }
    if "provider_wait_started_at" not in existing:
        op.add_column(
            "explanation_preparations",
            sa.Column("provider_wait_started_at", sa.DateTime(), nullable=True),
        )
    if "deferred_until" not in existing:
        op.add_column(
            "explanation_preparations",
            sa.Column("deferred_until", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_explanation_preparations_deferred_until",
            "explanation_preparations",
            ["deferred_until"],
            unique=False,
        )


def downgrade() -> None:
    raise RuntimeError(
        "Deferred VLM recovery is additive and cannot be downgraded automatically."
    )
