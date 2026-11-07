"""Add internal pipeline stage and failure reason to explanation preparation.

Revision ID: 20261031_0006
Revises: 20261025_0005
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261031_0006"
down_revision: Union[str, None] = "20261025_0005"
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
    for name in ("stage", "failure_reason"):
        if name not in existing:
            op.add_column(
                "explanation_preparations", sa.Column(name, sa.String(), nullable=True)
            )


def downgrade() -> None:
    raise RuntimeError(
        "The preparation stage migration is additive and cannot be downgraded automatically."
    )
