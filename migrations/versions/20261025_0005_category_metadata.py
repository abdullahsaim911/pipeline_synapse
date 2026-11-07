"""Add explicit visual-category and confidence metadata.

Existing ``interventions.clip_confidence`` values are retained verbatim.  In
the pre-migration pipeline that column contained a complexity-derived value,
so existing rows are marked ``legacy_complexity`` rather than being promoted
to classifier confidence.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261025_0005"
down_revision: Union[str, None] = "20261024_0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("interventions"):
        return

    existing = {column["name"] for column in inspector.get_columns("interventions")}
    additions = (
        ("classification_confidence", sa.Float()),
        ("confidence_semantics", sa.String(length=32), {"nullable": False, "server_default": "legacy_complexity"}),
        ("raw_category_labels_json", sa.Text()),
        ("canonical_categories_json", sa.Text()),
        ("category_mapping_notes_json", sa.Text()),
    )
    for item in additions:
        name, column_type, *options = item
        if name not in existing:
            op.add_column("interventions", sa.Column(name, column_type, **(options[0] if options else {})))

    # Do not rewrite or fabricate historical confidence.  The explicit marker
    # makes the legacy meaning visible to application code and future audits.
    op.execute(
        sa.text(
            "UPDATE interventions SET confidence_semantics = 'legacy_complexity' "
            "WHERE confidence_semantics IS NULL"
        )
    )


def downgrade() -> None:
    raise RuntimeError(
        "The category metadata migration is additive and cannot be downgraded automatically."
    )

