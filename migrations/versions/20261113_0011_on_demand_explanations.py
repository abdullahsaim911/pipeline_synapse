"""Private explanation requests and shared lecture-session interests."""
from alembic import op
import sqlalchemy as sa

revision = "20261113_0011"
down_revision = "20261113_0010"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("interventions", sa.Column("origin", sa.String(16), nullable=False, server_default="detected"))
    op.add_column("interventions", sa.Column("evidence_kind", sa.String(16), nullable=False, server_default="visual"))
    op.add_column("interventions", sa.Column("context_key", sa.String(64)))
    op.create_index("ix_interventions_context_key", "interventions", ["context_key"], unique=True)
    op.add_column("explanation_preparations", sa.Column("priority_at", sa.DateTime()))
    op.create_index("ix_explanation_preparations_priority_at", "explanation_preparations", ["priority_at"])
    op.create_table("lecture_session_interests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.Uuid(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("video_id", sa.String(11), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("account_digest", sa.String(64)),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()))
    op.create_index("ix_lecture_session_interests_user_id", "lecture_session_interests", ["user_id"])
    op.create_index("ix_lecture_session_interests_video_id", "lecture_session_interests", ["video_id"])
    op.create_index("ix_lecture_session_interests_expiry", "lecture_session_interests", ["state", "expires_at"])
    op.create_table("explanation_requests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.Uuid(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(36), sa.ForeignKey("lecture_session_interests.id", ondelete="SET NULL")),
        sa.Column("operation_id", sa.String(64), nullable=False),
        sa.Column("intent_hash", sa.String(64), nullable=False),
        sa.Column("video_id", sa.String(11), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("intervention_id", sa.String(), sa.ForeignKey("interventions.id", ondelete="SET NULL")),
        sa.Column("context_key", sa.String(64)),
        sa.Column("output_mode", sa.String(32), nullable=False),
        sa.Column("resolution_error", sa.String(64)),
        sa.Column("saved_explanation_id", sa.String(36), sa.ForeignKey("saved_explanations.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("heard_at", sa.DateTime()),
        sa.Column("partial_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("satisfaction", sa.String(8)),
        sa.Column("dismissed", sa.Boolean(), nullable=False, server_default="0"),
        sa.UniqueConstraint("user_id", "operation_id", name="uq_explanation_requests_operation"),
        sa.CheckConstraint("output_mode IN ('brief', 'explanatory', 'detailed')", name="ck_explanation_requests_mode"))
    op.create_index("ix_explanation_requests_owner_video", "explanation_requests", ["user_id", "video_id", "created_at", "id"])


def downgrade():
    if op.get_bind().execute(sa.text("SELECT COUNT(*) FROM explanation_requests")).scalar():
        raise RuntimeError("Export explanation requests and review progress before removing this schema.")
    op.drop_table("explanation_requests")
    op.drop_table("lecture_session_interests")
    with op.batch_alter_table("explanation_preparations") as table:
        table.drop_index("ix_explanation_preparations_priority_at")
        table.drop_column("priority_at")
    with op.batch_alter_table("interventions") as table:
        table.drop_index("ix_interventions_context_key")
        table.drop_column("context_key")
        table.drop_column("evidence_kind")
        table.drop_column("origin")
