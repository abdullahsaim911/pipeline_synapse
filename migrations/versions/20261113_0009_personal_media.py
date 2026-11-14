"""Preserved explanations and personal voice recordings.

Revision ID: 20261113_0009
Revises: 20261110_0008
"""
from alembic import op
import sqlalchemy as sa
import json

revision = "20261113_0009"
down_revision = "20261110_0008"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("library_items") as batch:
        batch.drop_constraint("fk_library_items_video_id_videos", type_="foreignkey")
        batch.add_column(sa.Column("lecture_snapshot", sa.Text()))
    # Preserve existing memberships and lecture labels before decoupling cache
    # lifetime. SQLAlchemy bindings keep this portable across SQLite/PostgreSQL.
    connection = op.get_bind()
    for row in connection.execute(sa.text("SELECT li.id, v.id AS video_id, v.title, v.source_url, v.thumbnail_url FROM library_items li JOIN videos v ON v.id = li.video_id")).mappings():
        connection.execute(sa.text("UPDATE library_items SET lecture_snapshot = :snapshot WHERE id = :id"), {
            "id": row["id"], "snapshot": json.dumps({"videoId": row["video_id"], "videoTitle": row["title"], "videoUrl": row["source_url"], "videoThumbnail": row["thumbnail_url"]})})
    op.create_table("saved_explanations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.Uuid(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("intervention_id", sa.String(), nullable=False),
        sa.Column("output_mode", sa.String(32), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("bookmarked_at", sa.DateTime()),
        sa.Column("viewed_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "fingerprint", name="uq_saved_explanations_owner_content"))
    op.create_index("ix_saved_explanations_owner_video", "saved_explanations", ["user_id", "video_id"])
    op.create_table("voice_notes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.Uuid(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("operation_id", sa.String(128), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("video_id", sa.String(), nullable=False),
        sa.Column("lecture_seconds", sa.Float(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("media_key", sa.String(), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "operation_id", name="uq_voice_notes_owner_operation"),
        sa.CheckConstraint("lecture_seconds >= 0 AND lecture_seconds <= 172800", name="ck_voice_notes_time"),
        sa.CheckConstraint("duration_ms > 0 AND duration_ms <= 300250", name="ck_voice_notes_duration"))
    op.create_index("ix_voice_notes_owner_video", "voice_notes", ["user_id", "video_id"])
    op.create_table("voice_note_explanations",
        sa.Column("note_id", sa.String(36), sa.ForeignKey("voice_notes.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("explanation_id", sa.String(36), sa.ForeignKey("saved_explanations.id"), primary_key=True))
    op.create_table("media_deletions",
        sa.Column("media_key", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()))


def downgrade():
    raise RuntimeError("Export personal recordings and saved explanations before a destructive downgrade.")
