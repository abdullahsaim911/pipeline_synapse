"""Add local student learning history tables.

Revision ID: 20261024_0004
Revises: 20261024_0003
Create Date: 2026-10-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261024_0004"
down_revision: Union[str, None] = "20261024_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if not inspector.has_table("lecture_activity"):
        op.create_table(
            "lecture_activity",
            sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("user_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("video_id", sa.String(), nullable=False),
            sa.Column("first_opened_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("last_opened_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("open_count", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("last_intervention_id", sa.String()),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_lecture_activity_user_id_users", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], name="fk_lecture_activity_video_id_videos", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["last_intervention_id"], ["interventions.id"], name="fk_lecture_activity_last_intervention_id_interventions", ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id", name="pk_lecture_activity"),
            sa.UniqueConstraint("user_id", "video_id", name="uq_lecture_activity_user_video"),
        )
        op.create_index("ix_lecture_activity_video_id", "lecture_activity", ["video_id"])
        op.create_index("ix_lecture_activity_user_last_opened", "lecture_activity", ["user_id", "last_opened_at"])

    if not inspector.has_table("library_items"):
        op.create_table(
            "library_items",
            sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("user_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("video_id", sa.String(), nullable=False),
            sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("preferred_mode", sa.String(32), nullable=False, server_default="explanatory"),
            sa.CheckConstraint("preferred_mode IN ('brief', 'explanatory', 'detailed')", name="ck_library_items_preferred_mode"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_library_items_user_id_users", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], name="fk_library_items_video_id_videos", ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id", name="pk_library_items"),
            sa.UniqueConstraint("user_id", "video_id", name="uq_library_items_user_video"),
        )
        op.create_index("ix_library_items_video_id", "library_items", ["video_id"])
        op.create_index("ix_library_items_user_last_accessed", "library_items", ["user_id", "last_accessed_at"])

    if not inspector.has_table("playback_progress"):
        op.create_table(
            "playback_progress",
            sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("user_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("video_id", sa.String(), nullable=False),
            sa.Column("position_seconds", sa.Float(), nullable=False, server_default="0"),
            sa.Column("duration_seconds", sa.Float()),
            sa.Column("completed", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.CheckConstraint("position_seconds >= 0", name="ck_playback_progress_nonnegative_position"),
            sa.CheckConstraint("duration_seconds IS NULL OR duration_seconds >= 0", name="ck_playback_progress_nonnegative_duration"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_playback_progress_user_id_users", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], name="fk_playback_progress_video_id_videos", ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id", name="pk_playback_progress"),
            sa.UniqueConstraint("user_id", "video_id", name="uq_playback_progress_user_video"),
        )
        op.create_index("ix_playback_progress_video_id", "playback_progress", ["video_id"])
        op.create_index("ix_playback_progress_user_updated", "playback_progress", ["user_id", "updated_at"])

    if not inspector.has_table("recent_explanations"):
        op.create_table(
            "recent_explanations",
            sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("user_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("video_id", sa.String(), nullable=False),
            sa.Column("intervention_id", sa.String(), nullable=False),
            sa.Column("explanation_id", sa.Integer()),
            sa.Column("output_mode", sa.String(32), nullable=False),
            sa.Column("viewed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.CheckConstraint("output_mode IN ('brief', 'explanatory', 'detailed')", name="ck_recent_explanations_output_mode"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_recent_explanations_user_id_users", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], name="fk_recent_explanations_video_id_videos", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["intervention_id"], ["interventions.id"], name="fk_recent_explanations_intervention_id_interventions", ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["explanation_id"], ["explanations.id"], name="fk_recent_explanations_explanation_id_explanations", ondelete="SET NULL"),
            sa.PrimaryKeyConstraint("id", name="pk_recent_explanations"),
            sa.UniqueConstraint("user_id", "explanation_id", name="uq_recent_explanations_user_explanation"),
        )
        op.create_index("ix_recent_explanations_user_viewed", "recent_explanations", ["user_id", "viewed_at"])
        op.create_index("ix_recent_explanations_user_video", "recent_explanations", ["user_id", "video_id"])


def downgrade() -> None:
    raise RuntimeError(
        "The student-history migration is non-destructive and cannot be downgraded automatically."
    )
