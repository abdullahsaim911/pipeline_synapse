"""Establish the legacy baseline and local user foundation.

Revision ID: 20261024_0001
Revises: None
Create Date: 2026-10-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261024_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

LOCAL_USER_ID = "00000000-0000-4000-8000-000000000001"
LOCAL_PROFILE_KEY = "local-development"


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _ensure_index(name: str, table: str, columns: list[str], unique: bool = False) -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table)}
    if name not in indexes:
        op.create_index(name, table, columns, unique=unique)


def _create_shared_cache_tables() -> None:
    if not _has_table("videos"):
        op.create_table(
            "videos",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("title", sa.String(), nullable=False),
            sa.Column("duration_seconds", sa.Integer()),
            sa.Column("duration_formatted", sa.String()),
            sa.Column("total_keyframes", sa.Integer()),
            sa.Column("subject", sa.String()),
            sa.Column("source_url", sa.String(), nullable=False),
            sa.Column("date_processed", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("thumbnail_url", sa.String()),
            sa.PrimaryKeyConstraint("id", name="pk_videos"),
        )
    _ensure_index("ix_videos_id", "videos", ["id"])

    if not _has_table("transcripts"):
        op.create_table(
            "transcripts",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("video_id", sa.String(), nullable=False),
            sa.Column("start_time", sa.Float(), nullable=False),
            sa.Column("end_time", sa.Float(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], name="fk_transcripts_video_id_videos"),
            sa.PrimaryKeyConstraint("id", name="pk_transcripts"),
        )
    _ensure_index("ix_transcripts_id", "transcripts", ["id"])
    _ensure_index("ix_transcripts_video_id", "transcripts", ["video_id"])
    _ensure_index("ix_transcripts_start_time", "transcripts", ["start_time"])

    if not _has_table("interventions"):
        op.create_table(
            "interventions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("video_id", sa.String(), nullable=False),
            sa.Column("timestamp", sa.Float(), nullable=False),
            sa.Column("timestamp_formatted", sa.String()),
            sa.Column("frame_path", sa.String(), nullable=False),
            sa.Column("content_type", sa.String()),
            sa.Column("complexity_score", sa.Float()),
            sa.Column("clip_confidence", sa.Float()),
            sa.Column("trigger_reason", sa.String()),
            sa.Column("confidence", sa.String()),
            sa.Column("transcript_context", sa.Text()),
            sa.Column("is_bookmarked", sa.Boolean(), server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], name="fk_interventions_video_id_videos"),
            sa.PrimaryKeyConstraint("id", name="pk_interventions"),
        )
    _ensure_index("ix_interventions_id", "interventions", ["id"])
    _ensure_index("ix_interventions_video_id", "interventions", ["video_id"])
    _ensure_index("ix_interventions_timestamp", "interventions", ["timestamp"])
    _ensure_index("ix_interventions_is_bookmarked", "interventions", ["is_bookmarked"])

    if not _has_table("explanations"):
        op.create_table(
            "explanations",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("intervention_id", sa.String(), nullable=False),
            sa.Column("text_explanation", sa.Text(), nullable=False),
            sa.Column("audio_file_path", sa.String()),
            sa.Column("text_file_path", sa.String()),
            sa.Column("vlm_snapshot_path", sa.String()),
            sa.Column("output_mode", sa.String()),
            sa.Column("vlm_snapshot", sa.Text()),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["intervention_id"], ["interventions.id"], name="fk_explanations_intervention_id_interventions"),
            sa.PrimaryKeyConstraint("id", name="pk_explanations"),
        )
    _ensure_index("ix_explanations_id", "explanations", ["id"])
    _ensure_index("ix_explanations_intervention_id", "explanations", ["intervention_id"])

    if not _has_table("explanation_preparations"):
        op.create_table(
            "explanation_preparations",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("video_id", sa.String(), nullable=False),
            sa.Column("intervention_id", sa.String(), nullable=False),
            sa.Column("output_mode", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False, server_default="pending"),
            sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_code", sa.String()),
            sa.Column("lease_owner", sa.String()),
            sa.Column("lease_started_at", sa.DateTime()),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("ready_at", sa.DateTime()),
            sa.Column("failed_at", sa.DateTime()),
            sa.CheckConstraint("output_mode IN ('brief', 'explanatory', 'detailed')", name="ck_explanation_preparation_mode"),
            sa.CheckConstraint("status IN ('pending', 'generating', 'ready', 'failed')", name="ck_explanation_preparation_status"),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"], name="fk_explanation_preparations_video_id_videos"),
            sa.ForeignKeyConstraint(["intervention_id"], ["interventions.id"], name="fk_explanation_preparations_intervention_id_interventions"),
            sa.PrimaryKeyConstraint("id", name="pk_explanation_preparations"),
            sa.UniqueConstraint("video_id", "intervention_id", "output_mode", name="uq_explanation_preparation_key"),
        )
    for column in ("id", "video_id", "intervention_id", "output_mode", "status"):
        _ensure_index(f"ix_explanation_preparations_{column}", "explanation_preparations", [column])


def _create_user_tables() -> None:
    if not _has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("profile_key", sa.String(64), nullable=False),
            sa.Column("display_name", sa.String(120), nullable=False),
            sa.Column("is_local", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint("id", name="pk_users"),
            sa.UniqueConstraint("profile_key", name="uq_users_profile_key"),
        )

    if not _has_table("user_preferences"):
        op.create_table(
            "user_preferences",
            sa.Column("user_id", sa.Uuid(as_uuid=False), nullable=False),
            sa.Column("theme", sa.String(32), nullable=False, server_default="paper"),
            sa.Column("presentation_mode", sa.String(32), nullable=False, server_default="accessibility"),
            sa.Column("explanation_mode", sa.String(32), nullable=False, server_default="explanatory"),
            sa.Column("playback_rate", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("text_scale", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("high_contrast", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("reduced_motion", sa.Boolean()),
            sa.Column("locale", sa.String(16), nullable=False, server_default="en"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.CheckConstraint("theme IN ('paper', 'dark')", name="ck_user_preferences_theme"),
            sa.CheckConstraint("presentation_mode IN ('accessibility', 'study')", name="ck_user_preferences_presentation_mode"),
            sa.CheckConstraint("explanation_mode IN ('brief', 'explanatory', 'detailed')", name="ck_user_preferences_explanation_mode"),
            sa.CheckConstraint("playback_rate >= 0.5 AND playback_rate <= 3.0", name="ck_user_preferences_playback_rate"),
            sa.CheckConstraint("text_scale >= 0.75 AND text_scale <= 2.0", name="ck_user_preferences_text_scale"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_user_preferences_user_id_users", ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("user_id", name="pk_user_preferences"),
        )


def _seed_local_profile() -> None:
    bind = op.get_bind()
    users = sa.table(
        "users",
        sa.column("id", sa.Uuid(as_uuid=False)),
        sa.column("profile_key", sa.String()),
        sa.column("display_name", sa.String()),
        sa.column("is_local", sa.Boolean()),
    )
    preferences = sa.table(
        "user_preferences",
        sa.column("user_id", sa.Uuid(as_uuid=False)),
    )
    user_id = bind.execute(
        sa.select(users.c.id).where(users.c.profile_key == LOCAL_PROFILE_KEY)
    ).scalar_one_or_none()
    if user_id is None:
        bind.execute(users.insert().values(
            id=LOCAL_USER_ID,
            profile_key=LOCAL_PROFILE_KEY,
            display_name="Local Synapse User",
            is_local=True,
        ))
        user_id = LOCAL_USER_ID
    elif user_id != LOCAL_USER_ID:
        raise RuntimeError("The local-development profile key is assigned to an unexpected user ID")

    has_preferences = bind.execute(
        sa.select(preferences.c.user_id).where(preferences.c.user_id == LOCAL_USER_ID)
    ).scalar_one_or_none()
    if has_preferences is None:
        bind.execute(preferences.insert().values(user_id=LOCAL_USER_ID))


def upgrade() -> None:
    _create_shared_cache_tables()
    _create_user_tables()
    _seed_local_profile()


def downgrade() -> None:
    raise RuntimeError("The Synapse foundation migration is intentionally non-destructive")
