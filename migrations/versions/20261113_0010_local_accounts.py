"""Local credentials and revocable persistent sessions."""
from alembic import op
import sqlalchemy as sa

revision = "20261113_0010"
down_revision = "20261113_0009"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("account_credentials",
        sa.Column("user_id", sa.Uuid(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()))
    op.create_table("account_sessions",
        sa.Column("token_digest", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.Uuid(as_uuid=False), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime()))
    op.create_index("ix_account_sessions_user_id", "account_sessions", ["user_id"])


def downgrade():
    if op.get_bind().execute(sa.text("SELECT COUNT(*) FROM account_credentials")).scalar():
        raise RuntimeError("Export account data before removing local authentication.")
    op.drop_index("ix_account_sessions_user_id", table_name="account_sessions")
    op.drop_table("account_sessions")
    op.drop_table("account_credentials")
