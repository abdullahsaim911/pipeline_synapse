"""Small programmatic wrapper around the repository's Alembic configuration."""

from pathlib import Path

from alembic import command
from alembic.config import Config

from api.database import DATABASE_URL, PROJECT_ROOT


def get_alembic_config(database_url: str = DATABASE_URL) -> Config:
    config = Config(str(Path(PROJECT_ROOT) / "alembic.ini"))
    config.set_main_option("script_location", str(Path(PROJECT_ROOT) / "migrations"))
    config.attributes["database_url"] = database_url
    return config


def upgrade_database(database_url: str = DATABASE_URL, revision: str = "head") -> None:
    """Upgrade a database without creating or dropping it implicitly."""
    command.upgrade(get_alembic_config(database_url), revision)
