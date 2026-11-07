"""Programmatic migration helpers and the API startup schema preflight."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine, make_url

from api.database import Base, DATABASE_URL, PROJECT_ROOT


class DatabaseSchemaError(RuntimeError):
    """The configured database cannot safely serve the current application."""


class DatabaseBackupError(RuntimeError):
    """A pre-migration database backup could not be created or verified."""


def get_alembic_config(database_url: str = DATABASE_URL) -> Config:
    config = Config(str(Path(PROJECT_ROOT) / "alembic.ini"))
    config.set_main_option("script_location", str(Path(PROJECT_ROOT) / "migrations"))
    config.attributes["database_url"] = database_url
    return config


def upgrade_database(database_url: str = DATABASE_URL, revision: str = "head") -> None:
    """Upgrade a database without creating or dropping it implicitly."""
    command.upgrade(get_alembic_config(database_url), revision)


def create_verified_sqlite_backup(database_url: str = DATABASE_URL) -> Path | None:
    """Create and integrity-check a point-in-time SQLite backup before migration."""
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return None

    source_path = Path(url.database).resolve()
    if not source_path.exists():
        return None

    backup_dir = source_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backup_dir / f"{source_path.stem}-pre-migration-{timestamp}{source_path.suffix}.backup"
    try:
        with closing(sqlite3.connect(source_path)) as source:
            with closing(sqlite3.connect(backup_path)) as destination:
                source.backup(destination)
                destination.commit()
                integrity = destination.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise DatabaseBackupError(
                f"SQLite backup integrity check failed for {backup_path}"
            )
    except Exception as exc:
        if backup_path.exists():
            backup_path.unlink()
        if isinstance(exc, DatabaseBackupError):
            raise
        raise DatabaseBackupError("Could not create a verified SQLite migration backup") from exc
    return backup_path


def _format_revisions(revisions: Iterable[str]) -> str:
    values = sorted(revisions)
    return ", ".join(values) if values else "unversioned"


def require_current_schema(database_engine: Engine) -> None:
    """Fail before workers start unless revisions, tables, and columns are current."""
    config = get_alembic_config(database_engine.url.render_as_string(hide_password=False))
    expected_heads = set(ScriptDirectory.from_config(config).get_heads())
    with database_engine.connect() as connection:
        current_heads = set(MigrationContext.configure(connection).get_current_heads())

    if current_heads != expected_heads:
        raise DatabaseSchemaError(
            "Database schema is not current "
            f"(current: {_format_revisions(current_heads)}; "
            f"required: {_format_revisions(expected_heads)}). "
            "Back up the database, then run `python -m api.setup` and restart the API."
        )

    # Importing models registers all mapped tables with Base.metadata. This is
    # read-only and also detects a database stamped at head without its schema.
    from api import models as _models  # noqa: F401

    database_inspector = inspect(database_engine)
    missing_tables: list[str] = []
    missing_columns: list[str] = []
    actual_tables = set(database_inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in actual_tables:
            missing_tables.append(table.name)
            continue
        actual_columns = {
            column["name"] for column in database_inspector.get_columns(table.name)
        }
        for column in table.columns:
            if column.name not in actual_columns:
                missing_columns.append(f"{table.name}.{column.name}")

    if missing_tables or missing_columns:
        details = []
        if missing_tables:
            details.append("missing tables: " + ", ".join(sorted(missing_tables)))
        if missing_columns:
            details.append("missing columns: " + ", ".join(sorted(missing_columns)))
        raise DatabaseSchemaError(
            "Database revision is current but its structure is incomplete ("
            + "; ".join(details)
            + "). Restore a verified backup or repair the migration state before restarting."
        )
