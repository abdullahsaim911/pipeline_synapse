"""Migration safety tests against clean and copied legacy SQLite databases."""

import shutil
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from alembic import command
from sqlalchemy import text

from api.database import PROJECT_ROOT, create_database_engine, normalize_database_url
from api.migrations import get_alembic_config, upgrade_database
from api.models import LOCAL_DEVELOPMENT_USER_ID, LOCAL_DEVELOPMENT_PROFILE_KEY


ROOT = Path(__file__).resolve().parents[2]
LEGACY_TABLES = (
    "videos",
    "transcripts",
    "interventions",
    "explanations",
    "explanation_preparations",
)
HISTORY_TABLES = (
    "lecture_activity",
    "library_items",
    "playback_progress",
    "recent_explanations",
)


def sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def snapshot_records(path: Path) -> dict[str, tuple[int, tuple]]:
    with closing(sqlite3.connect(path)) as connection:
        snapshot = {}
        for table in LEGACY_TABLES:
            columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            key = "id" if "id" in columns else next(iter(columns))
            values = tuple(row[0] for row in connection.execute(
                f"SELECT {key} FROM {table} ORDER BY {key}"
            ))
            snapshot[table] = (len(values), values)
        return snapshot


class DatabaseMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.temp_path = Path(self.temp.name)

    def test_clean_database_upgrade_is_repeatable_and_seeds_local_profile(self):
        database_path = self.temp_path / "clean.sqlite"
        url = sqlite_url(database_path)

        upgrade_database(url)
        upgrade_database(url)

        with closing(sqlite3.connect(database_path)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertTrue(set(LEGACY_TABLES).issubset(tables))
            self.assertIn("users", tables)
            self.assertIn("user_preferences", tables)
            self.assertIn("processing_jobs", tables)
            self.assertIn("processing_job_requested_modes", tables)
            self.assertTrue(set(HISTORY_TABLES).issubset(tables))
            self.assertEqual(
                connection.execute("SELECT version_num FROM alembic_version").fetchone()[0],
                "20261024_0004",
            )
            self.assertEqual(
                connection.execute(
                    "SELECT id, profile_key FROM users WHERE profile_key = ?",
                    (LOCAL_DEVELOPMENT_PROFILE_KEY,),
                ).fetchone(),
                (LOCAL_DEVELOPMENT_USER_ID.replace("-", ""), LOCAL_DEVELOPMENT_PROFILE_KEY),
            )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM user_preferences").fetchone()[0], 1)

    def test_upgrade_of_existing_database_copy_preserves_every_existing_record(self):
        source = ROOT / "synapse.db"
        self.assertTrue(source.exists(), "The migration preservation test requires synapse.db")
        database_path = self.temp_path / "synapse-backup.sqlite"
        shutil.copy2(source, database_path)
        before = snapshot_records(database_path)

        # Verify the Task 4 foundation independently before applying Task 5A.
        upgrade_database(sqlite_url(database_path), "20261024_0001")
        after_foundation = snapshot_records(database_path)
        upgrade_database(sqlite_url(database_path), "20261024_0002")
        after_first = snapshot_records(database_path)
        upgrade_database(sqlite_url(database_path), "20261024_0003")
        after_second = snapshot_records(database_path)
        upgrade_database(sqlite_url(database_path), "20261024_0004")
        after_history = snapshot_records(database_path)
        upgrade_database(sqlite_url(database_path), "20261024_0004")
        after_repeated_history = snapshot_records(database_path)

        self.assertEqual(after_foundation, before)
        self.assertEqual(after_first, before)
        self.assertEqual(after_second, before)
        self.assertEqual(after_history, before)
        self.assertEqual(after_repeated_history, before)
        with closing(sqlite3.connect(database_path)) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM user_preferences").fetchone()[0], 1)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM processing_jobs").fetchone()[0], 0)
            for table in HISTORY_TABLES:
                self.assertEqual(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_requested_modes_migration_backfills_existing_processing_jobs(self):
        database_path = self.temp_path / "requested-modes.sqlite"
        url = sqlite_url(database_path)
        upgrade_database(url, "20261024_0002")
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                """
                INSERT INTO processing_jobs (
                    id, video_id, source_url, requested_output_mode,
                    status, stage, attempt_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "33333333333343338333333333333333",
                    "Backfill123",
                    "https://www.youtube.com/watch?v=Backfill123",
                    "detailed",
                    "succeeded",
                    "complete",
                    1,
                ),
            )
            connection.commit()

        upgrade_database(url, "20261024_0003")
        upgrade_database(url, "20261024_0003")

        with closing(sqlite3.connect(database_path)) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT output_mode FROM processing_job_requested_modes"
                ).fetchall(),
                [("detailed",)],
            )

    def test_sqlite_connections_enforce_foreign_keys_and_user_delete_cascades(self):
        database_path = self.temp_path / "foreign-keys.sqlite"
        url = sqlite_url(database_path)
        upgrade_database(url)
        engine = create_database_engine(url)
        self.addCleanup(engine.dispose)

        with engine.begin() as connection:
            self.assertEqual(connection.execute(text("PRAGMA foreign_keys")).scalar_one(), 1)
            connection.execute(
                text("DELETE FROM users WHERE profile_key = :profile_key"),
                {"profile_key": LOCAL_DEVELOPMENT_PROFILE_KEY},
            )
            self.assertEqual(connection.execute(text("SELECT COUNT(*) FROM user_preferences")).scalar_one(), 0)

    def test_relative_sqlite_url_is_anchored_to_the_repository(self):
        expected = f"sqlite:///{(PROJECT_ROOT / 'custom.sqlite').as_posix()}"
        self.assertEqual(normalize_database_url("sqlite:///./custom.sqlite"), expected)

    def test_clean_migration_matches_sqlalchemy_metadata(self):
        database_path = self.temp_path / "metadata.sqlite"
        url = sqlite_url(database_path)
        upgrade_database(url)

        command.check(get_alembic_config(url))


if __name__ == "__main__":
    unittest.main()
