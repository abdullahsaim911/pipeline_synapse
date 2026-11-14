"""
Setup script for Synapse API.

Applies database migrations and verifies the installation.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.database import SessionLocal
from api.migrations import create_verified_sqlite_backup, upgrade_database
from api.models import (
    Explanation,
    ExplanationPreparation,
    Intervention,
    LectureActivity,
    LibraryItem,
    PlaybackProgress,
    ProcessingJob,
    ProcessingJobRequestedMode,
    RecentExplanation,
    Transcript,
    User,
    UserPreference,
    Video,
    SavedExplanation,
    VoiceNote,
    VoiceNoteExplanation,
    MediaDeletion,
    AccountCredential,
    AccountSession,
    ExplanationRequest,
    LectureSessionInterest,
)


def migrate_database():
    """Apply every pending non-destructive Alembic migration."""
    backup_path = create_verified_sqlite_backup()
    if backup_path is not None:
        print(f"Verified pre-migration backup: {backup_path}")
        print("Rollback: stop the API, preserve the failed database, then restore this backup in its place.")
    else:
        print("No existing SQLite database required a backup (fresh DB or non-SQLite backend).")
    print("Applying database migrations...")
    upgrade_database()
    print("[OK] Database migrations applied successfully")


def verify_tables():
    """Verify tables exist."""
    print("\nVerifying database tables...")
    db = SessionLocal()

    try:
        # Check each table
        tables = [
            SavedExplanation,
            VoiceNote,
            VoiceNoteExplanation,
            MediaDeletion,
            AccountCredential,
            AccountSession,
            ExplanationRequest,
            LectureSessionInterest,
            User,
            UserPreference,
            Video,
            Transcript,
            Intervention,
            Explanation,
            ExplanationPreparation,
            ProcessingJob,
            ProcessingJobRequestedMode,
            LectureActivity,
            LibraryItem,
            PlaybackProgress,
            RecentExplanation,
        ]
        for table in tables:
            count = db.query(table).count()
            print(f"[OK] {table.__tablename__}: {count} records")

        print("\n[OK] All tables verified successfully")

    except Exception as e:
        print(f"[ERROR] Error verifying tables: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Synapse API Setup")
    print("=" * 60)
    print()

    migrate_database()
    verify_tables()

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("To start the API server, run:")
    print("  python -m api.main")
    print()
    print("Or with uvicorn:")
    print("  uvicorn api.main:app --reload")
