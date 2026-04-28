"""
Setup script for Synapse API.

Creates the database tables and verifies the installation.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api.database import engine, Base, SessionLocal
from api.models import Video, Intervention, Transcript, Explanation


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")


def verify_tables():
    """Verify tables exist."""
    print("\nVerifying database tables...")
    db = SessionLocal()

    try:
        # Check each table
        tables = [Video, Intervention, Transcript, Explanation]
        for table in tables:
            count = db.query(table).count()
            print(f"✓ {table.__tablename__}: {count} records")

        print("\n✓ All tables verified successfully")

    except Exception as e:
        print(f"✗ Error verifying tables: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Synapse API Setup")
    print("=" * 60)
    print()

    create_tables()
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
