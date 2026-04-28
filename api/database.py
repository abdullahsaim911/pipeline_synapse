"""
Database configuration for Synapse API.

Uses SQLite with SQLAlchemy for local storage.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLite database file path
SQLALCHEMY_DATABASE_URL = "sqlite:///./synapse.db"

# Create engine
# check_same_thread=False is needed for SQLite with FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency to get database session.

    Usage in FastAPI endpoint:
        db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
