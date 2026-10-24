"""Database configuration shared by the API and migration tooling."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import MetaData, create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import declarative_base, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_SQLITE_PATH = (PROJECT_ROOT / "synapse.db").resolve()
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"


def normalize_database_url(database_url: str) -> str:
    """Anchor relative SQLite files at the repository; leave server URLs intact."""
    url = make_url(database_url)
    if (
        url.get_backend_name() == "sqlite"
        and url.database
        and url.database != ":memory:"
        and not Path(url.database).is_absolute()
    ):
        url = url.set(database=str((PROJECT_ROOT / url.database).resolve().as_posix()))
    return url.render_as_string(hide_password=False)


DATABASE_URL = normalize_database_url(os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
SQLALCHEMY_DATABASE_URL = DATABASE_URL

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def create_database_engine(database_url: str = DATABASE_URL) -> Engine:
    """Create an engine with backend-specific connection safety settings."""
    url = make_url(database_url)
    kwargs = {"pool_pre_ping": True}
    if url.get_backend_name() == "sqlite":
        kwargs["connect_args"] = {"check_same_thread": False}

    database_engine = create_engine(database_url, **kwargs)
    if url.get_backend_name() == "sqlite":
        @event.listens_for(database_engine, "connect")
        def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return database_engine


engine = create_database_engine()
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)
Base = declarative_base(metadata=MetaData(naming_convention=NAMING_CONVENTION))


def get_db():
    """Yield one request-scoped SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
