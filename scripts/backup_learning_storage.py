"""Back up SQLite and managed media together while the local API is stopped.

Run: python -m scripts.backup_learning_storage
Verify: python -m scripts.backup_learning_storage --verify <backup-directory>
Restore with the API stopped: verify, preserve current files, then restore BOTH
synapse.db and data/ from the same backup directory. Never mix backup versions.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sqlite3

from sqlalchemy.engine import make_url
from api.database import DATABASE_URL, PROJECT_ROOT
from api.personal_storage import data_root


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_backup(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for key, expected in manifest["files"].items():
        target = (root / key).resolve()
        if not target.is_relative_to(root) or not target.is_file() or digest(target) != expected:
            raise RuntimeError(f"Backup verification failed: {key}")
    with closing(sqlite3.connect(root / "synapse.db")) as database:
        if database.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError("Backup database integrity check failed.")
    return manifest


def create_backup(destination, database_url=DATABASE_URL, media_root=None):
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise RuntimeError("This local backup command requires a file-backed SQLite database.")
    source_database = Path(url.database).resolve()
    if not source_database.is_file():
        raise RuntimeError("Database does not exist.")
    root = Path(destination).resolve()
    media = Path(media_root or data_root()).resolve()
    if root.is_relative_to(media):
        raise RuntimeError("The backup destination must be outside the media directory.")
    root.mkdir(parents=True, exist_ok=False)
    with closing(sqlite3.connect(source_database)) as source, closing(sqlite3.connect(root / "synapse.db")) as target:
        source.backup(target)
    if media.exists():
        for path in media.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"Resolve media symlinks before backing up: {path}")
            if path.is_file() and not path.name.startswith(".pending-"):
                target = root / "data" / path.relative_to(media)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    files = {path.relative_to(root).as_posix(): digest(path) for path in root.rglob("*") if path.is_file()}
    (root / "manifest.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
        "source_database": str(source_database), "files": files}, indent=2), encoding="utf-8")
    verify_backup(root)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify")
    arguments = parser.parse_args()
    if arguments.verify:
        verify_backup(arguments.verify)
        print("Database and media backup verified.")
        return
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", 8000)) == 0:
            raise RuntimeError("Stop the local API before creating a coordinated media backup.")
    name = datetime.now(timezone.utc).strftime("learning-storage-%Y%m%dT%H%M%S%fZ")
    print(create_backup(PROJECT_ROOT / "backups" / name))


if __name__ == "__main__":
    main()
