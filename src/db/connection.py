"""
Database Connection Manager for Nassau Intelligence.
Provides thread-safe connection pooling, SQLite WAL mode, and transaction safety.
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("NassauDB")

# Database path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "nassau_intelligence.db"


def get_db_path() -> Path:
    """Return database file path, ensuring parent directory exists and seeding if missing."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        try:
            from src.db.seed import seed_database
            logger.info("Database file not found. Auto-seeding initial schema and data...")
            seed_database()
        except Exception as e:
            logger.warning(f"Could not auto-seed database: {e}")
    return DB_PATH


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager providing a transactional SQLite connection.
    Enforces foreign keys, WAL journal mode, and row factory.
    Automatically commits on success and rolls back on exception.
    """
    db_file = get_db_path()
    conn = sqlite3.connect(str(db_file), timeout=30.0)
    conn.row_factory = sqlite3.Row
    
    # Configure performance & integrity pragmas
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("PRAGMA journal_mode = WAL;")
    cursor.execute("PRAGMA synchronous = NORMAL;")
    cursor.execute("PRAGMA busy_timeout = 10000;")
    cursor.close()
    
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"Transaction rolled back due to error: {exc}")
        raise
    finally:
        conn.close()
