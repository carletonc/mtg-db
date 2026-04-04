"""
Postgres connection helpers — thin wrapper around psycopg2.
"""

import psycopg2
import psycopg2.extras
from contextlib import contextmanager

from utils.config import DATABASE_URL


@contextmanager
def get_connection():
    """Yield a psycopg2 connection; commit on success, rollback on error."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(conn=None, *, dict_cursor: bool = False):
    """
    Yield a cursor from an existing connection, or open a fresh one.
    If a fresh connection is opened it auto-commits/rolls-back.
    """
    if conn is not None:
        cur_factory = psycopg2.extras.RealDictCursor if dict_cursor else None
        cur = conn.cursor(cursor_factory=cur_factory)
        try:
            yield cur
        finally:
            cur.close()
    else:
        with get_connection() as fresh_conn:
            cur_factory = psycopg2.extras.RealDictCursor if dict_cursor else None
            cur = fresh_conn.cursor(cursor_factory=cur_factory)
            try:
                yield cur
            finally:
                cur.close()
