"""
Postgres connection pool.

A pool maintains a small set of long-lived connections and hands them out
on request, recycling them when done. Two big wins over per-request connects:
  - Skip TCP/TLS/auth handshake on every query (~50ms saved)
  - Bound the number of concurrent connections so we don't overwhelm Postgres

The pool is initialized on FastAPI startup and closed on shutdown via the
lifespan handler in main.py. Endpoints get connections via the get_db()
dependency, which auto-returns the connection to the pool when the request ends.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)


_pool: ConnectionPool | None = None


def init_pool() -> None:
    """Create the connection pool. Called once on app startup."""
    global _pool
    dsn = os.getenv(
        "POSTGRES_DSN",
        "postgresql://sentiment_user:sentiment_pass@localhost:5432/sentiment_db",
    )
    min_size = int(os.getenv("DB_POOL_MIN", "2"))
    max_size = int(os.getenv("DB_POOL_MAX", "10"))

    _pool = ConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        # Read-only API: no need for transactions across requests
        kwargs={"autocommit": True},
        open=True,
    )
    logger.info("DB pool ready (min=%d, max=%d)", min_size, max_size)


def close_pool() -> None:
    """Close the pool. Called once on app shutdown."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("DB pool closed")


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """Yield a connection from the pool; auto-return on exit."""
    if _pool is None:
        raise RuntimeError("DB pool not initialized — call init_pool() first")
    with _pool.connection() as conn:
        yield conn
