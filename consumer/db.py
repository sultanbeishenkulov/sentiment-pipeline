"""
Postgres writer for scored posts.

Matches the schema in db/init/01_schema.sql:
    reddit_id (UNIQUE), subreddit, title, body, created_utc,
    sentiment_label, sentiment_score

ON CONFLICT (reddit_id) DO NOTHING makes the consumer safely replayable —
if we crash between DB write and Kafka commit, reprocessing the same offsets
is a no-op instead of polluting the table with duplicates.
"""
from __future__ import annotations

import logging
from typing import Iterable

import psycopg

logger = logging.getLogger(__name__)


INSERT_SQL = """
    INSERT INTO sentiment_posts
        (reddit_id, subreddit, title, body, created_utc,
         sentiment_label, sentiment_score)
    VALUES
        (%(reddit_id)s, %(subreddit)s, %(title)s, %(body)s,
         to_timestamp(%(created_utc)s),
         %(sentiment_label)s, %(sentiment_score)s)
    ON CONFLICT (reddit_id) DO NOTHING
"""


class PostgresWriter:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None

    def connect(self) -> None:
        self._conn = psycopg.connect(self.dsn, autocommit=False)
        logger.info("Connected to Postgres")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def write_batch(self, rows: Iterable[dict]) -> int:
        """Insert a batch. Returns number of rows actually inserted
        (duplicates skipped by ON CONFLICT are not counted)."""
        rows = list(rows)
        if not rows:
            return 0

        assert self._conn is not None, "call connect() first"
        try:
            with self._conn.cursor() as cur:
                cur.executemany(INSERT_SQL, rows)
                inserted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            self._conn.commit()
            skipped = len(rows) - inserted
            logger.info(
                "DB batch: %d inserted, %d skipped (dupes) of %d",
                inserted, skipped, len(rows),
            )
            return inserted
        except Exception:
            self._conn.rollback()
            logger.exception("Batch insert failed; rolled back")
            raise
