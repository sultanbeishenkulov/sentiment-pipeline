"""
Sprint 3: Kafka consumer.

Flow per message batch:
    1. Poll Kafka until we have BATCH_SIZE messages or BATCH_TIMEOUT_SEC elapses
    2. Run DistilBERT inference on the whole batch (single forward pass)
    3. INSERT ... ON CONFLICT DO NOTHING into sentiment_posts
    4. Commit Kafka offsets *after* the DB write succeeds

The commit-after-write ordering gives us at-least-once delivery. On crash,
we'll reprocess some messages on restart, and the UNIQUE constraint on
reddit_id absorbs the duplicates.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Any

from confluent_kafka import Consumer, KafkaError, Message
from dotenv import load_dotenv

from db import PostgresWriter
from sentiment_model import SentimentModel

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("consumer")


# ---- config ---------------------------------------------------------------

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reddit-posts")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "sentiment-consumer")

POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://sentiment_user:sentiment_pass@localhost:5432/sentiment_db",
)

MODEL_PATH = os.getenv("MODEL_PATH", "../model/artifacts")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
BATCH_TIMEOUT_SEC = float(os.getenv("BATCH_TIMEOUT_SEC", "2.0"))
POLL_TIMEOUT_SEC = float(os.getenv("POLL_TIMEOUT_SEC", "1.0"))


# ---- graceful shutdown ----------------------------------------------------

_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    logger.info("Got signal %s — finishing current batch then exiting", signum)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---- message parsing ------------------------------------------------------

def parse_message(raw: bytes) -> dict[str, Any] | None:
    """Parse a Kafka message into a row dict (minus sentiment fields).

    Expected producer payload from Sprint 2:
        {
            "id": "abc123",                # reddit_id
            "subreddit": "technology",
            "title": "...",                # may be None/empty for comments
            "body": "..." or "selftext": "...",
            "created_utc": 1234567890.0
        }
    """
    try:
        msg = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning("Skipping malformed message: %s", e)
        return None

    reddit_id = msg.get("id") or msg.get("reddit_id")
    subreddit = msg.get("subreddit")
    created_utc = msg.get("created_utc")
    if not reddit_id or not subreddit or created_utc is None:
        logger.warning("Skipping message missing required fields: %s", msg)
        return None

    title = msg.get("title") or None
    body = msg.get("body") or msg.get("selftext") or None

    # Need something to classify
    if not title and not body:
        logger.debug("Skipping empty post %s", reddit_id)
        return None

    return {
        "reddit_id": reddit_id,
        "subreddit": subreddit,
        "title": title,
        "body": body,
        "created_utc": float(created_utc),
    }


def text_for_inference(row: dict) -> str:
    """What we actually feed the model — title + body, space-joined."""
    parts = [row.get("title") or "", row.get("body") or ""]
    return " ".join(p for p in parts if p).strip()


# ---- main loop ------------------------------------------------------------

def process_batch(
    model: SentimentModel,
    writer: PostgresWriter,
    consumer: Consumer,
    batch: list[tuple[Message, dict]],
) -> None:
    """Score a batch, write to DB, commit Kafka offsets."""
    if not batch:
        return

    rows = [row for _, row in batch]
    texts = [text_for_inference(r) for r in rows]

    t0 = time.monotonic()
    results = model.predict(texts)
    infer_ms = (time.monotonic() - t0) * 1000

    for row, res in zip(rows, results):
        row["sentiment_label"] = res.label
        row["sentiment_score"] = res.score

    writer.write_batch(rows)

    # Only commit after the DB write succeeds.
    kafka_msgs = [m for m, _ in batch]
    consumer.commit(message=kafka_msgs[-1], asynchronous=False)

    logger.info(
        "Processed batch of %d in %.0f ms inference (%.1f msg/s)",
        len(batch), infer_ms, len(batch) / max(infer_ms / 1000, 1e-6),
    )


def run() -> None:
    logger.info("Loading sentiment model…")
    model = SentimentModel(model_path=MODEL_PATH)
    logger.info("Model ready on device=%s", model.device)

    writer = PostgresWriter(POSTGRES_DSN)
    writer.connect()

    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": KAFKA_GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([KAFKA_TOPIC])
    logger.info("Subscribed topic=%s group=%s", KAFKA_TOPIC, KAFKA_GROUP_ID)

    batch: list[tuple[Message, dict]] = []
    batch_started_at = time.monotonic()

    try:
        while not _shutdown:
            msg = consumer.poll(timeout=POLL_TIMEOUT_SEC)

            if msg is not None:
                if msg.error():
                    # PARTITION_EOF is informational when the topic is drained
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug("Reached end of partition")
                    else:
                        logger.error("Kafka error: %s", msg.error())
                else:
                    row = parse_message(msg.value())
                    if row is not None:
                        batch.append((msg, row))
                    else:
                        # Parse failure is a poison message — commit past it so
                        # we don't loop forever. DB write never happened, so
                        # there's nothing to roll back.
                        consumer.commit(message=msg, asynchronous=False)

            # Flush batch on size OR timeout
            age = time.monotonic() - batch_started_at
            if batch and (len(batch) >= BATCH_SIZE or age >= BATCH_TIMEOUT_SEC):
                process_batch(model, writer, consumer, batch)
                batch = []
                batch_started_at = time.monotonic()

        # Drain remaining batch on shutdown
        if batch:
            logger.info("Flushing final batch of %d before exit", len(batch))
            process_batch(model, writer, consumer, batch)

    finally:
        logger.info("Closing consumer and DB connection")
        consumer.close()
        writer.close()


if __name__ == "__main__":
    run()
