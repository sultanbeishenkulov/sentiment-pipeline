"""
Sprint 1: Bluesky firehose -> Kafka producer.

Streams live posts from the AT Protocol firehose, filters to English-only,
and publishes them as JSON to Kafka. The JSON shape matches what the Sprint 3
consumer expects, so no consumer changes are needed when we wire things up.

Shape of a message on Kafka:
    {
        "id": "at://did:plc:xyz/app.bsky.feed.post/abc123",
        "subreddit": "bluesky",
        "title": null,
        "body": "the post text",
        "created_utc": 1728612345.678
    }

We keep field names matching what the existing Kafka topic / consumer expect
(`id`, `subreddit`, `title`, `body`, `created_utc`) so that the pipeline
stays generic — the source happens to be Bluesky today, could be something
else tomorrow. We'll rename to generic terms in Sprint 4 before demo.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

from confluent_kafka import Producer
from dotenv import load_dotenv

from bluesky_firehose import BlueskyPost, stream_posts

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("producer")


# ---- config ---------------------------------------------------------------

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reddit-posts")

LANG_FILTER = os.getenv("LANG_FILTER", "en").strip()
MAX_POSTS_PER_SEC = float(os.getenv("MAX_POSTS_PER_SEC", "0"))  # 0 = unlimited


# ---- graceful shutdown ----------------------------------------------------

_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    logger.info("Got signal %s — shutting down", signum)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---- helpers --------------------------------------------------------------

def _parse_created_utc(iso_ts: str) -> float:
    """Convert AT Proto ISO-8601 timestamp to a unix timestamp float.

    Falls back to now() if parsing fails — better to have a slightly wrong
    timestamp than to drop the post.
    """
    try:
        # AT Proto uses RFC 3339, e.g. "2024-10-11T10:30:00.123Z"
        ts = iso_ts.replace("Z", "+00:00") if iso_ts.endswith("Z") else iso_ts
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, AttributeError):
        return datetime.now(timezone.utc).timestamp()


def _to_kafka_message(post: BlueskyPost) -> dict:
    return {
        "id": post.uri,
        "subreddit": "bluesky",  # static source tag; consumer stores as `subreddit`
        "title": None,
        "body": post.text,
        "created_utc": _parse_created_utc(post.created_at),
    }


def _passes_lang_filter(post: BlueskyPost) -> bool:
    if not LANG_FILTER:
        return True
    # Posts often tag multiple langs ["en", "ja"]; we accept if ours is in there.
    # Posts with no langs at all are skipped — we can't trust them to be English.
    return LANG_FILTER in post.langs


def _delivery_report(err, msg) -> None:
    """Kafka produce callback. Only log failures — success is the common case."""
    if err is not None:
        logger.error("Kafka delivery failed: %s", err)


# ---- main loop ------------------------------------------------------------

def run() -> None:
    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "linger.ms": 50,              # batch messages for up to 50ms for throughput
        "compression.type": "lz4",    # cheap CPU cost, big bandwidth/disk win
    })
    logger.info("Producing to %s topic=%s", KAFKA_BOOTSTRAP, KAFKA_TOPIC)
    logger.info("Lang filter: %r  Rate limit: %s msg/s",
                LANG_FILTER or "(none)",
                MAX_POSTS_PER_SEC if MAX_POSTS_PER_SEC > 0 else "unlimited")

    produced = 0
    skipped_lang = 0
    last_log_t = time.monotonic()
    min_interval = 1.0 / MAX_POSTS_PER_SEC if MAX_POSTS_PER_SEC > 0 else 0.0
    last_send_t = 0.0

    try:
        for post in stream_posts():
            if _shutdown:
                break

            if not _passes_lang_filter(post):
                skipped_lang += 1
                continue

            # Rate limit (only engages if MAX_POSTS_PER_SEC > 0)
            if min_interval > 0:
                elapsed = time.monotonic() - last_send_t
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_send_t = time.monotonic()

            msg = _to_kafka_message(post)
            try:
                producer.produce(
                    KAFKA_TOPIC,
                    key=msg["id"].encode("utf-8"),
                    value=json.dumps(msg).encode("utf-8"),
                    on_delivery=_delivery_report,
                )
            except BufferError:
                # Producer queue is full; flush and retry once.
                logger.warning("Kafka producer queue full, flushing")
                producer.flush(5)
                producer.produce(
                    KAFKA_TOPIC,
                    key=msg["id"].encode("utf-8"),
                    value=json.dumps(msg).encode("utf-8"),
                    on_delivery=_delivery_report,
                )

            producer.poll(0)  # serve delivery callbacks
            produced += 1

            # Periodic stats line so you can watch it breathing
            now = time.monotonic()
            if now - last_log_t >= 10:
                rate = produced / max(now - last_log_t, 1e-6)
                logger.info(
                    "produced=%d (rate=%.1f/s) skipped_lang=%d",
                    produced, rate, skipped_lang,
                )
                produced = 0
                skipped_lang = 0
                last_log_t = now

    finally:
        logger.info("Flushing Kafka producer")
        producer.flush(10)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(0)
