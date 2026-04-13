"""
Bluesky firehose client.

Connects to the AT Protocol firehose (a websocket that streams every public
event on the Bluesky network), decodes the CBOR/CAR payloads, and yields
only the events we care about: new post creations.

The firehose emits *all* event types — likes, follows, blocks, profile edits,
and posts. We filter in two stages:
  1. Only "commit" events (data writes), skip handles/identity events
  2. Only ops where the record type is app.bsky.feed.post AND action is "create"

Reference: https://atproto.com/specs/event-stream
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

from atproto import CAR, AtUri, FirehoseSubscribeReposClient, firehose_models, models
from atproto import parse_subscribe_repos_message

logger = logging.getLogger(__name__)


@dataclass
class BlueskyPost:
    """Minimal shape we extract from a firehose post-create event."""
    uri: str           # at://did:plc:.../app.bsky.feed.post/abc123 — globally unique
    did: str           # author DID (did:plc:...)
    text: str          # post body, max 300 chars
    langs: list[str]   # self-declared languages, e.g. ["en"]
    created_at: str    # ISO-8601 timestamp from the post record


def _iter_post_creates(commit) -> Iterator[BlueskyPost]:
    """Walk a commit's ops, yield BlueskyPost for each post creation."""
    # The commit carries a CAR (content-addressable archive) with the actual
    # record data. We need to decode it to read the post text.
    try:
        car = CAR.from_bytes(commit.blocks)
    except Exception as e:
        logger.debug("Could not decode CAR block: %s", e)
        return

    for op in commit.ops:
        if op.action != "create":
            continue
        # op.path looks like "app.bsky.feed.post/3jzfcijpj2z2a"
        if not op.path.startswith("app.bsky.feed.post/"):
            continue
        if op.cid is None:
            continue

        record_raw = car.blocks.get(op.cid)
        if record_raw is None:
            continue

        # The record should deserialize to a post. Some records are malformed
        # or use schemas we don't expect — skip those silently.
        try:
            record = models.get_or_create(record_raw, strict=False)
        except Exception:
            continue

        if not isinstance(record, models.AppBskyFeedPost.Record):
            continue

        text = (record.text or "").strip()
        if not text:
            continue

        uri = str(AtUri.from_str(f"at://{commit.repo}/{op.path}"))
        langs = list(record.langs or [])

        yield BlueskyPost(
            uri=uri,
            did=commit.repo,
            text=text,
            langs=langs,
            created_at=record.created_at,
        )


def stream_posts() -> Iterator[BlueskyPost]:
    """Generator yielding BlueskyPost objects from the live firehose.

    Handles reconnection internally via the atproto client. Blocks forever
    unless the caller breaks out or the process is signaled.
    """
    client = FirehoseSubscribeReposClient()

    # We need a channel between the library's callback-style API and our
    # generator. Simplest pattern: a queue.
    import queue
    q: queue.Queue = queue.Queue(maxsize=10_000)
    _sentinel = object()

    def on_message(message: firehose_models.MessageFrame) -> None:
        try:
            commit = parse_subscribe_repos_message(message)
        except Exception as e:
            logger.debug("Skipping unparseable frame: %s", e)
            return

        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            return
        if commit.blocks is None:
            return

        for post in _iter_post_creates(commit):
            try:
                q.put_nowait(post)
            except queue.Full:
                # If we can't keep up with the firehose, drop events rather
                # than block the websocket thread. Log once per 1000 drops.
                _drop_counter[0] += 1
                if _drop_counter[0] % 1000 == 0:
                    logger.warning(
                        "Dropped %d posts due to full queue (consumer too slow?)",
                        _drop_counter[0],
                    )

    _drop_counter = [0]

    # Run the websocket client in a background thread; drain the queue here.
    import threading
    def _run_client():
        try:
            client.start(on_message)
        except Exception as e:
            logger.error("Firehose client exited: %s", e)
            q.put(_sentinel)

    t = threading.Thread(target=_run_client, daemon=True, name="firehose-ws")
    t.start()

    while True:
        item = q.get()
        if item is _sentinel:
            logger.error("Firehose stream ended")
            return
        yield item
