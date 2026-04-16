"""
Sprint 3: FastAPI service exposing sentiment data.

Three endpoints reading from sentiment_posts:
  - GET /stats                       overall label distribution
  - GET /trends?bucket=hour&hours=24 time-bucketed counts for charts
  - GET /recent?limit=50&label=...   latest scored posts (filterable)

Run locally:
    uvicorn main:app --reload

Auto-generated docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from db import close_pool, get_conn, init_pool

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")


# ---- lifespan: init/close DB pool around the app -------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    yield
    close_pool()


app = FastAPI(
    title="Sentiment Pipeline API",
    description="Aggregate and recent-post views over Bluesky sentiment data.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- response models -----------------------------------------------------

SentimentLabel = Literal["NEGATIVE", "NEUTRAL", "POSITIVE"]


class StatsResponse(BaseModel):
    negative: int = Field(..., description="Count of NEGATIVE posts")
    neutral: int = Field(..., description="Count of NEUTRAL posts")
    positive: int = Field(..., description="Count of POSITIVE posts")
    total: int = Field(..., description="Total posts scored")


class TrendBucket(BaseModel):
    bucket_start: datetime = Field(..., description="Start of this time bucket (UTC)")
    negative: int
    neutral: int
    positive: int
    total: int


class TrendsResponse(BaseModel):
    bucket: str
    hours: int
    data: list[TrendBucket]


class RecentPost(BaseModel):
    reddit_id: str
    subreddit: str
    title: str | None
    body: str | None
    sentiment_label: SentimentLabel
    sentiment_score: float
    created_utc: datetime
    ingested_at: datetime


class RecentResponse(BaseModel):
    count: int
    posts: list[RecentPost]


class HealthResponse(BaseModel):
    status: str
    db: str


# ---- endpoints ------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["health"])
def health():
    """Basic health check. Verifies DB is reachable."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return HealthResponse(status="ok", db="ok")
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"DB unreachable: {e}")


@app.get("/stats", response_model=StatsResponse, tags=["analytics"])
def stats():
    """Overall sentiment distribution across all scored posts."""
    sql = """
        SELECT sentiment_label, COUNT(*) AS n
        FROM sentiment_posts
        GROUP BY sentiment_label
    """
    counts = {"NEGATIVE": 0, "NEUTRAL": 0, "POSITIVE": 0}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            for label, n in cur.fetchall():
                if label in counts:
                    counts[label] = n

    return StatsResponse(
        negative=counts["NEGATIVE"],
        neutral=counts["NEUTRAL"],
        positive=counts["POSITIVE"],
        total=sum(counts.values()),
    )


@app.get("/trends", response_model=TrendsResponse, tags=["analytics"])
def trends(
    bucket: Literal["minute", "hour", "day"] = Query(
        "hour", description="Time bucket size for aggregation"
    ),
    hours: int = Query(
        24, ge=1, le=168, description="How many hours of history to include (max 7 days)"
    ),
):
    """Time-bucketed sentiment counts. Powers the dashboard's line chart."""
    # date_trunc bucketizes ingested_at to the requested granularity.
    # We aggregate counts per bucket per label, then pivot to one row per bucket.
    sql = f"""
        SELECT
            date_trunc(%s, ingested_at) AS bucket_start,
            COUNT(*) FILTER (WHERE sentiment_label = 'NEGATIVE') AS negative,
            COUNT(*) FILTER (WHERE sentiment_label = 'NEUTRAL')  AS neutral,
            COUNT(*) FILTER (WHERE sentiment_label = 'POSITIVE') AS positive,
            COUNT(*) AS total
        FROM sentiment_posts
        WHERE ingested_at >= NOW() - INTERVAL '%s hours'
        GROUP BY bucket_start
        ORDER BY bucket_start ASC
    """
    # Note: psycopg's parameter binding doesn't work for INTERVAL string interp,
    # so we format hours into the SQL directly. Safe because `hours` is an int
    # validated by FastAPI's Query(ge=1, le=168) — no injection risk.
    sql = sql % ("%s", hours)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (bucket,))
            rows = cur.fetchall()

    data = [
        TrendBucket(
            bucket_start=row[0],
            negative=row[1],
            neutral=row[2],
            positive=row[3],
            total=row[4],
        )
        for row in rows
    ]
    return TrendsResponse(bucket=bucket, hours=hours, data=data)


@app.get("/recent", response_model=RecentResponse, tags=["feed"])
def recent(
    limit: int = Query(50, ge=1, le=500, description="Max posts to return"),
    label: SentimentLabel | None = Query(
        None, description="Filter by sentiment label (optional)"
    ),
):
    """Latest scored posts, ordered by ingestion time descending."""
    sql = """
        SELECT reddit_id, subreddit, title, body,
               sentiment_label, sentiment_score, created_utc, ingested_at
        FROM sentiment_posts
        {where}
        ORDER BY ingested_at DESC
        LIMIT %s
    """
    if label:
        sql = sql.format(where="WHERE sentiment_label = %s")
        params = (label, limit)
    else:
        sql = sql.format(where="")
        params = (limit,)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    posts = [
        RecentPost(
            reddit_id=row[0],
            subreddit=row[1],
            title=row[2],
            body=row[3],
            sentiment_label=row[4],
            sentiment_score=float(row[5]),
            created_utc=row[6],
            ingested_at=row[7],
        )
        for row in rows
    ]
    return RecentResponse(count=len(posts), posts=posts)
