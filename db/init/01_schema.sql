CREATE TABLE IF NOT EXISTS sentiment_posts (
    id              BIGSERIAL PRIMARY KEY,
    reddit_id       TEXT UNIQUE NOT NULL,
    subreddit       TEXT NOT NULL,
    title           TEXT,
    body            TEXT,
    created_utc     TIMESTAMPTZ NOT NULL,
    sentiment_label TEXT NOT NULL,
    sentiment_score REAL NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sentiment_created ON sentiment_posts (created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_subreddit ON sentiment_posts (subreddit);
