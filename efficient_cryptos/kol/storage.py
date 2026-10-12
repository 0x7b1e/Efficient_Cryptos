from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import EntityMention, TweetAnnotation, TweetRecord


class KolStore:
    """Idempotent SQLite store for tweets, links, annotations, and cursors."""

    def __init__(self, path: str | Path = "data/kol_factor/kol.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS tweets (
                tweet_id TEXT PRIMARY KEY, kol_handle TEXT NOT NULL,
                created_at TEXT NOT NULL, text TEXT NOT NULL,
                reply_count INTEGER NOT NULL, retweet_count INTEGER NOT NULL,
                like_count INTEGER NOT NULL, quote_count INTEGER NOT NULL,
                view_count INTEGER NOT NULL, raw_json TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS entity_mentions (
                tweet_id TEXT NOT NULL, symbol TEXT NOT NULL,
                matched_alias TEXT NOT NULL, confidence REAL NOT NULL,
                PRIMARY KEY (tweet_id, symbol)
            );
            CREATE TABLE IF NOT EXISTS annotations (
                tweet_id TEXT NOT NULL, symbol TEXT NOT NULL,
                sentiment REAL NOT NULL, conviction REAL NOT NULL,
                novelty REAL NOT NULL, hype REAL NOT NULL,
                relevance REAL NOT NULL, rationale TEXT NOT NULL,
                model TEXT NOT NULL, annotated_at TEXT NOT NULL,
                PRIMARY KEY (tweet_id, symbol)
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert_tweet(self, tweet: TweetRecord) -> None:
        self.conn.execute(
            """INSERT INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tweet_id) DO UPDATE SET
              text=excluded.text, reply_count=excluded.reply_count,
              retweet_count=excluded.retweet_count, like_count=excluded.like_count,
              quote_count=excluded.quote_count, view_count=excluded.view_count,
              raw_json=excluded.raw_json, fetched_at=excluded.fetched_at""",
            (tweet.tweet_id, tweet.kol_handle, tweet.created_at, tweet.text,
             tweet.reply_count, tweet.retweet_count, tweet.like_count,
             tweet.quote_count, tweet.view_count, tweet.raw_json, tweet.fetched_at),
        )

    def upsert_mention(self, mention: EntityMention) -> None:
        self.conn.execute(
            """INSERT INTO entity_mentions VALUES (?, ?, ?, ?)
            ON CONFLICT(tweet_id, symbol) DO UPDATE SET
              matched_alias=excluded.matched_alias, confidence=excluded.confidence""",
            (mention.tweet_id, mention.symbol, mention.matched_alias, mention.confidence),
        )

    def upsert_annotation(self, annotation: TweetAnnotation) -> None:
        item = annotation.clipped()
        self.conn.execute(
            """INSERT INTO annotations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tweet_id, symbol) DO UPDATE SET
              sentiment=excluded.sentiment, conviction=excluded.conviction,
              novelty=excluded.novelty, hype=excluded.hype,
              relevance=excluded.relevance, rationale=excluded.rationale,
              model=excluded.model, annotated_at=excluded.annotated_at""",
            (item.tweet_id, item.symbol, item.sentiment, item.conviction,
             item.novelty, item.hype, item.relevance, item.rationale,
             item.model, item.annotated_at),
        )

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_metadata(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return default if row is None else str(row["value"])

    def commit(self) -> None:
        self.conn.commit()
