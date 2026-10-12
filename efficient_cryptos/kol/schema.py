from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TweetRecord:
    tweet_id: str
    kol_handle: str
    created_at: str
    text: str
    reply_count: int = 0
    retweet_count: int = 0
    like_count: int = 0
    quote_count: int = 0
    view_count: int = 0
    raw_json: str = "{}"
    fetched_at: str = ""

    @property
    def engagement(self) -> float:
        return float(
            self.reply_count + 2 * self.retweet_count + 2 * self.quote_count
            + self.like_count + self.view_count * 0.001
        )


@dataclass(frozen=True)
class EntityMention:
    tweet_id: str
    symbol: str
    matched_alias: str
    confidence: float = 1.0


@dataclass(frozen=True)
class TweetAnnotation:
    tweet_id: str
    symbol: str
    sentiment: float
    conviction: float
    novelty: float
    hype: float
    relevance: float = 1.0
    rationale: str = ""
    model: str = ""
    annotated_at: str = ""

    def clipped(self) -> "TweetAnnotation":
        bounded = lambda value, low=0.0, high=1.0: min(high, max(low, float(value)))
        return TweetAnnotation(
            tweet_id=self.tweet_id,
            symbol=self.symbol.upper(),
            sentiment=min(1.0, max(-1.0, float(self.sentiment))),
            conviction=bounded(self.conviction), novelty=bounded(self.novelty),
            hype=bounded(self.hype), relevance=bounded(self.relevance),
            rationale=self.rationale, model=self.model,
            annotated_at=self.annotated_at or datetime.now(timezone.utc).isoformat(),
        )


def tweet_from_twscrape(tweet: Any, kol_handle: str) -> TweetRecord:
    """Convert a twscrape Tweet without importing twscrape at module import time."""
    raw = tweet.dict() if hasattr(tweet, "dict") else {}
    user = getattr(tweet, "user", None)
    handle = str(getattr(user, "username", kol_handle) or kol_handle).lower()
    created = getattr(tweet, "date", datetime.now(timezone.utc))
    return TweetRecord(
        tweet_id=str(getattr(tweet, "id", getattr(tweet, "id_str", ""))),
        kol_handle=handle,
        created_at=created.isoformat(),
        text=str(getattr(tweet, "rawContent", "")),
        reply_count=int(getattr(tweet, "replyCount", 0) or 0),
        retweet_count=int(getattr(tweet, "retweetCount", 0) or 0),
        like_count=int(getattr(tweet, "likeCount", 0) or 0),
        quote_count=int(getattr(tweet, "quoteCount", 0) or 0),
        view_count=int(getattr(tweet, "viewCount", 0) or 0),
        raw_json=json.dumps(raw, ensure_ascii=False, default=str),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
