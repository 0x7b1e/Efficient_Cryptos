from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import TweetAnnotation, TweetRecord


@dataclass(frozen=True)
class AnnotationConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"


def annotation_prompt(tweet: TweetRecord, symbol: str) -> str:
    return (
        "Return JSON only. Analyze the tweet's actionable information for the "
        f"crypto asset {symbol}. sentiment must be in [-1,1]; conviction, novelty, "
        "hype, relevance must be in [0,1]. Do not infer facts not present in the tweet.\n\n"
        f"KOL: @{tweet.kol_handle}\nTweet: {tweet.text}"
    )


def parse_annotation(tweet_id: str, symbol: str, payload: str | dict,
                     *, model: str = "") -> TweetAnnotation:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    return TweetAnnotation(
        tweet_id=tweet_id, symbol=symbol,
        sentiment=float(data.get("sentiment", 0.0)),
        conviction=float(data.get("conviction", 0.0)),
        novelty=float(data.get("novelty", 0.0)),
        hype=float(data.get("hype", 0.0)),
        relevance=float(data.get("relevance", 0.0)),
        rationale=str(data.get("rationale", "")), model=model,
    ).clipped()
