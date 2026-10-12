from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd

from .schema import TweetAnnotation, TweetRecord


@dataclass(frozen=True)
class ScoreConfig:
    half_life_hours: float = 12.0
    relevance_threshold: float = 0.50
    novelty_weight: float = 0.0
    hype_weight: float = 0.0
    engagement_log_scale: bool = True


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _decay(age_hours: float, half_life_hours: float) -> float:
    return 2.0 ** (-max(0.0, age_hours) / max(half_life_hours, 1e-9))


def _engagement(tweet: TweetRecord, log_scale: bool) -> float:
    raw = max(0.0, tweet.engagement)
    return math.log1p(raw) if log_scale else raw


def aggregate_30m(
    tweets: Iterable[TweetRecord], annotations: Iterable[TweetAnnotation],
    kol_weights: dict[str, float], *, now: datetime | None = None,
    config: ScoreConfig | None = None,
) -> pd.DataFrame:
    """Aggregate tweet/entity annotations into one row per 30m/symbol."""
    config = config or ScoreConfig()
    now = now or datetime.now(timezone.utc)
    tweets_by_id = {tweet.tweet_id: tweet for tweet in tweets}
    rows = []
    for annotation in annotations:
        item = annotation.clipped()
        if item.relevance < config.relevance_threshold or item.tweet_id not in tweets_by_id:
            continue
        tweet = tweets_by_id[item.tweet_id]
        created = _parse_time(tweet.created_at)
        age_hours = (now - created).total_seconds() / 3600.0
        novelty_multiplier = 1.0 + config.novelty_weight * (item.novelty - 0.5) * 2.0
        hype_multiplier = 1.0 + config.hype_weight * (item.hype - 0.5) * 2.0
        score = (
            float(kol_weights.get(tweet.kol_handle.lower(), 0.0))
            * _engagement(tweet, config.engagement_log_scale)
            * _decay(age_hours, config.half_life_hours)
            * item.sentiment * item.conviction
            * novelty_multiplier * hype_multiplier
        )
        rows.append({
            "timestamp": pd.Timestamp(created).tz_convert("UTC").floor("30min"),
            "symbol": item.symbol.upper(), "score": score,
            "abs_score": abs(score), "kol_handle": tweet.kol_handle.lower(),
        })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "symbol", "kol_factor", "kol_abs_factor", "mention_count", "positive_score", "negative_score"])
    raw = pd.DataFrame(rows)
    raw["positive"] = raw["score"].clip(lower=0)
    raw["negative"] = raw["score"].clip(upper=0)
    out = raw.groupby(["timestamp", "symbol"], as_index=False).agg(
        kol_factor=("score", "sum"), kol_abs_factor=("abs_score", "sum"),
        mention_count=("score", "size"), positive_score=("positive", "sum"),
        negative_score=("negative", "sum"),
    )
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
