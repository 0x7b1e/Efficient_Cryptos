from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from .schema import TweetRecord, tweet_from_twscrape


def whitelist_query(handle: str, *, since: str | None = None, language: str | None = "en",
                    exclude_retweets: bool = True) -> str:
    parts = [f"from:{handle.lstrip('@')}"]
    if exclude_retweets:
        parts.append("-is:retweet")
    if language:
        parts.append(f"lang:{language}")
    if since:
        parts.append(f"since:{since}")
    return " ".join(parts)


async def collect_whitelist(
    api, handles: Iterable[str], *, since_by_handle: dict[str, str] | None = None,
    limit_per_kol: int = 100, language: str | None = "en",
    exclude_retweets: bool = True,
) -> AsyncIterator[TweetRecord]:
    """Use the current twscrape async API without importing it at module load."""
    since_by_handle = since_by_handle or {}
    for handle in handles:
        handle = str(handle).lstrip("@").lower()
        query = whitelist_query(
            handle, since=since_by_handle.get(handle), language=language,
            exclude_retweets=exclude_retweets,
        )
        async for tweet in api.search(query, limit=limit_per_kol):
            yield tweet_from_twscrape(tweet, handle)
