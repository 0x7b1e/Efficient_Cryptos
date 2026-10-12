"""Incrementally collect whitelisted KOL tweets into the local SQLite store."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from .collector import collect_whitelist
from .linking import link_entities
from .storage import KolStore


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _query_since(value: str | None) -> str | None:
    """Convert an ISO cursor to X's date-based ``since:`` syntax."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value[:10]


async def _collect(args: argparse.Namespace) -> int:
    # Lazy import keeps scoring/building usable without the optional twscrape dep.
    from twscrape import API

    whitelist = _load_yaml(args.config)
    entity_cfg = _load_yaml(args.entity_map)
    kol_rows = whitelist.get("kol", whitelist.get("kols", []))
    handles = [str(item["handle"]).lstrip("@").lower() for item in kol_rows]
    if not handles:
        raise SystemExit(f"No KOL handles found in {args.config}")

    aliases = entity_cfg.get("entities", entity_cfg)
    store = KolStore(args.db)
    api = API(
        args.accounts_db,
        proxy=args.proxy,
        raise_when_no_account=True,
        wait_timeout=args.wait_timeout,
        wait_interval=1,
    )
    since_by_handle = {
        handle: _query_since(store.get_metadata(f"last_seen:{handle}"))
        for handle in handles
    }
    total = 0
    newest: dict[str, str] = {}
    try:
        async for tweet in collect_whitelist(
            api, handles, since_by_handle=since_by_handle,
            limit_per_kol=args.limit, language=args.language,
            exclude_retweets=not args.include_retweets,
        ):
            store.upsert_tweet(tweet)
            for mention in link_entities(tweet.text, aliases):
                store.upsert_mention(replace(mention, tweet_id=tweet.tweet_id))
            current = newest.get(tweet.kol_handle)
            if current is None or tweet.created_at > current:
                newest[tweet.kol_handle] = tweet.created_at
            total += 1
            if total % 100 == 0:
                store.commit()
                print(f"collected={total}", flush=True)
    finally:
        for handle, created_at in newest.items():
            store.set_metadata(f"last_seen:{handle}", created_at)
        store.commit()
        store.close()

    print(f"collected={total} kols={len(handles)} db={args.db}")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/kol_factor/tweets.sqlite")
    parser.add_argument("--accounts-db", default="data/kol_factor/twscrape_accounts.db")
    parser.add_argument("--config", default="config/kol/kol_whitelist.yaml")
    parser.add_argument("--entity-map", default="config/kol/entity_map.yaml")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--language", default="en")
    parser.add_argument("--include-retweets", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=30.0)
    args = parser.parse_args()
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.accounts_db).parent.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_collect(args))


if __name__ == "__main__":
    raise SystemExit(main())
