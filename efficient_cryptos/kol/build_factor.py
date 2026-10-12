from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .schema import TweetAnnotation, TweetRecord
from .scoring import ScoreConfig, aggregate_30m
from .storage import KolStore


def _load_rows(store: KolStore):
    tweets = [TweetRecord(**dict(row)) for row in store.conn.execute("SELECT * FROM tweets")]
    annotations = [TweetAnnotation(**dict(row)) for row in store.conn.execute("SELECT * FROM annotations")]
    return tweets, annotations


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the 30m KOL factor from SQLite annotations")
    ap.add_argument("--db", default="data/kol_factor/kol.db")
    ap.add_argument("--config", default="config/kol/kol_whitelist.yaml")
    ap.add_argument("--out", default="data/features/kol_30m.parquet")
    args = ap.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    rows = payload.get("kol", [])
    weights = {str(row["handle"]).lower(): float(row.get("weight", 1.0)) for row in rows}
    scoring = payload.get("scoring", {})
    config = ScoreConfig(**{k: scoring[k] for k in ScoreConfig.__dataclass_fields__ if k in scoring})
    store = KolStore(args.db)
    try:
        tweets, annotations = _load_rows(store)
    finally:
        store.close()
    factor = aggregate_30m(tweets, annotations, weights, config=config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    factor.to_parquet(args.out, index=False)
    print(json.dumps({"rows": len(factor), "out": str(Path(args.out).resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
