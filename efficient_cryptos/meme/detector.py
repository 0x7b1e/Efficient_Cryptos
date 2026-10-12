"""Event trigger logic for meme-coin monitoring."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class EventDetectorConfig:
    min_event_percentile: float = 0.95
    min_volume_shock: float = 0.50
    min_abs_return_5m: float = 0.01
    min_assets: int = 8


class EventDetector:
    def __init__(self, config: EventDetectorConfig | None = None):
        self.config = config or EventDetectorConfig()

    def detect(self, features: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "symbol", "event_score", "event_percentile", "volume_shock_5m", "ret_5m"}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"missing event columns: {sorted(missing)}")
        out = features.copy()
        active = (
            (out["event_percentile"] >= self.config.min_event_percentile)
            & (out["volume_shock_5m"] >= self.config.min_volume_shock)
            & (out["ret_5m"].abs() >= self.config.min_abs_return_5m)
        )
        out["event_active"] = active
        out["event_side"] = 0
        out.loc[active & (out["event_score"] > 0), "event_side"] = 1
        out.loc[active & (out["event_score"] < 0), "event_side"] = -1
        return out
