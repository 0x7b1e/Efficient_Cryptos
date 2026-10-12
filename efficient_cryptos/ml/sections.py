"""Cross-sectional sequence samples for the ML models (needs torch)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

def select_features(frame: pd.DataFrame, features: list[str], *, train_fraction: float = 0.70,
                    max_missing_rate: float = 0.35, max_features: int = 192,
                    min_std: float = 1e-8,
                    train_end: str | pd.Timestamp | None = None) -> list[str]:
    """Filter features using the training period only, optionally cap by correlation.

    Core raw/funding/positioning inputs receive priority. Remaining slots are
    ranked by absolute pooled training correlation with the T+1 target. This is
    a lightweight pre-filter, not a test-period feature selection.
    """
    if train_end is not None:
        cutoff = pd.Timestamp(train_end)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        cutoff = cutoff + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        train = frame[pd.to_datetime(frame["timestamp"], utc=True) <= cutoff]
    else:
        timestamps = np.asarray(sorted(frame["timestamp"].unique()))
        cutoff_index = min(len(timestamps) - 1, max(0, int(len(timestamps) * train_fraction) - 1))
        train = frame[frame["timestamp"] <= timestamps[cutoff_index]]
    if train.empty:
        raise ValueError("feature-selection training period is empty")
    valid = []
    scores = {}
    for feature in features:
        series = pd.to_numeric(train[feature], errors="coerce")
        if float(series.isna().mean()) > max_missing_rate:
            continue
        if series.nunique(dropna=True) < 3 or float(series.std(skipna=True)) <= min_std:
            continue
        valid.append(feature)
        pair = pd.concat([series, train["target"]], axis=1).dropna()
        scores[feature] = abs(float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))) if len(pair) >= 3 else 0.0
    if max_features <= 0 or len(valid) <= max_features:
        return valid
    # Explicitly selected/generated factor caches have already passed their
    # own train-only screen. Preserve them ahead of opportunistic technical
    # features when a feature cap is used.
    priority = [feature for feature in valid if feature in CORE_FEATURES]
    priority.extend(feature for feature in valid
                    if feature.startswith("factor__") and feature not in priority)
    remaining = sorted((feature for feature in valid if feature not in priority),
                       key=lambda feature: scores.get(feature, 0.0), reverse=True)
    return (priority + remaining)[:max_features]


@dataclass(frozen=True)
class Section:
    timestamp: pd.Timestamp
    symbols: tuple[str, ...]
    windows: tuple[np.ndarray, ...]
    y: np.ndarray

    @property
    def x(self) -> np.ndarray:
        """Materialize one cross-section only when it is sent to the model."""
        return np.stack(self.windows)


def build_sections(frame: pd.DataFrame, features: list[str], seq_len: int,
                   min_symbols: int = 5) -> list[Section]:
    by_timestamp: dict[pd.Timestamp, list[tuple[str, np.ndarray, float]]] = {}
    for symbol, part in frame.groupby("symbol", sort=False):
        part = part.sort_values("timestamp").reset_index(drop=True)
        values = part[features].to_numpy(dtype=np.float32)
        targets = part["target"].to_numpy(dtype=np.float32)
        times = part["timestamp"].tolist()
        for i in range(seq_len - 1, len(part)):
            x = values[i - seq_len + 1:i + 1]
            if not np.isfinite(x).all() or not np.isfinite(targets[i]):
                continue
            by_timestamp.setdefault(times[i], []).append((symbol, x, float(targets[i])))
    sections = []
    for timestamp in sorted(by_timestamp):
        rows = by_timestamp[timestamp]
        if len(rows) < min_symbols:
            continue
        sections.append(Section(
            timestamp=timestamp,
            symbols=tuple(row[0] for row in rows),
            windows=tuple(row[1] for row in rows),
            y=np.asarray([row[2] for row in rows], dtype=np.float32),
        ))
    if not sections:
        raise ValueError("No valid cross-sections; lower --min-symbols or inspect feature coverage")
    return sections


def section_tensors(section: Section, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(section.x).to(device), torch.from_numpy(section.y).to(device)
