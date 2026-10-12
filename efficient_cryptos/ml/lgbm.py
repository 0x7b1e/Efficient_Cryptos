"""Optional LightGBM adapter for the 30m cross-sectional model."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def train_lgbm(sections, *, num_boost_round: int = 300, learning_rate: float = 0.03,
               num_leaves: int = 31, seed: int = 42):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("--model lgbm requires lightgbm; install requirements-ml.txt") from exc
    x = np.concatenate([s.x.reshape(len(s.symbols), -1) for s in sections], axis=0)
    y = np.concatenate([s.y for s in sections], axis=0)
    model = lgb.LGBMRegressor(
        n_estimators=int(num_boost_round), learning_rate=float(learning_rate),
        num_leaves=int(num_leaves), subsample=0.9, colsample_bytree=0.8,
        random_state=int(seed), objective="regression", verbosity=-1,
    )
    model.fit(x, y)
    return model


def predict_lgbm(model, sections):
    x = np.concatenate([s.x.reshape(len(s.symbols), -1) for s in sections], axis=0)
    return model.predict(x)
