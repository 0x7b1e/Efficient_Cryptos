"""Standalone meme-coin abnormal-event research package.

This package is intentionally separate from the existing cross-sectional
30-minute strategy.  It works on short-horizon 1m/5m event features and can
later be connected to a paper/live execution adapter.
"""

from .detector import EventDetector, EventDetectorConfig
from .cooldown import CooldownBook, CooldownConfig

__all__ = ["EventDetector", "EventDetectorConfig", "CooldownBook", "CooldownConfig"]
