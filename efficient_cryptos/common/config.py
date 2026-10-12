"""Configuration loading: read YAML into a dict with dotted-path access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class Config(dict):
    """Simple config container supporting both cfg['a']['b'] and cfg.get_path('a.b')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for key in dotted.split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> Config:
    """Load a config file; overrides use dotted paths, e.g. {'general.leverage': 5}."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(copy.deepcopy(raw))
    for dotted, value in (overrides or {}).items():
        node: Dict[str, Any] = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return cfg
