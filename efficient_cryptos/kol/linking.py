from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .schema import EntityMention


def link_entities(text: str, aliases: Mapping[str, Sequence[str]]) -> list[EntityMention]:
    """Conservative cashtag/alias linker; ambiguous aliases belong in config review."""
    text_lower = str(text).lower()
    cashtags = {f"{x.upper()}USDT" for x in re.findall(r"\$([A-Za-z0-9]{2,15})", text)}
    found: dict[str, EntityMention] = {}
    for symbol, names in aliases.items():
        symbol = str(symbol).upper()
        if symbol in cashtags:
            found[symbol] = EntityMention("", symbol, f"${symbol[:-4]}", 1.0)
            continue
        for alias in names:
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(str(alias).lower()) + r"(?![A-Za-z0-9_])"
            if re.search(pattern, text_lower):
                found[symbol] = EntityMention("", symbol, str(alias), 0.8)
                break
    return list(found.values())
