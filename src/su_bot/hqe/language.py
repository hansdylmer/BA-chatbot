from __future__ import annotations

import re

from ..models import Lang


def detect_lang(text: str, default: Lang = Lang.da) -> Lang:
    if any(ch in "æøåÆØÅ" for ch in text):
        return Lang.da
    if len(text) < 80:
        return default
    if re.search(r"\b(the|and|is|to|of|you|in|with|can)\b", text.lower()):
        return Lang.en
    return default

