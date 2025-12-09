from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

from ..data_model import Lang

# Make language detection deterministic for the same input
DetectorFactory.seed = 0


def detect_lang(text: str, default: Lang = Lang.da) -> Lang:
    cleaned = (text or "").strip()
    if not cleaned:
        return default

    try:
        code = detect(cleaned)
    except LangDetectException:
        return default

    if code.startswith("da"):
        return Lang.da
    if code.startswith("en"):
        return Lang.en
    return default
