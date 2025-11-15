from __future__ import annotations

from __future__ import annotations

from typing import Dict

from openai import OpenAI

from .config import OpenAIConfig

_CLIENT_CACHE: Dict[str, OpenAI] = {}


def get_openai_client(cfg: OpenAIConfig | None = None) -> OpenAI:
    """Return a cached OpenAI client keyed by API key."""
    config = cfg or OpenAIConfig()
    api_key = config.require_key()
    cached = _CLIENT_CACHE.get(api_key)
    if cached:
        return cached
    client = OpenAI(api_key=api_key)
    _CLIENT_CACHE[api_key] = client
    return client
