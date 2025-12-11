from __future__ import annotations

from typing import Dict, Any, List, Tuple

from openai import OpenAI

from ..config import OpenAIConfig
from ..openai_client import get_openai_client
from .prompts import build_instructions


def answer_query(
    *,
    prompt: str,
    context_chunks: List[Dict[str, Any]],
    scores: List[float],
    chat_model: str,
    per_section_chars: int,
    total_chars: int,
    openai_cfg: OpenAIConfig,
    client: OpenAI | None = None,
) -> Tuple[str, List[Tuple[Dict[str, Any], float]]]:
    from .search import make_context  # local import to avoid cycle

    context = make_context(context_chunks, per_section_chars=per_section_chars, total_chars=total_chars)
    instructions = build_instructions(prompt, context)
    llm = client or get_openai_client(openai_cfg)
    response = llm.responses.create(model=chat_model, input=instructions, stream=False, store=False, temperature = 0)
    answer = (response.output_text or "").strip()
    shown_sources = list(zip(context_chunks, scores))
    return answer, shown_sources

