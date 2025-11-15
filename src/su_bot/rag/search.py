from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from openai import OpenAI
from uuid import uuid5, NAMESPACE_URL

from ..openai_client import get_openai_client
from ..config import OpenAIConfig


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(str(path), mmap_mode="r")


def load_meta(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_corpus(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_section_lookup(corpus: List[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    def doc_id(link: str, title: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{link}::{title}"))

    lut: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for doc in corpus:
        did = doc_id(doc["link"], doc["title"])
        combined_parts: List[str] = []
        headings: List[str] = []
        for idx, sec in enumerate(doc["sections"]):
            lut[(did, idx)] = {
                "title": doc["title"],
                "link": doc["link"],
                "heading": sec.get("heading") or "(ingen overskrift)",
                "content": sec.get("content") or "",
            }
            heading = sec.get("heading") or "(ingen overskrift)"
            headings.append(heading)
            combined_parts.append(f"{heading}\n{sec.get('content') or ''}")

        if combined_parts:
            lut[(did, -1)] = {
                "title": doc["title"],
                "link": doc["link"],
                "heading": "(samlet dokument)",
                "content": "\n\n".join(combined_parts),
                "combined_headings": headings,
            }

    return lut


def embed_query(question: str, *, cfg: OpenAIConfig, client: OpenAI | None = None) -> np.ndarray:
    llm = client or get_openai_client(cfg)
    response = llm.embeddings.create(model=cfg.embedding_model, input=question)
    return np.array(response.data[0].embedding, dtype=np.float32)


def topk_dot(matrix: np.ndarray, q_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = matrix @ q_vec
    if k >= len(scores):
        idx = np.argsort(scores)[::-1]
    else:
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])][::-1]
    return idx, scores[idx]


def make_context(chunks: List[Dict[str, Any]], per_section_chars: int, total_chars: int) -> str:
    parts: List[str] = []
    total = 0
    for chunk in chunks:
        title = chunk.get("title") or "(ukendt titel)"
        heading = chunk.get("heading") or "(ingen overskrift)"
        link = chunk.get("link") or ""
        content = chunk.get("content") or ""
        block = f"### {title} - {heading}\nKilde: {link}\n{_truncate(content, per_section_chars)}"
        if total_chars > 0 and (total + len(block)) > total_chars:
            remaining = max(0, total_chars - total)
            block = _truncate(block, remaining)
            parts.append(block)
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
