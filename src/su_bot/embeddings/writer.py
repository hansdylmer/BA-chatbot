from __future__ import annotations

import argparse
import json
import logging
import time
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable

import numpy as np

from openai import OpenAI

from ..config import OpenAIConfig
from ..log_utils import setup_logging
from ..openai_client import get_openai_client


@dataclass(slots=True)
class EmbeddingArtifacts:
    matrix: np.ndarray
    meta: List[Dict[str, Any]]


def read_hqe_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def flatten_questions(hqe_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for rec in hqe_records:
        doc_id = rec["doc_id"]
        sec_idx = rec["section_index"]
        meta = rec.get("meta", {})
        for q in rec.get("questions", []):
            q = (q or "").strip()
            if not q:
                continue
            qid = f"hqe:{doc_id}:{sec_idx}:{hashlib.sha1(q.encode('utf-8')).hexdigest()}"
            items.append(
                {
                    "id": qid,
                    "q": q,
                    "meta": {"doc_id": doc_id, "section_index": sec_idx, "question": q, **meta},
                }
            )
    return items


def embed_batch(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def chunked(seq: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_embeddings(
    hqe_records: List[Dict[str, Any]],
    *,
    openai_cfg: OpenAIConfig,
    batch_size: int = 128,
    client: Optional[OpenAI] = None,
) -> EmbeddingArtifacts:
    setup_logging()
    flat = flatten_questions(hqe_records)
    total = len(flat)
    if total == 0:
        raise ValueError("No HQE questions to embed.")

    llm = client or get_openai_client(openai_cfg)
    probe_vec = embed_batch(llm, [flat[0]["q"]], openai_cfg.embedding_model)[0]
    dim = len(probe_vec)

    matrix = np.zeros((total, dim), dtype=np.float32)
    meta: List[Dict[str, Any]] = []

    t_start = time.perf_counter()
    done = 0
    for batch in chunked(flat, batch_size):
        texts = [b["q"] for b in batch]
        t_call = time.perf_counter()
        vectors = embed_batch(llm, texts, openai_cfg.embedding_model)
        dt = time.perf_counter() - t_call

        for offset, vec in enumerate(vectors):
            matrix[done + offset, :] = np.asarray(vec, dtype=np.float32)
            meta.append(batch[offset]["meta"])
        done += len(vectors)

        elapsed = time.perf_counter() - t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = max(0, total - done)
        eta = (remaining / rate) if rate > 0 else 0.0
        logging.info(
            "[EMB] +%s (%.2fs) | %s/%s | %.2f q/s | ETA %ss",
            len(vectors),
            dt,
            done,
            total,
            rate,
            int(eta),
        )

    return EmbeddingArtifacts(matrix=matrix, meta=meta)


def write_artifacts(artifacts: EmbeddingArtifacts, out_prefix: Path) -> Tuple[Path, Path]:
    out_emb = out_prefix.with_suffix(".embeddings.npy")
    out_meta = out_prefix.with_suffix(".meta.json")
    out_emb.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_emb, artifacts.matrix)
    out_meta.write_text(json.dumps(artifacts.meta, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved embeddings → %s (shape=%s)", out_emb, artifacts.matrix.shape)
    logging.info("Saved metadata → %s", out_meta)
    return out_emb, out_meta
