from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple

import numpy as np


class BM25Index:
    """Lightweight BM25 index over a list of texts."""

    def __init__(self, documents: List[str], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.num_docs = len(documents)

        tokenized = [self._tokenize(doc) for doc in documents]
        self.term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lens: List[int] = []

        for tokens in tokenized:
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)
            self.doc_lens.append(len(tokens))
            for tok in tf:
                self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1

        self.avgdl = (sum(self.doc_lens) / self.num_docs) if self.num_docs else 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Simple tokenization to support DA/EN without heavy dependencies
        return re.findall(r"\w+", text.lower())

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        if df == 0 or self.num_docs == 0:
            return 0.0
        return math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_docs == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        q_tokens = self._tokenize(query)
        scores = np.zeros(self.num_docs, dtype=np.float32)
        avgdl = self.avgdl or 1.0

        for idx, tf in enumerate(self.term_freqs):
            dl = self.doc_lens[idx] or 1
            denom = self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            s = 0.0
            for term in q_tokens:
                tf_d = tf.get(term)
                if not tf_d:
                    continue
                idf = self._idf(term)
                numer = tf_d * (self.k1 + 1.0)
                s += idf * (numer / (tf_d + denom))
            scores[idx] = s

        if k >= len(scores):
            idxs = np.argsort(scores)[::-1]
        else:
            idxs = np.argpartition(scores, -k)[-k:]
            idxs = idxs[np.argsort(scores[idxs])][::-1]
        return idxs, scores[idxs]


def build_bm25_index_from_meta(meta_rows: List[Dict[str, Any]], *, text_field: str = "question") -> BM25Index:
    docs: List[str] = []
    for row in meta_rows:
        text = row.get(text_field) or row.get("aggregated_content") or ""
        docs.append(str(text))
    return BM25Index(docs)


def reciprocal_rank_fusion(rank_lists: List[np.ndarray], *, k: int, rrf_k: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse multiple ranked index arrays using RRF."""
    combined: Dict[int, float] = {}
    for arr in rank_lists:
        for pos, idx in enumerate(arr.tolist(), start=1):
            combined[idx] = combined.get(idx, 0.0) + 1.0 / (rrf_k + pos)

    sorted_items = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:k]
    if not sorted_items:
        return np.array([], dtype=int), np.array([], dtype=np.float32)

    fused_idx = np.array([item[0] for item in sorted_items], dtype=int)
    fused_scores = np.array([item[1] for item in sorted_items], dtype=np.float32)
    return fused_idx, fused_scores
