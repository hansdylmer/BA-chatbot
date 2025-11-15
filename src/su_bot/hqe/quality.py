from __future__ import annotations

import re
from typing import List

TOKEN_SPLIT = re.compile(r"[^\wæøåéóü]+", re.IGNORECASE)


def normalize_q(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip("?.!").strip() + "?"


def within_len(q: str, min_w: int = 5, max_w: int = 18) -> bool:
    wc = len(q.split())
    return min_w <= wc <= max_w


def tokenize(q: str) -> List[str]:
    return [t for t in TOKEN_SPLIT.split(q.lower()) if t]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def dedup_by_text(questions: List[str], thresh: float = 0.8) -> List[str]:
    keep: List[str] = []
    keep_sets: List[set[str]] = []
    for q in questions:
        s = set(tokenize(q))
        if not keep:
            keep.append(q)
            keep_sets.append(s)
            continue
        if all(jaccard(s, ks) < thresh for ks in keep_sets):
            keep.append(q)
            keep_sets.append(s)
    return keep


def is_self_contained(q: str) -> bool:
    bad = re.search(r"\b(this|that|it|they|these|those|den|det|de|disse|dette)\b", q.strip().lower())
    return not bad


def has_domain_terms(q: str) -> bool:
    terms = [
        "su",
        "minsu",
        "fribeløb",
        "stipendium",
        "digital post",
        "bekendtgørelse",
        "allowance",
        "grant",
        "income",
        "loan",
        "threshold",
        "udlandsstipendium",
        "slutløn",
        "handicaptillæg",
        "forsørgertillæg",
    ]
    ql = q.lower()
    return any(t in ql for t in terms)


def quality_filter(questions: List[str]) -> List[str]:
    out: List[str] = []
    for q in questions:
        qn = normalize_q(q)
        if within_len(qn) and is_self_contained(qn) and has_domain_terms(qn):
            out.append(qn)
    if out:
        return out
    return [normalize_q(questions[0])] if questions else []

