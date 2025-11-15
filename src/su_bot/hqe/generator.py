from __future__ import annotations

import datetime
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional

from openai import OpenAI

from ..config import HQEConfig, OpenAIConfig
from ..log_utils import setup_logging
from ..models import Budget, Corpus, HQERecord, Lang, Document
from ..openai_client import get_openai_client
from .language import detect_lang
from .quality import quality_filter, dedup_by_text, normalize_q


LLM_TIMES: List[float] = []


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def q_budget(text: str, budget: Budget) -> int:
    length = len(text)
    if length < 500:
        return budget.small
    if length < 1500:
        return budget.medium
    return budget.large


def gen_hqe_questions_llm(
    title: str,
    headline: str,
    section_text: str,
    *,
    n: int,
    language: Lang,
    max_chars: int,
    model: str,
    client: Optional[OpenAI] = None,
) -> List[str]:
    text = section_text[:max_chars] if max_chars and len(section_text) > max_chars else section_text
    sys_prompt = f"You generate {language.value.upper()} student-style search questions."
    user_prompt = f"""
Generer {n} forskellige spørgsmål, som en studerende kunne stille, hvor følgende nedenstående tekst er relevant for at finde svaret.
Vær opmærksom på at Titel og overskrifter som du får ofte er vigtige for at forstå konteksten.
Overhold følgende:
- Output skal være en JSON-array af strenge, f.eks. ["spørgsmål 1?", "spørgsmål 2?", ...].
- Spørgsmål skal være relevante for teksten.
- Hvert spørgsmål skal være komplet (ingen pronominer som 'det/den' uden et subjekt).
- Hold spørgsmålene korte (~5-18 ord).
- Sprog: {language.value}.

Titel: \"\"\"{title}\"\"\"
Overskrift: \"\"\"{headline}\"\"\"
Tekst: \"\"\"{text}\"\"\""""

    llm = client or get_openai_client()
    t0 = time.perf_counter()
    resp = llm.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        stream=False,
        store=False,
    )
    dt = time.perf_counter() - t0
    LLM_TIMES.append(dt)
    logging.debug("LLM call: %s chars → %.3fs", len(text), dt)

    content = (resp.output_text or "").strip()
    try:
        arr = json.loads(content)
        if not isinstance(arr, list):
            raise ValueError("Expected a JSON list.")
    except Exception:
        arr = [content] if content else []
    arr = [normalize_q(x) for x in arr if isinstance(x, str) and x.strip()]
    return arr[:n]


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def section_hash(text: str) -> str:
    return f"sha1:{sha1(text)}"


def build_hqe_sidecar(
    corpus: Corpus,
    *,
    budget: Budget,
    lang_strategy: str,
    cfg: HQEConfig,
    openai_cfg: OpenAIConfig,
    client: Optional[OpenAI] = None,
) -> List[Dict[str, Any]]:
    setup_logging()
    llm = client or get_openai_client(openai_cfg)
    out: List[Dict[str, Any]] = []
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    total_docs = len(corpus.documents())
    processed = 0
    t_start = time.perf_counter()
    log_every = 25

    for doc in corpus.documents():
        doc_text, headings = aggregate_document(doc)
        if not doc_text.strip():
            continue

        if lang_strategy == "auto":
            lang = detect_lang(doc_text)
        elif lang_strategy in {"da", "en"}:
            lang = Lang(lang_strategy)
        else:
            lang = detect_lang(doc_text)

        n = q_budget(doc_text, budget)
        qs_raw = gen_hqe_questions_llm(
            doc.title,
            "(samlet dokument)",
            doc_text,
            n=n,
            language=lang,
            max_chars=cfg.max_chars,
            model=openai_cfg.chat_model,
            client=llm,
        )

        qs_q = quality_filter(qs_raw)
        qs_final = dedup_by_text(qs_q, thresh=cfg.dedup_jaccard)
        if not qs_final:
            fallback = "Hvad er hovedreglen?" if lang == Lang.da else "What is the main rule?"
            qs_final = qs_q[:1] or qs_raw[:1] or [fallback]

        record = HQERecord(
            doc_id=doc.doc_id,
            section_index=-1,
            language=lang,
            gen_model=openai_cfg.chat_model,
            prompt_version=cfg.prompt_version,
            generated_at=now,
            questions=qs_final,
            meta={
                "source": str(doc.link),
                "page_title": doc.title,
                "section_heading": "(samlet dokument)",
                "combined_headings": headings,
                "aggregated_content": doc_text,
                "text_hash": section_hash(doc_text),
                "section_char_len": len(doc_text),
            },
        )
        out.append(record.model_dump())

        processed += 1
        if processed % log_every == 0 or processed == total_docs:
            avg = sum(LLM_TIMES) / len(LLM_TIMES) if LLM_TIMES else 0.0
            remaining = max(0, total_docs - processed)
            eta_secs = remaining * avg
            elapsed = time.perf_counter() - t_start
            logging.info(
                "[HQE] %s/%s documents | avg_call=%.2fs | elapsed=%s | ETA=%s",
                processed,
                total_docs,
                avg,
                fmt_duration(elapsed),
                fmt_duration(eta_secs),
            )

    if LLM_TIMES:
        try:
            import statistics as stats

            p90 = stats.quantiles(LLM_TIMES, n=10)[8] if len(LLM_TIMES) >= 10 else max(LLM_TIMES)
        except Exception:
            p90 = max(LLM_TIMES)
        logging.info(
            "LLM calls: %s | avg=%.3fs | p90=%.3fs",
            len(LLM_TIMES),
            sum(LLM_TIMES) / len(LLM_TIMES),
            p90,
        )

    return out


def load_corpus(path: str | Path) -> Corpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = [Document(**doc) for doc in data]
    return Corpus(root=docs)


def aggregate_document(doc: Document) -> tuple[str, List[str]]:
    headings: List[str] = []
    parts: List[str] = []
    for sec in doc.sections:
        heading = sec.heading or "(ingen overskrift)"
        headings.append(heading)
        parts.append(f"{heading}\n{sec.content}")
    return "\n\n".join(parts).strip(), headings
