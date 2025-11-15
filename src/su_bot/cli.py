from __future__ import annotations

import asyncio
import datetime
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from .config import AppConfig, load_config
from .hqe.generator import build_hqe_sidecar, load_corpus
from .models import Budget
from .log_utils import setup_logging
from .embeddings.writer import read_hqe_jsonl, build_embeddings, write_artifacts
from .scraping.links import scrape_links
from .scraping.content import scrape_content
from .rag.search import (
    load_embeddings as load_np_embeddings,
    load_meta as load_meta_file,
    load_corpus as load_corpus_json,
    build_section_lookup,
    embed_query,
    topk_dot,
)
from .rag.chat import answer_query


app = typer.Typer(help="CLI for the SU BOT pipeline.", no_args_is_help=True)


def _default_date(date: Optional[str]) -> str:
    return date or datetime.date.today().isoformat()


def _infer_embed_model(dim: int, default: str) -> str:
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    return default


@app.command("scrape-links")
def cli_scrape_links(
    date: Optional[str] = typer.Option(None, help="ISO date to use for output folder"),
):
    """Scrape site map links and store them under the configured data directory."""
    cfg: AppConfig = load_config()
    day = _default_date(date)
    asyncio.run(scrape_links(day, cfg.paths, cfg.scrape))


@app.command("scrape-content")
def cli_scrape_content(
    date: Optional[str] = typer.Option(None, help="ISO date matching the link scrape"),
):
    """Scrape page content for all links found previously."""
    cfg: AppConfig = load_config()
    day = _default_date(date)
    asyncio.run(scrape_content(day, cfg.paths, cfg.scrape))


@app.command("hqe-generate")
def cli_hqe_generate(
    corpus: Path = typer.Argument(..., exists=True, readable=True, help="Scraped corpus JSON"),
    out: Path = typer.Argument(..., help="Output JSONL path for HQE records"),
    lang_strategy: str = typer.Option("da", help="Language strategy (auto|da|en)"),
    min_questions: int = typer.Option(None, help="Override minimum questions per document"),
    mid_questions: int = typer.Option(None, help="Override medium questions per document"),
    max_questions: int = typer.Option(None, help="Override max questions per document"),
    model: Optional[str] = typer.Option(None, help="Override chat model for generation"),
):
    """Generate HQE questions for the scraped corpus."""
    cfg = load_config()
    if min_questions is not None:
        cfg.hqe.min_questions = min_questions
    if mid_questions is not None:
        cfg.hqe.mid_questions = mid_questions
    if max_questions is not None:
        cfg.hqe.max_questions = max_questions
    if model:
        cfg.openai.chat_model = model

    budget = Budget(
        small=cfg.hqe.min_questions,
        medium=cfg.hqe.mid_questions,
        large=cfg.hqe.max_questions,
    )

    corpus_obj = load_corpus(corpus)
    records = build_hqe_sidecar(
        corpus_obj,
        budget=budget,
        lang_strategy=lang_strategy,
        cfg=cfg.hqe,
        openai_cfg=cfg.openai,
    )
    write_jsonl(out, records)


@app.command("embed-hqe")
def cli_embed_hqe(
    hqe_path: Path = typer.Argument(..., exists=True, readable=True, help="HQE JSONL path"),
    out_prefix: Path = typer.Argument(..., help="Output prefix for embeddings/meta"),
    batch_size: int = typer.Option(128, min=1, help="Embedding batch size"),
    model: Optional[str] = typer.Option(None, help="Override embedding model"),
):
    """Embed HQE questions and write NumPy + metadata files."""
    cfg = load_config()
    records = read_hqe_jsonl(hqe_path)
    if model:
        cfg.openai.embedding_model = model
    artifacts = build_embeddings(records, openai_cfg=cfg.openai, batch_size=batch_size)
    write_artifacts(artifacts, out_prefix)


@app.command("qa-console")
def cli_qa_console(
    emb: Path = typer.Argument(..., exists=True, readable=True, help="Embeddings .npy path"),
    meta: Path = typer.Argument(..., exists=True, readable=True, help=".meta.json path"),
    corpus: Path = typer.Argument(..., exists=True, readable=True, help="Corpus JSON path"),
    topk: int = typer.Option(4, help="Number of entries to retrieve"),
    min_score: float = typer.Option(-1.0, help="Optional minimum similarity score"),
    chat_model: Optional[str] = typer.Option(None, help="Override chat model"),
):
    """Interactive QA session in the terminal."""
    cfg = load_config()
    if chat_model:
        cfg.openai.chat_model = chat_model
    matrix = load_np_embeddings(emb)
    meta_rows = load_meta_file(meta)
    lut = build_section_lookup(load_corpus_json(corpus))
    cfg.openai.embedding_model = _infer_embed_model(matrix.shape[1], cfg.openai.embedding_model)

    print(f"[INIT] embeddings={matrix.shape} | meta={len(meta_rows)}")
    while True:
        try:
            question = input("\nIndtast dit spørgsmål (tom linje for stop): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break

        q_vec = embed_query(question, cfg=cfg.openai)
        if len(q_vec) != matrix.shape[1]:
            print("Dimension mismatch mellem query og embeddings.", file=sys.stderr)
            continue
        idx, scores = topk_dot(matrix, q_vec, topk)
        chunks = []
        shown_scores = []
        for i, score in zip(idx, scores):
            if score < min_score:
                continue
            md = meta_rows[i]
            key = (md.get("doc_id"), md.get("section_index"))
            sec = lut.get(key)
            if not sec:
                sec = {
                    "title": md.get("page_title") or "(ukendt titel)",
                    "link": md.get("source") or "",
                    "heading": md.get("section_heading") or "(ingen overskrift)",
                    "content": md.get("aggregated_content")
                    or f"(Kun HQE-spørgsmål)\n{md.get('question','')}",
                }
            chunks.append(sec)
            shown_scores.append(float(score))

        answer, shown_sources = answer_query(
            prompt=question,
            context_chunks=chunks,
            scores=shown_scores,
            chat_model=cfg.openai.chat_model,
            per_section_chars=900,
            total_chars=2500,
            openai_cfg=cfg.openai,
        )
        print("\n- SVAR -")
        print(answer)
        print("\n- KILDER -")
        for idx, (sec, score) in enumerate(shown_sources, 1):
            print(f"[{idx}] {sec.get('title')} - {sec.get('heading')} <{sec.get('link')}> <{score:.4f}>")


def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    setup_logging()
    app()


if __name__ == "__main__":
    main()
