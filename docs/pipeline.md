# SU BOT Pipeline Overview

This repository now ships as an installable package via `pyproject.toml` with a single command-line surface: `su-bot`. The CLI wires together scraping, HQE generation, embedding, and interactive QA on top of OpenAI models.

## Quick Start

```
pip install -e .
su-bot scrape-links --date 2025-09-17
su-bot scrape-content --date 2025-09-17
su-bot hqe-generate Scraping/data/2025-09-17/links_content_2025-09-17_async.json Scraping/data/2025-09-17/links_content_2025-09-17_async.hqe.v1.jsonl --lang-strategy auto
su-bot embed-hqe Scraping/data/2025-09-17/links_content_2025-09-17_async.hqe.v1.jsonl Scraping/data/2025-09-17/links_content_2025-09-17_async.hqe.v1
su-bot qa-console Scraping/data/2025-09-17/links_content_2025-09-17_async.hqe.v1.embeddings.npy Scraping/data/2025-09-17/links_content_2025-09-17_async.hqe.v1.meta.json Scraping/data/2025-09-17/links_content_2025-09-17_async.json
```

All OpenAI calls rely on the standard `OPENAI_API_KEY` environment variable. You can override model names inline with `--model`.

HQE generation now operates one record per URL: all sections from a page are merged (within the `max_chars` budget) before we ask the LLM for questions, so metadata refers to `(doc_id, -1)` when you embed or query later.

## Key Modules

- `src/su_bot/scraping/` – async link + content scraping built on Playwright.
- `src/su_bot/hqe/` - question generation helpers, quality/dedup logic.
- `src/su_bot/embeddings/` – utilities for flattening HQE questions and writing `.npy` artefacts.
- `src/su_bot/rag/` – NumPy search helpers, prompts, and chat orchestration.
- `src/su_bot/config.py` – central configuration dataclasses (paths, models, scraping defaults).
- `src/su_bot/cli.py` – Typer CLI exposing the end-to-end workflow (`su-bot ...`).

Legacy wruth; use the `su-bot` CLI for scraping, HQE generation, embeddings, and console QA. The Streamlit UI (`Scraping/app_streamlit_localqa.py`) still consumes the same saved embeddings if you prefer a browser-based front-end.rapper scripts under `Scraping/` have been removed to keep a single source of t
