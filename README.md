# SU BOT

## How to Build
- Ensure Python 3.11+ is installed.
- (Recommended) create and activate a virtual environment:
  - Windows PowerShell:
    ```
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
  - macOS/Linux:
    ```
    python -m venv .venv
    source .venv/bin/activate
    ```
- Install the package and dependencies in editable mode:
  ```
  pip install -e .
  ```
- Install Playwright browsers for scraping:
  ```
  python -m playwright install chromium
  ```
- Verify the CLI is available:
  ```
  su-bot --help
  ```

Modular pipeline for scraping SU.dk content, generating hypothetical questions (HQE), embedding them, and serving local RAG-style QA experiences.

## Highlights

- Async Playwright scrapers for link discovery and content extraction.
- HQE generation with OpenAI responses + deterministic quality/dedup layers (one record per URL).
- NumPy-based retrieval utilities with shared prompts and console/Streamlit UIs.
- Typer-powered CLI (`su-bot`) that orchestrates scraping -> HQE -> embeddings -> QA.

## Repository layout
- `src/su_bot/` - Python package (scraping, HQE, embeddings, RAG, CLI entrypoint).
- `docs/pipeline.md` - end-to-end workflow and CLI examples.
- `data/` - dated scrape outputs; used by the CLI defaults.
- `Gammelt/` - legacy experiments and archived scripts (including the old `Scraping/` artifacts).
- `run_chat.py` - optional Streamlit UI on top of saved embeddings.

Legacy one-off scripts have been removed to avoid duplicating the packaged CLI; use `su-bot <command>` for all pipeline steps.

See `docs/pipeline.md` for the day-to-day workflow and CLI examples.
