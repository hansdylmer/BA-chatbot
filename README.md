# SU BOT

Modular pipeline for scraping SU.dk content, generating hypothetical questions (HQE), embedding them, and serving local RAG-style QA experiences.

## Highlights

- Async Playwright scrapers for link discovery and content extraction.
- HQE generation with OpenAI responses + deterministic quality/dedup layers (one record per URL).
- NumPy-based retrieval utilities with shared prompts and console/Streamlit UIs.
- Typer-powered CLI (`su-bot`) that orchestrates scraping -> HQE -> embeddings -> QA.

See `docs/pipeline.md` for the day-to-day workflow and CLI examples.
