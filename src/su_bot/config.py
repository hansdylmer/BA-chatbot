from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


DEFAULT_DATA_DIR = Path("data")
DEFAULT_EMBED_DIR = Path("Gammelt")


@dataclass(slots=True)
class PathsConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    embeddings_dir: Path = DEFAULT_EMBED_DIR

    def dated_subdir(self, date_str: str) -> Path:
        return self.data_dir / date_str


@dataclass(slots=True)
class OpenAIConfig:
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        return self.api_key


@dataclass(slots=True)
class ScrapeConfig:
    base_url: str = "https://www.su.dk"
    max_concurrency: int = 8
    crawl_delay: float = 0.0
    headless: bool = True


@dataclass(slots=True)
class HQEConfig:
    prompt_version: str = "hqe-v2"
    min_questions: int = 3
    mid_questions: int = 5
    max_questions: int = 8
    max_chars: int = 2500
    dedup_jaccard: float = 0.80


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    scrape: ScrapeConfig = field(default_factory=ScrapeConfig)
    hqe: HQEConfig = field(default_factory=HQEConfig)


def load_config() -> AppConfig:
    """Central place to build configuration – extend as needed."""
    return AppConfig()

