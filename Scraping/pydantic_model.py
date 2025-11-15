from __future__ import annotations

import json
from pathlib import Path
from typing import List

from su_bot.models import Corpus, Document, Section


def load_corpus(path: str | Path) -> Corpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs: List[Document] = [Document(**doc) for doc in data]
    return Corpus(root=docs)


if __name__ == "__main__":
    sample_path = Path("Scraping/data/2025-09-04/links_content_2025-09-04_async.json")
    corpus = load_corpus(sample_path)
    print(len(corpus), "dokumenter")
    first_doc = corpus.documents()[0]
    print(first_doc.model_dump())

