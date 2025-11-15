#!/usr/bin/env python3
"""Console RAG client backed by ChromaDB embeddings."""

from __future__ import annotations

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from su_bot.config import load_config
from su_bot.openai_client import get_openai_client
from su_bot.rag.prompts import build_instructions


def main() -> int:
    cfg = load_config()
    client = PersistentClient(path="./Embeddings/vector_database/seneste")
    collection = client.get_collection("SU_vector_database")
    encoder = SentenceTransformer("intfloat/multilingual-e5-small")

    query = input("Indtast dit spørgsmål: ").strip()
    if not query:
        return 0

    embedded_query = encoder.encode(query)
    results = collection.query(
        query_embeddings=[embedded_query.tolist()],
        n_results=3,
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    docs = results["documents"][0]
    distances = results["distances"][0]
    context_chunks = "\n".join(f"- {doc}" for doc in docs)
    instructions = build_instructions(query, context_chunks)

    llm = get_openai_client(cfg.openai)
    response = llm.responses.create(
        model=cfg.openai.chat_model,
        input=instructions,
        stream=False,
        store=False,
    )
    print(response.output_text)

    print("\n- KILDER -")
    for idx, (doc, score) in enumerate(zip(docs, distances), 1):
        print(f"[{idx}] {doc} <{score:.4f}>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

