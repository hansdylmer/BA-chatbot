# run_chat.py
# Streamlit chat UI for local NumPy + OpenAI RAG over HQE embeddings

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

from su_bot.config import load_config
from su_bot.openai_client import get_openai_client
from su_bot.rag.prompts import build_instructions
from su_bot.rag.search import (
    load_embeddings as load_embeddings_np,
    load_meta as load_meta_json,
    load_corpus as load_corpus_json,
    build_section_lookup,
    embed_query,
    topk_dot,
    make_context,
)


cfg = load_config()

def find_default_artifacts(base_dir: Path) -> Tuple[str, str, str]:
    """Pick the newest embedding set under the configured data dir."""
    try:
        candidates = sorted(base_dir.rglob("*.embeddings.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    except FileNotFoundError:
        return "", "", ""

    if not candidates:
        return "", "", ""

    emb_path = candidates[0]
    stem_no_ext = emb_path.name.replace(".embeddings.npy", "")
    meta_path = emb_path.with_name(f"{stem_no_ext}.meta.json")

    # Heuristic to guess the corpus filename from the embedding prefix
    if stem_no_ext.endswith(".hqe.v1"):
        corpus_guess = stem_no_ext.removesuffix(".hqe.v1")
    elif "_embedded_hqe" in stem_no_ext:
        corpus_guess = stem_no_ext.replace("_embedded_hqe", "")
    else:
        corpus_guess = stem_no_ext
    corpus_path = emb_path.with_name(f"{corpus_guess}.json")

    return (str(emb_path), str(meta_path), str(corpus_path))


default_emb, default_meta, default_corpus = find_default_artifacts(Path(cfg.paths.data_dir))

default_emb = str(Path(cfg.paths.data_dir) / "2025-09-17" / "2025-09-17-HQE.v1.embeddings.npy")
default_meta = str(Path(cfg.paths.data_dir) / "2025-09-17" / "2025-09-17-HQE.v1.meta.json")
default_corpus = str(Path(cfg.paths.data_dir) / "2025-09-17" / "links_content_2025-09-17_async.json")
def companions_from_emb(emb_path: str) -> Tuple[str, str]:
    """Derive matching meta/corpus paths from an embeddings path."""
    if not emb_path:
        return "", ""
    p = Path(emb_path)
    stem_no_ext = p.name.replace(".embeddings.npy", "")
    meta = p.with_name(f"{stem_no_ext}.meta.json")
    if stem_no_ext.endswith(".hqe.v1"):
        corpus_guess = stem_no_ext.removesuffix(".hqe.v1")
    elif "_embedded_hqe" in stem_no_ext:
        corpus_guess = stem_no_ext.replace("_embedded_hqe", "")
    else:
        corpus_guess = stem_no_ext
    corpus = p.with_name(f"{corpus_guess}.json")
    return str(meta), str(corpus)


def infer_embed_model_from_dim(dim: int) -> str:
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    return cfg.openai.embedding_model


@st.cache_resource(show_spinner=True)
def load_embeddings_cached(path: str) -> np.ndarray:
    return load_embeddings_np(path)


@st.cache_resource(show_spinner=True)
def load_meta_cached(path: str) -> List[Dict]:
    return load_meta_json(path)


@st.cache_resource(show_spinner=True)
def load_corpus_cached(path: str) -> List[Dict]:
    return load_corpus_json(path)


@st.cache_resource(show_spinner=True)
def build_lookup_cached(corpus_path: str) -> Dict[Tuple[str, int], Dict[str, str]]:
    corpus = load_corpus_cached(corpus_path)
    return build_section_lookup(corpus)


@st.cache_resource(show_spinner=True)
def init_openai_client():
    return get_openai_client(cfg.openai)


st.set_page_config(page_title="SU chatbot", page_icon="🤖", layout="wide")
st.title("SU Chatbot")

with st.sidebar:
    st.header("⚙️ Indstillinger")
    emb_path = st.text_input("Embeddings (.npy)", value=default_emb)
    derived_meta, derived_corpus = companions_from_emb(emb_path or default_emb)
    meta_path = st.text_input("Meta (.json)", value=derived_meta or default_meta)
    corpus_path = st.text_input("Corpus (.json)", value=default_corpus or derived_corpus)
    topk = st.slider("Top-K kontekst", min_value=1, max_value=10, value=4)
    min_score = st.slider("Min. score", min_value=-1.0, max_value=1.0, value=-1.0, step=0.05)
    per_section_chars = st.slider("Maks tegn pr. sektion", min_value=200, max_value=2000, value=900, step=50)
    total_chars = st.slider("Maks samlet kontekst", min_value=500, max_value=4000, value=2500, step=100)
    chat_model = st.selectbox(
        "Chat-model",
        [
            cfg.openai.chat_model,
            "gpt-4.1-mini",
            "gpt-4.1",
        ],
    )
    embed_model_mode = st.radio(
        "Embedding-model",
        ["Auto (fra dimension)", "text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )
    load_btn = st.button("🔄 Indlæs / Opdater artefakter")

if "messages" not in st.session_state:
    st.session_state.messages = []

if load_btn or "emb" not in st.session_state:
    try:
        for p in [emb_path, meta_path, corpus_path]:
            if not p or not Path(p).exists():
                st.error(f"Filen findes ikke: {p or '(tom)'}")
                st.stop()

        t0 = time.perf_counter()
        matrix = load_embeddings_cached(emb_path)
        meta_rows = load_meta_cached(meta_path)
        lut = build_lookup_cached(corpus_path)
        load_secs = time.perf_counter() - t0

        st.session_state.emb = matrix
        st.session_state.meta = meta_rows
        st.session_state.lut = lut
        st.session_state.emb_dim = int(matrix.shape[1])

        if embed_model_mode.startswith("Auto"):
            st.session_state.embed_model = infer_embed_model_from_dim(st.session_state.emb_dim)
        else:
            st.session_state.embed_model = embed_model_mode

        st.success(
            f"Indlæst: embeddings={matrix.shape}, meta={len(meta_rows)} på {load_secs:.2f}s "
            f"- embed model: {st.session_state.embed_model}"
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

if "emb" not in st.session_state:
    st.info("Udfyld stierne i venstre side og klik **Indlæs**.")
    st.stop()

try:
    client = init_openai_client()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Skriv dit spørgsmål om SU.")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        matrix = st.session_state.emb
        meta_rows = st.session_state.meta
        lut = st.session_state.lut

        cfg.openai.embedding_model = st.session_state.embed_model
        q_vec = embed_query(prompt, cfg=cfg.openai, client=client)
        if len(q_vec) != st.session_state.emb_dim:
            st.error(f"Dimension mismatch: query={len(q_vec)} vs matrix={st.session_state.emb_dim}.")
            st.stop()

        idx, scores = topk_dot(matrix, q_vec, k=topk)

        picked: List[Dict[str, str]] = []
        shown_sources: List[Tuple[Dict[str, str], float]] = []
        for j, score in zip(idx, scores):
            if score < min_score:
                continue
            meta_row = meta_rows[j]
            key = (meta_row.get("doc_id"), meta_row.get("section_index"))
            sec = lut.get(key)
            if not sec:
                sec = {
                    "title": meta_row.get("page_title") or "(ukendt titel)",
                    "link": meta_row.get("source") or "",
                    "heading": meta_row.get("section_heading") or "(ingen overskrift)",
                    "content": meta_row.get("aggregated_content")
                    or f"(Kun HQE-spørgsmål)\n{meta_row.get('question','')}",
                }
            picked.append(sec)
            shown_sources.append((sec, float(score)))

        context = make_context(picked, per_section_chars=per_section_chars, total_chars=total_chars)
        instructions = build_instructions(prompt, context)
        resp = client.responses.create(model=chat_model, input=instructions, stream=False, store=False, temperature=0.8)
        answer = (resp.output_text or "").strip()

        st.markdown(answer)
        with st.expander("Kilder (Top-K)"):
            for idx_src, (sec, score) in enumerate(shown_sources, 1):
                st.markdown(
                    f"**[{idx_src}] {sec.get('title')} - {sec.get('heading')}**  \n"
                    f"<{sec.get('link')}>  \n"
                    f"**score:** {score:.4f}",
                    unsafe_allow_html=True,
                )

        took = time.perf_counter() - t0
        st.caption(f"Færdig på {took:.2f}s · model: {chat_model} · embed: {st.session_state.embed_model}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
