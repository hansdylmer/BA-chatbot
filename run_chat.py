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
from su_bot.rag.bm25 import build_bm25_index_from_meta, reciprocal_rank_fusion


cfg = load_config()

def _posix_str(p: Path | str) -> str:
    return Path(p).as_posix()


def _normalize_input_path(p: str) -> Path:
    # Convert Windows-style backslashes to a Path usable on POSIX
    return Path(p.replace("\\", "/"))


def find_default_artifacts(base_dir: Path) -> Tuple[str, str, str]:
    """Pick the newest emb/meta/corpus artifacts under the configured data dir."""
    def newest(pattern: str) -> str:
        try:
            return _posix_str(max(base_dir.rglob(pattern), key=lambda p: p.stat().st_mtime))
        except (ValueError, FileNotFoundError):
            return ""

    emb_path = newest("*.embeddings.npy")
    meta_path = newest("*.meta.json")
    corpus_path = newest("links_content*.json") or newest("*.json")
    return (emb_path, meta_path, corpus_path)


default_emb, default_meta, default_corpus = find_default_artifacts(Path(cfg.paths.data_dir))
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
    return _posix_str(meta), _posix_str(corpus)


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


def _is_clarification(text: str) -> bool:
    pref = (text or "").lstrip().lower()
    return pref.startswith("clarification question:") or pref.startswith("opklarende spørgsmål:") or pref.startswith("opklarende spoergsmaal:")


st.set_page_config(page_title="SU chatbot", page_icon="🤖", layout="wide")
st.title("SU Chatbot")

with st.sidebar:
    st.header("⚙️ Indstillinger")
    emb_path = st.text_input("Embeddings (.npy)", value=default_emb)
    derived_meta, derived_corpus = companions_from_emb(emb_path or default_emb)
    meta_path = st.text_input("Meta (.json)", value=derived_meta or default_meta)
    corpus_path = st.text_input("Corpus (.json)", value=default_corpus or derived_corpus)
    topk = st.slider("Top-K kontekst", min_value=1, max_value=50, value=10)
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
    retrieval_mode = st.selectbox(
        "Retriever",
        ["Hybrid (RRF)", "Embeddings only", "BM25 only"],
        index=0,
        help="Hybrid kombinerer embeddings og BM25 med reciprocal-rank-fusion.",
    )
    load_btn = st.button("🔄 Indlæs / Opdater artefakter")

st.session_state.retrieval_mode = retrieval_mode

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

if load_btn or "emb" not in st.session_state:
    try:
        emb_path_norm = _normalize_input_path(emb_path or "")
        meta_path_norm = _normalize_input_path(meta_path or "")
        corpus_path_norm = _normalize_input_path(corpus_path or "")

        for p in [emb_path_norm, meta_path_norm, corpus_path_norm]:
            if not p or not Path(p).exists():
                st.error(f"Filen findes ikke: {p or '(tom)'}")
                st.stop()

        t0 = time.perf_counter()
        matrix = load_embeddings_cached(str(emb_path_norm))
        meta_rows = load_meta_cached(str(meta_path_norm))
        lut = build_lookup_cached(str(corpus_path_norm))
        bm25_index = build_bm25_index_from_meta(meta_rows)
        load_secs = time.perf_counter() - t0

        st.session_state.emb = matrix
        st.session_state.meta = meta_rows
        st.session_state.lut = lut
        st.session_state.bm25 = bm25_index
        st.session_state.emb_dim = int(matrix.shape[1])
        st.session_state.retrieval_mode = retrieval_mode

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
    pending = st.session_state.pending_clarification
    if pending:
        combined_prompt = f"Original question: {pending['original_question']}\nUser clarification: {prompt}"
        base_question = pending["original_question"]
    else:
        combined_prompt = prompt
        base_question = prompt

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        matrix = st.session_state.emb
        meta_rows = st.session_state.meta
        lut = st.session_state.lut
        bm25_index = st.session_state.get("bm25")
        current_retriever = st.session_state.get("retrieval_mode", retrieval_mode)

        vec_idx = np.array([], dtype=int)
        vec_scores = np.array([], dtype=np.float32)
        bm25_idx = np.array([], dtype=int)
        bm25_scores = np.array([], dtype=np.float32)

        if current_retriever != "BM25 only":
            cfg.openai.embedding_model = st.session_state.embed_model
            q_vec = embed_query(combined_prompt, cfg=cfg.openai, client=client)
            if len(q_vec) != st.session_state.emb_dim:
                st.error(f"Dimension mismatch: query={len(q_vec)} vs matrix={st.session_state.emb_dim}.")
                st.stop()
            vec_idx, vec_scores = topk_dot(matrix, q_vec, k=topk)

        if bm25_index and current_retriever != "Embeddings only":
            bm25_idx, bm25_scores = bm25_index.search(combined_prompt, k=topk)

        if current_retriever == "Embeddings only":
            final_idx, final_scores = vec_idx, vec_scores
        elif current_retriever == "BM25 only":
            final_idx, final_scores = bm25_idx, bm25_scores
        else:
            rank_lists = [arr for arr in [vec_idx, bm25_idx] if arr.size > 0]
            final_idx, final_scores = reciprocal_rank_fusion(rank_lists, k=topk)

        components: Dict[int, Dict[str, float]] = {}
        for i, s in zip(vec_idx, vec_scores):
            components.setdefault(int(i), {})["emb"] = float(s)
        for i, s in zip(bm25_idx, bm25_scores):
            components.setdefault(int(i), {})["bm25"] = float(s)

        picked: List[Dict[str, str]] = []
        shown_sources: List[Tuple[Dict[str, str], float]] = []
        shown_meta_idx: List[int] = []
        for j, score in zip(final_idx, final_scores):
            if current_retriever == "Embeddings only" and score < min_score:
                continue
            meta_row = meta_rows[int(j)]
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
            shown_meta_idx.append(int(j))

        context = make_context(picked, per_section_chars=per_section_chars, total_chars=total_chars)
        instructions = build_instructions(combined_prompt, context)
        resp = client.responses.create(model=chat_model, input=instructions, stream=False, store=False, temperature=0.8)
        answer = (resp.output_text or "").strip()
        is_clar = _is_clarification(answer)
        st.session_state.pending_clarification = {"original_question": base_question} if is_clar else None

        st.markdown(answer)
        if not is_clar:
            with st.expander("Kilder (Top-K)"):
                for idx_src, ((sec, score), meta_idx) in enumerate(zip(shown_sources, shown_meta_idx), 1):
                    comp = components.get(meta_idx, {})
                    if current_retriever == "Hybrid (RRF)":
                        parts = []
                        if "emb" in comp:
                            parts.append(f"emb={comp['emb']:.4f}")
                        if "bm25" in comp:
                            parts.append(f"bm25={comp['bm25']:.4f}")
                        parts_text = ", ".join(parts)
                        detail = f"rrf={score:.4f}" if not parts_text else f"rrf={score:.4f} ({parts_text})"
                    elif current_retriever == "BM25 only":
                        detail = f"bm25={score:.4f}"
                    else:
                        detail = f"emb={score:.4f}"
                    st.markdown(
                        f"**[{idx_src}] {sec.get('title')} - {sec.get('heading')}**  \n"
                        f"<{sec.get('link')}>  \n"
                        f"**score:** {detail}",
                        unsafe_allow_html=True,
                    )

        took = time.perf_counter() - t0
        caption_parts = [
            f"Færdig på {took:.2f}s",
            f"model: {chat_model}",
            f"embed: {st.session_state.embed_model}",
            f"retriever: {current_retriever}",
        ]
        if is_clar:
            caption_parts.append("afventer opklaring")
        st.caption(" · ".join(caption_parts))

        st.session_state.messages.append({"role": "assistant", "content": answer})
