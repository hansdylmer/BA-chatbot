#!/usr/bin/env python3
"""
Terminal QA wrapper delegating to the shared su_bot CLI implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from su_bot.cli import cli_qa_console


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local QA using HQE embeddings (.npy) + OpenAI")
    parser.add_argument("--emb", required=True, help="Path to embeddings .npy")
    parser.add_argument("--meta", required=True, help="Path to meta .json")
    parser.add_argument("--corpus", required=True, help="Path to original corpus JSON")
    parser.add_argument("--topk", type=int, default=4, help="Top-K sections to include as context")
    parser.add_argument("--min-score", type=float, default=-1.0, help="Optional min dot score to accept (<=1.0)")
    parser.add_argument("--model", default=None, help="Override chat model")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cli_qa_console(
        emb=Path(args.emb),
        meta=Path(args.meta),
        corpus=Path(args.corpus),
        topk=args.topk,
        min_score=args.min_score,
        chat_model=args.model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

