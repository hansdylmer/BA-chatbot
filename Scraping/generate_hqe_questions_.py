#!/usr/bin/env python3
"""
Backward-compatible wrapper that delegates HQE generation to the new su_bot package.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from su_bot.config import load_config
from su_bot.hqe.generator import build_hqe_sidecar, load_corpus
from su_bot.cli import write_jsonl
from su_bot.models import Budget
from su_bot.log_utils import setup_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate HQE sidecar JSONL (questions only, no embeddings)")
    parser.add_argument("--corpus", required=True, help="Path to scraped corpus JSON (list of documents)")
    parser.add_argument("--out", required=True, help="Path to write JSONL with HQE records")
    parser.add_argument("--lang-strategy", default="da", choices=["auto", "da", "en"], help="Language for HQE questions")
    parser.add_argument("--min", type=int, default=None, help="Min questions for short documents")
    parser.add_argument("--mid", type=int, default=None, help="Questions for medium documents")
    parser.add_argument("--max", type=int, default=None, help="Questions for long documents")
    return parser.parse_args(argv)


def main(argv=None):
    setup_logging()
    cfg = load_config()
    args = parse_args(argv)

    if args.min is not None:
        cfg.hqe.min_questions = args.min
    if args.mid is not None:
        cfg.hqe.mid_questions = args.mid
    if args.max is not None:
        cfg.hqe.max_questions = args.max

    budget = Budget(
        small=cfg.hqe.min_questions,
        medium=cfg.hqe.mid_questions,
        large=cfg.hqe.max_questions,
    )

    corpus = load_corpus(Path(args.corpus))
    logging.info("[HQE] Loaded corpus: %s documents", len(corpus))
    records = build_hqe_sidecar(
        corpus,
        budget=budget,
        lang_strategy=args.lang_strategy,
        cfg=cfg.hqe,
        openai_cfg=cfg.openai,
    )
    write_jsonl(Path(args.out), records)
    logging.info("[HQE] Wrote %s document records -> %s", len(records), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
