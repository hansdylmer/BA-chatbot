#!/usr/bin/env python3
"""
Wrapper around su_bot.embeddings.writer so existing scripts keep working.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from su_bot.config import load_config
from su_bot.embeddings.writer import read_hqe_jsonl, build_embeddings, write_artifacts
from su_bot.log_utils import setup_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Embed HQE questions to .npy + .meta.json")
    parser.add_argument("--hqe", required=True, help="Path to HQE JSONL")
    parser.add_argument("--out-prefix", required=True, help="Output prefix (without extension)")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    return parser.parse_args(argv)


def main(argv=None):
    setup_logging()
    cfg = load_config()
    args = parse_args(argv)

    records = read_hqe_jsonl(Path(args.hqe))
    artifacts = build_embeddings(records, openai_cfg=cfg.openai, batch_size=args.batch_size)
    write_artifacts(artifacts, Path(args.out_prefix))
    logging.info("Done embedding HQE questions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

