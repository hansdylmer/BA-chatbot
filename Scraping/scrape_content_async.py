#!/usr/bin/env python3
"""
Legacy entry-point that delegates to the structured content scraper.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from su_bot.config import load_config
from su_bot.scraping.content import scrape_content


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Scrape SU content for previously discovered links.")
    parser.add_argument("--date", default=None, help="ISO date matching link scrape (default: today)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config()
    date = args.date or time.strftime("%Y-%m-%d")
    asyncio.run(scrape_content(date, cfg.paths, cfg.scrape))
    return 0


if __name__ == "__main__":
    sys.exit(main())

