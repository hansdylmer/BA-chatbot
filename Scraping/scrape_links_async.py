#!/usr/bin/env python3
"""
Thin wrapper to run the structured link scraper from the legacy entry-point.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from su_bot.config import load_config
from su_bot.scraping.links import scrape_links


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Scrape SU.dk for internal links.")
    parser.add_argument("--date", default=None, help="ISO date to use for output folder (default: today)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config()
    date = args.date or time.strftime("%Y-%m-%d")
    asyncio.run(scrape_links(date, cfg.paths, cfg.scrape))
    return 0


if __name__ == "__main__":
    sys.exit(main())

