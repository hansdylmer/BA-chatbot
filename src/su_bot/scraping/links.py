from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set

from playwright.async_api import async_playwright

from ..config import PathsConfig, ScrapeConfig
from ..log_utils import setup_logging


@dataclass(slots=True)
class LinkScrapeResult:
    date: str
    links: Set[str]
    output_path: Path


class LinkScraper:
    def __init__(self, *, date: str, paths: PathsConfig, cfg: ScrapeConfig) -> None:
        setup_logging()
        self.date = date
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.output_dir = paths.dated_subdir(date)
        self.output_path = self.output_dir / f"links_{date}.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visited: Set[str] = set()
        self.links: Set[str] = set()
        if self.output_path.exists():
            self.links.update(self.output_path.read_text(encoding="utf-8").splitlines())

    async def crawl(self) -> LinkScrapeResult:
        to_visit: asyncio.Queue[str] = asyncio.Queue()
        await to_visit.put("/")

        semaphore = asyncio.Semaphore(self.cfg.max_concurrency)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.cfg.headless)
            context = await browser.new_context()

            async def worker() -> None:
                while True:
                    path = await to_visit.get()
                    if path in self.visited:
                        to_visit.task_done()
                        continue
                    self.visited.add(path)
                    async with semaphore:
                        await self._process_path(context, path, to_visit)
                    to_visit.task_done()

            workers = [asyncio.create_task(worker()) for _ in range(self.cfg.max_concurrency)]
            await to_visit.join()
            for w in workers:
                w.cancel()
            await browser.close()

        self._write_links()
        return LinkScrapeResult(date=self.date, links=self.links, output_path=self.output_path)

    async def _process_path(self, context, path: str, queue: asyncio.Queue[str]) -> None:
        full_url = self.base_url + path
        logging.info("Navigating to %s", full_url)
        try:
            page = await context.new_page()
            await page.goto(full_url)
            await page.wait_for_load_state("load")
            elements = await page.query_selector_all("a")
            hrefs = [await el.get_attribute("href") for el in elements]
            await self._handle_links(hrefs, queue)
            await page.close()
            if self.cfg.crawl_delay:
                await asyncio.sleep(self.cfg.crawl_delay)
        except Exception as exc:
            logging.warning("Error navigating to %s: %s", full_url, exc)

    async def _handle_links(self, hrefs: Iterable[str], queue: asyncio.Queue[str]) -> None:
        new_links = set()
        for href in hrefs:
            if not href:
                continue
            if not href.startswith("/"):
                continue
            if any(
                [
                    href.startswith("/nyheder"),
                    href.startswith("https"),
                    "localLink" in href,
                    href.endswith((".pdf", ".png")),
                    "#" in href,
                ]
            ):
                continue
            if href not in self.links:
                new_links.add(href)

        if not new_links:
            return

        self.links.update(new_links)
        for link in new_links:
            await queue.put(link)

    def _write_links(self) -> None:
        lines = sorted(self.links)
        self.output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.info("Saved %s links → %s", len(lines), self.output_path)


async def scrape_links(date: str, paths: PathsConfig, cfg: ScrapeConfig) -> LinkScrapeResult:
    scraper = LinkScraper(date=date, paths=paths, cfg=cfg)
    return await scraper.crawl()

