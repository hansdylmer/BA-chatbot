from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from playwright.async_api import async_playwright, Page

from ..config import PathsConfig, ScrapeConfig
from ..log_utils import setup_logging
from ..models import Corpus, Document, Section


NOISE_PHRASES = {
    "this is a modal window",
    "beginning of dialog window",
    "escape will cancel",
    "subtitles off",
    "opens subtitles settings dialog",
    "end of dialog window",
    "fandt du ikke, hvad du ledte efter",
    "genveje",
}


@dataclass(slots=True)
class ScrapedSection:
    heading: Optional[str]
    content: str


class ContentScraper:
    def __init__(self, *, date: str, paths: PathsConfig, cfg: ScrapeConfig) -> None:
        setup_logging()
        self.date = date
        self.cfg = cfg
        self.paths = paths
        self.links_path = paths.dated_subdir(date) / f"all_relative_links_{date}_async.txt"
        if not self.links_path.exists():
            raise FileNotFoundError(f"Link list file not found: {self.links_path}")
        self.base_url = cfg.base_url.rstrip("/")
        self.output_path = paths.dated_subdir(date) / f"links_content_{date}_async.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.links_path.open("r", encoding="utf-8") as f:
            self.links = [line.strip() for line in f if line.strip()]

    async def scrape(self) -> Corpus:
        semaphore = asyncio.Semaphore(self.cfg.max_concurrency)
        results: List[Document] = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.cfg.headless)

            async def worker(path: str) -> Optional[Document]:
                async with semaphore:
                    return await self._scrape_link(browser, path)

            tasks = [worker(path) for path in self.links]
            pages = await asyncio.gather(*tasks)
            for doc in pages:
                if doc:
                    results.append(doc)

            await browser.close()

        corpus = Corpus(root=results)
        self._write_corpus(corpus)
        return corpus

    async def _scrape_link(self, browser, path: str) -> Optional[Document]:
        full_url = self.base_url + path
        logging.info("Scraping %s", full_url)
        page = await browser.new_page()
        try:
            await page.goto(full_url, timeout=60000)
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(1)
            await self._expand_dropdowns(page)
            sections = await self._extract_sections(page)
            title = await self._extract_title(page)

            if sections and title != "(no title)":
                return Document(
                    link=f"{self.base_url}{path}",
                    title=sanitize_text(title),
                    sections=[
                        Section(
                            heading=sanitize_text(s.heading) if s.heading else "(ingen overskrift)",
                            content=sanitize_text(s.content),
                        )
                        for s in sections
                        if s.content.strip()
                    ],
                )
        except Exception as exc:
            logging.warning("Failed to load %s: %s", full_url, exc)
        finally:
            await page.close()
        return None

    async def _expand_dropdowns(self, page: Page) -> None:
        await asyncio.sleep(1)
        section_headers = await page.query_selector_all("div.section-header")
        for header in section_headers:
            try:
                await header.click()
                await asyncio.sleep(0.5)
            except Exception:
                continue

    async def _extract_sections(self, page: Page) -> List[ScrapedSection]:
        containers = [
            await page.query_selector("div.span-9 > div.web-page"),
            await page.query_selector("div.span-9 > div.plh-bottom"),
        ]
        elements = []
        for container in containers:
            if container:
                elements += await container.query_selector_all("h1, h2, h3, p, li")

        sections: List[ScrapedSection] = []
        current = ScrapedSection(heading=None, content="")
        content_parts: List[str] = []

        for el in elements:
            is_visible = await el.evaluate("el => el.offsetHeight > 0 && el.offsetWidth > 0")
            if not is_visible:
                continue
            tag = (await el.evaluate("el => el.tagName")).lower()
            raw_text = await el.inner_text()
            text = sanitize_text(raw_text)
            if not text or any(noise in text.lower() for noise in NOISE_PHRASES):
                continue

            if tag in {"h1", "h2", "h3"}:
                if content_parts:
                    combined = sanitize_text(" ".join(content_parts))
                    if combined:
                        sections.append(ScrapedSection(heading=current.heading, content=combined))
                current = ScrapedSection(heading=text, content="")
                content_parts = []
            else:
                content_parts.append(text)

        if content_parts:
            combined = sanitize_text(" ".join(content_parts))
            if combined:
                sections.append(ScrapedSection(heading=current.heading, content=combined))

        return [s for s in sections if s.content]

    async def _extract_title(self, page: Page) -> str:
        try:
            selector = "#ContentPlaceHolderDefault_toolSection_breadcrumb_ctrl > div > ul > li.active > span"
            title_el = await page.query_selector(selector)
            return sanitize_text(await title_el.inner_text()) if title_el else "(no title)"
        except Exception:
            return "(no title)"

    def _write_corpus(self, corpus: Corpus) -> None:
        data = [doc.model_dump(mode="json") for doc in corpus.documents()]
        self.output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logging.info("Saved scraped content → %s", self.output_path)


async def scrape_content(date: str, paths: PathsConfig, cfg: ScrapeConfig) -> Corpus:
    scraper = ContentScraper(date=date, paths=paths, cfg=cfg)
    return await scraper.scrape()


def sanitize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = text.replace("\u00a0", " ").strip()
    # Remove lone surrogate code points that json.dumps cannot encode
    encoded = normalized.encode("utf-8", "ignore")
    return encoded.decode("utf-8", "ignore")
