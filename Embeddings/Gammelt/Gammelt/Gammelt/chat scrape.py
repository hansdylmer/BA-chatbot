from playwright.sync_api import sync_playwright
import os
import time

class PageLinkScrape:
    def __init__(self, base_url="https://www.su.dk", output_path="./data/all_relative_links.txt"):
        self.base_url = base_url.rstrip("/")
        self.output_path = output_path
        self.visited = set()
        self.to_visit = set(["/"])  # start fra forsiden
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def run(self):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()

            while self.to_visit:
                path = self.to_visit.pop()
                if path in self.visited:
                    continue
                self.visited.add(path)

                full_url = self.base_url + path
                print(f"🌐 Visiting: {full_url}")
                try:
                    page.goto(full_url)
                    page.wait_for_load_state("networkidle")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"❌ Failed to load {full_url}: {e}")
                    continue

                link_elements = page.query_selector_all("a")
                self.extract_links(link_elements)

            browser.close()
            self.save_links_to_file()

    def extract_links(self, elements):
        for el in elements:
            try:
                href = el.get_attribute("href")
                if self.should_keep(href):
                    clean_href = href.split("#")[0].rstrip("/")  # fjern anchors og trailing slash
                    if clean_href not in self.visited and clean_href not in self.to_visit:
                        self.to_visit.add(clean_href)
                        print(f"➕ Found new link: {clean_href}")
            except Exception:
                continue

    def should_keep(self, href):
        if not href or not href.startswith("/"):
            return False
        if (
            href.startswith("/english") or
            href.startswith("/nyheder") or
            href.startswith("#")
        ):
            return False
        return True

    def save_links_to_file(self):
        print(f"💾 Saving {len(self.visited)} links to file: {self.output_path}")
        with open(self.output_path, "w", encoding="utf-8") as f:
            for link in sorted(self.visited):
                f.write(link + "\n")
        print("✅ Links saved successfully.")
        
PageLinkScrape().run()