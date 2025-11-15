from playwright.sync_api import sync_playwright
import time
import json

### To-do:
    # Undersøg om alt indhold fra alle undersider er blevet indsamlet korrekt.
    # Undersøg om der mangler at blive scrabet indhold fra sub-subsites, herunder sørg for at gemme undersøgte links, til at finde nye sider
    # Generaliser til andre sektioner på su.dk, eksempelvis SU-lån. Lav run() funktion med argument til at specificere sektion
    
    
def expand_dropdowns(page):
    """This function expands all dropdowns on the page."""
    print("🔽 Expanding all dropdowns...")
    time.sleep(1)  # Wait for page to settle
    # Find all section headers and click them to expand
    section_headers = page.query_selector_all("div.section-header")
    for header in section_headers:
        try:
            header.click()
            time.sleep(0.5)  # Allow time for dropdown to expand
        except Exception as e:
            print(f"⚠️ Could not click header: {header.inner_text()} — {e}")

def run():
    with sync_playwright() as p:
        # Launch browser and open page
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.su.dk/stoette-til-foraeldre") # Main page for støtte til forældre
        print("🌐 Navigated to main page.")
        page.wait_for_load_state("networkidle") # Wait for the page to load completely
        time.sleep(1)
        expand_dropdowns(page) # Expand all dropdowns if any

        # Retrieve all links for "støtte til forældre" sections. This should get all subsections of the main page.
        print("🔗 Retrieving links for 'støtte til forældre' sections...")
        links = []
        elements = page.query_selector_all("div.text > h2 > a") # Finder containers med links
        for el in elements:
            href = el.get_attribute("href") # Finder link
            if href:
                links.append(href)
        print("Found links:")
        for link in links:
            print(link)
             
        # Save links and their content to a JSON file
        data = []

        ### Visiting each link and scraping content
        print("🔍 Visiting each link to scrape content of all subsites...")
        for link in links:
            print(f"🔗 Opening link: {link}")
            page.goto("https://www.su.dk" + link)
            time.sleep(1)
            expand_dropdowns(page)

            # Find alle relevante DOM-elementer i den centrale tekst
            content_containers = [page.query_selector("div.span-9 > div.web-page"),
                                  page.query_selector("div.span-9 > div.plh-bottom")]
            if not content_containers:
                print(f"⚠️ No content found on {link}")
                continue

            # Find alle underelementer som h1, h2, h3, p, li osv.
            elements = []
            for el in content_containers:
                if el is None:
                    continue
                elements.extend(el.query_selector_all("h1, h2, h3, p, li"))

            sections = []
            current_section = {"heading": None, "content": []}

            # Fjern kendte støjfraser (case-insensitive match)
            noise_phrases = [
                "this is a modal window", ### Til at fjerne "skjult" tekst, som ikke er synlig for brugeren, i dette tilfælde et video vindue
                "beginning of dialog window",
                "escape will cancel",
                "subtitles off",
                "opens subtitles settings dialog",
                "end of dialog window",
                "fandt du ikke, hvad du ledte efter",
                "genveje"# Fjerne bokse som indeholder genveje
            ]
            for el in elements:
                # Tjek synlighed (undgår skjult UI-tekst)
                is_visible = el.evaluate("el => el.offsetHeight > 0 && el.offsetWidth > 0")
                if not is_visible:
                    continue  # spring over usynligt indhold

                tag = el.evaluate("el => el.tagName").lower()
                text = el.inner_text().strip().replace("\u00a0", " ")  # Erstat non-breaking space med almindelig space

                
                if any(noise in text.lower() for noise in noise_phrases):
                    continue  # spring over støj

                if tag in ["h1", "h2", "h3"]: ### Overskrifter
                    # Gem tidligere sektion hvis content ikke er tom
                    content_str = " ".join(current_section["content"]).strip()
                    if (current_section["heading"] or current_section["content"]) and content_str != "":
                        sections.append({
                            "heading": current_section["heading"],
                            "content": content_str
                        })
                    current_section = {
                        "heading": text,
                        "content": []
                    }
                else: ### Brødtekst som hører til overskriften
                    current_section["content"].append(text)
            # Tilføj sidste sektion hvis content ikke er tom
            content_str = " ".join(current_section["content"]).strip()
            if (current_section["heading"] or current_section["content"]) and content_str != "":
                sections.append({
                    "heading": current_section["heading"],
                    "content": content_str
                })

            data.append({
                "link": "https://www.su.dk" + link,
                "title": page.query_selector("#ContentPlaceHolderDefault_toolSection_breadcrumb_ctrl > div > ul > li.active > span").inner_text().strip(),
                "sections": sections
            })

        json_path = f"støtte_til_forældre{time.strftime('%Y-%m-%d')}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        browser.close()
        
run()