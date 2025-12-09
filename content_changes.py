import os
import json
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

def load_json_content(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def get_two_latest_content_files(data_dir=None):
    base_dir = Path(data_dir) if data_dir else _default_data_dir()
    json_files = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if name.startswith("links_content_") and name.endswith(".json") and "all_relative" not in name:
                full_path = os.path.join(root, name)
                try:
                    timestamp = datetime.strptime(name.split("_")[2].split(".")[0], "%Y-%m-%d")
                    json_files.append((timestamp, full_path))
                except Exception:
                    continue
    json_files.sort()
    if len(json_files) < 2:
        raise Exception("Der skal være mindst to links_content JSON-filer at sammenligne.")
    return json_files[-2][1], json_files[-1][1]

def index_by_link(entries):
    return {entry["link"]: entry for entry in entries}

def compare_sections(sections1, sections2):
    text1 = "\n".join(f"{s['heading']}:\n{s['content']}" for s in sections1)
    text2 = "\n".join(f"{s['heading']}:\n{s['content']}" for s in sections2)
    diff = list(unified_diff(text1.splitlines(), text2.splitlines(), lineterm=""))
    return diff

def compare_content(file1, file2):
    print(f"🔍 Sammenligner filer:\n  {file1}\n  {file2}\n")

    content1 = index_by_link(load_json_content(file1))
    content2 = index_by_link(load_json_content(file2))

    links1 = set(content1.keys())
    links2 = set(content2.keys())

    added = links2 - links1
    removed = links1 - links2
    common = links1 & links2

    if added:
        print("🔼 Nye links:")
        for link in sorted(added):
            print(" +", link)

    if removed:
        print("\n🔽 Fjernede links:")
        for link in sorted(removed):
            print(" -", link)

    print("\n📝 Ændringer i fælles links:")
    for link in sorted(common):
        entry1 = content1[link]
        entry2 = content2[link]

        title_changed = entry1.get("title", "").strip() != entry2.get("title", "").strip()
        section_diff = compare_sections(entry1.get("sections", []), entry2.get("sections", []))

        if title_changed or section_diff:
            print(f"\n🔁 {link}")
            if title_changed:
                print(f"  🏷️ Titel ændret:\n    Før: {entry1['title']}\n    Nu:  {entry2['title']}")
            if section_diff:
                print("  📄 Indhold ændret (diff):")
                for line in section_diff[:30]:  # vis kun første 30 linjer
                    print("   ", line)
                if len(section_diff) > 30:
                    print("   ... (forkortet diff)")

# Kør sammenligning
if __name__ == "__main__":
    file1, file2 = get_two_latest_content_files()
    compare_content(file1, file2)
