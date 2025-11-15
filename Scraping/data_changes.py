import os
import argparse

def read_links(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def compare_links(file1, file2):
    print(f"Comparing:\n  {file1}\n  {file2}")

    day1 = read_links(file1)
    day2 = read_links(file2)

    new_links = day2 - day1
    removed_links = day1 - day2

    if not new_links:
        print("✅ No new links found.")
    else:
        print("\n🔼 New links:")
        for link in sorted(new_links):
            print(link)

    if not removed_links:
        print("✅ No removed links found.")
    else:
        print("\n🔽 Removed links:")
        for link in sorted(removed_links):
            print(link)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two link files.")
    parser.add_argument("file1", help="Path to first file (older scrape)")
    parser.add_argument("file2", help="Path to second file (newer scrape)")
    args = parser.parse_args()

    if not os.path.isfile(args.file1):
        raise FileNotFoundError(f"File not found: {args.file1}")
    if not os.path.isfile(args.file2):
        raise FileNotFoundError(f"File not found: {args.file2}")

    compare_links(args.file1, args.file2)
