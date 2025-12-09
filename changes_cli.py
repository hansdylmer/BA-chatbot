from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Optional, Tuple

from content_changes import compare_content, get_two_latest_content_files
from data_changes import compare_links


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except OSError:
            pass


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


def _ensure_data_dir(path: Optional[Path]) -> Path:
    data_dir = (path or DEFAULT_DATA_DIR).expanduser().resolve()
    if not data_dir.exists():
        print(f"[ERR] Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    return data_dir


def _latest_link_files(data_dir: Path) -> Tuple[str, str]:
    candidates: list[tuple[dt.datetime, Path]] = []
    for txt_file in data_dir.glob("*/all_relative_links_*.txt"):
        try:
            timestamp = dt.datetime.strptime(txt_file.parent.name, "%Y-%m-%d")
        except ValueError:
            continue
        candidates.append((timestamp, txt_file))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < 2:
        raise ValueError("Need at least two link files to compare.")
    older, newer = candidates[-2][1], candidates[-1][1]
    return str(older), str(newer)


def _run_content_changes(file1: Optional[Path], file2: Optional[Path], data_dir: Path):
    if file1 and file2:
        first, second = str(file1), str(file2)
    else:
        try:
            first, second = get_two_latest_content_files(str(data_dir))
        except Exception as exc:  # pragma: no cover - helper already validates
            print(f"[ERR] Failed to locate content files: {exc}", file=sys.stderr)
            sys.exit(1)
    print(f"[INFO] Comparing content files:\n  {first}\n  {second}\n")
    compare_content(first, second)


def _run_link_changes(file1: Optional[Path], file2: Optional[Path], data_dir: Path):
    if file1 and file2:
        first, second = str(file1), str(file2)
    else:
        try:
            first, second = _latest_link_files(data_dir)
        except ValueError as exc:
            print(f"[ERR] {exc}", file=sys.stderr)
            sys.exit(1)
    print(f"[INFO] Comparing link files:\n  {first}\n  {second}\n")
    compare_links(first, second)


def _add_shared_args(subparser: argparse.ArgumentParser):
    subparser.add_argument(
        "--file1",
        type=Path,
        help="Explicit path to the older file (JSON for content, TXT for links).",
    )
    subparser.add_argument(
        "--file2",
        type=Path,
        help="Explicit path to the newer file (JSON for content, TXT for links).",
    )
    subparser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing dated scrape folders (defaults to ./data).",
    )


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convenience CLI for running data/content diff scripts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_content = subparsers.add_parser("content", help="Run links_content diff.")
    _add_shared_args(parser_content)

    parser_links = subparsers.add_parser("links", help="Run all_relative_links diff.")
    _add_shared_args(parser_links)

    parser_all = subparsers.add_parser("all", help="Run both diff scripts back-to-back.")
    parser_all.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing dated scrape folders (defaults to ./data).",
    )

    args = parser.parse_args(argv)
    if args.command == "content":
        base = _ensure_data_dir(args.data_dir)
        _run_content_changes(args.file1, args.file2, base)
    elif args.command == "links":
        base = _ensure_data_dir(args.data_dir)
        _run_link_changes(args.file1, args.file2, base)
    elif args.command == "all":
        base = _ensure_data_dir(args.data_dir)
        _run_content_changes(None, None, base)
        print("\n---\n")
        _run_link_changes(None, None, base)
    else:  # pragma: no cover - defensive
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
