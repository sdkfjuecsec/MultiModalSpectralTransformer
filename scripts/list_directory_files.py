#!/usr/bin/env python3
"""List every file below one directory and write the inventory to a text file.

Motivation: the dataset groups are tens of gigabytes, so the fastest way to
inspect a downloaded group (for example the extracted models/ folder in Colab)
is to produce a small text report and download that instead of the binaries.

Usage: exactly one directory per run, plus an optional report path.
    python scripts/list_directory_files.py models
    python scripts/list_directory_files.py /data/MultiModalSpectralTransformer/models -o models_files.txt
    python scripts/list_directory_files.py "D:\\Challenge\\MultiModalSpectralTransformer\\models"

The report contains:
    * one aligned row per file: relative path, exact bytes, human size, mtime
    * a per-extension summary (file count and total bytes)
    * an overall total (files, directories, bytes, human size)
"""

import argparse
import datetime
from pathlib import Path

DEFAULT_OUTPUT = "file_list.txt"


def human_size(size_bytes: int) -> str:
    """Convert bytes to a readable string using binary units (KiB, MiB, GiB)."""
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:,.2f} {unit}" if unit != "B" else f"{int(value):,} B"
        value /= 1024.0
    return f"{value:,.2f} TiB"


def collect_entries(directory: Path) -> tuple[list[dict], int]:
    """Return one record per file plus the number of directories visited."""
    entries = []
    directory_count = 0
    # rglob("*") walks the tree depth-first; sorted makes the report stable.
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            directory_count += 1
            continue
        stat = path.stat()
        entries.append(
            {
                "relative_path": str(path.relative_to(directory)),
                "size_bytes": stat.st_size,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "suffix": path.suffix.lower() or "(no extension)",
            }
        )
    return entries, directory_count


def render_report(directory: Path, entries: list[dict], directory_count: int) -> str:
    """Build the whole report as one string, so it can go to file and stdout."""
    lines = []
    total_bytes = sum(entry["size_bytes"] for entry in entries)

    lines.append("Directory inventory")
    lines.append(f"Root      : {directory}")
    lines.append(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Files     : {len(entries):,}")
    lines.append(f"Folders   : {directory_count:,}")
    lines.append(f"Total size: {total_bytes:,} bytes ({human_size(total_bytes)})")
    lines.append("")

    # Column widths follow the content, so long model file names are never cut.
    path_width = max([len("relative path")] + [len(e["relative_path"]) for e in entries])
    header = (
        f"{'relative path':<{path_width}}  {'bytes':>15}  {'size':>12}  {'modified':<19}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for entry in entries:
        lines.append(
            f"{entry['relative_path']:<{path_width}}  "
            f"{entry['size_bytes']:>15,}  "
            f"{human_size(entry['size_bytes']):>12}  "
            f"{entry['modified']:<19}"
        )
    lines.append("")

    # Per-extension summary answers "what is in this folder" at a glance.
    summary = {}
    for entry in entries:
        suffix = entry["suffix"]
        count, size = summary.get(suffix, (0, 0))
        summary[suffix] = (count + 1, size + entry["size_bytes"])

    if summary:
        # Width from the content, never narrower than the header word.
        suffix_width = max([len("extension")] + [len(suffix) for suffix in summary])
        lines.append("Summary by extension")
        header = (
            f"{'extension':<{suffix_width}}  {'files':>8}  {'bytes':>15}  {'size':>12}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        # Largest total first: the interesting rows end up on top.
        for suffix, (count, size) in sorted(
            summary.items(), key=lambda item: item[1][1], reverse=True
        ):
            lines.append(
                f"{suffix:<{suffix_width}}  {count:>8,}  {size:>15,}  {human_size(size):>12}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    """Parse the arguments, build the report and write it to the output file.

    Returns:
        int: 0 on success, 1 when the given path is not a directory.
    """
    parser = argparse.ArgumentParser(
        description="List all files below a directory and write a text report."
    )
    parser.add_argument(
        "directory",
        help="the one folder to inspect; quote the path if it contains spaces",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"report file to write (default: {DEFAULT_OUTPUT} in the current directory)",
    )
    arguments = parser.parse_args()

    directory = Path(arguments.directory).expanduser()
    if not directory.is_dir():
        print(f"error: not a directory: {directory}")
        return 1

    entries, directory_count = collect_entries(directory)
    report = render_report(directory, entries, directory_count)

    output_path = Path(arguments.output).expanduser()
    output_path.write_text(report, encoding="utf-8")

    # The same text goes to stdout, so it can be read without opening the file.
    print(report, end="")
    print(f"report written to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
