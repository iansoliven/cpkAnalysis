from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional

from ..models import SourceFile

PATTERN = re.compile(
    r".+_(?P<lot>[^_]+)_(?P<event>[^_]+)_(?P<interval>[^_.]+)\.(?P<ext>xlsx|stdf)$",
    re.IGNORECASE,
)


def run(
    directory: Optional[Path],
    metadata_out: Optional[Path] = None,
    assume_yes: bool = False,
) -> List[SourceFile]:
    """Collect source files and metadata from a directory."""
    dir_path = directory if directory is not None else Path.cwd()
    dir_path = dir_path.expanduser().resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    matched: list[SourceFile] = []
    for entry in sorted(dir_path.iterdir()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in {".xlsx", ".stdf"}:
            continue
        match = PATTERN.match(entry.name)
        if not match:
            continue
        groups = match.groupdict()
        matched.append(
            SourceFile(
                path=entry.resolve(),
                lot=groups["lot"],
                event=groups["event"],
                interval=groups["interval"],
                file_type="xlsx" if groups["ext"].lower() == "xlsx" else "stdf",
            )
        )

    if not matched:
        print("No files matching the required pattern were found.")
        return []

    print("Detected files:")
    print(f"{'Filename':60} {'LOT':10} {'EVENT':10} {'INTERVAL':10} {'TYPE':6}")
    for source in matched:
        print(
            f"{source.path.name:60} {source.lot:10} {source.event:10} {source.interval:10} {source.file_type.upper():6}"
        )

    if not assume_yes:
        confirm = input("Are these correct? (y/n): ").strip().lower()
        if confirm != "y":
            print("Aborted by user.")
            return []

    if metadata_out is not None:
        serialize(matched, metadata_out)

    return matched


def serialize(sources: Iterable[SourceFile], destination: Path) -> None:
    """Persist metadata for reuse by later stages."""
    payload = [
        {
            "path": str(source.path),
            "lot": source.lot,
            "event": source.event,
            "interval": source.interval,
            "file_type": source.file_type,
        }
        for source in sources
    ]
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

