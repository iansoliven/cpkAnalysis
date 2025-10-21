from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence


logger = logging.getLogger(__name__)


def _iter_sessions(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    yield from (path for path in root.iterdir() if path.is_dir() and path.name.startswith("session_"))


def prune_sessions(
    root: Path,
    *,
    older_than_hours: float | None = None,
    dry_run: bool = False,
) -> List[Path]:
    """Remove session directories beneath *root* that match the criteria."""
    now = time.time()
    threshold = None
    if older_than_hours is not None:
        threshold = now - (max(older_than_hours, 0.0) * 3600.0)

    to_remove: List[Path] = []
    for session_dir in sorted(_iter_sessions(root)):
        try:
            stat = session_dir.stat()
        except FileNotFoundError:
            continue
        if threshold is not None and stat.st_mtime > threshold:
            continue
        to_remove.append(session_dir)

    if dry_run:
        return to_remove

    removed: List[Path] = []
    for path in to_remove:
        try:
            shutil.rmtree(path)
            removed.append(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Failed to remove session directory '%s': %s", path, exc)
    return removed


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune CPK Analysis session directories.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("temp"),
        help="Root directory containing session_* folders (default: ./temp).",
    )
    parser.add_argument(
        "--older-than",
        type=float,
        help="Remove sessions older than the specified number of hours (default: remove all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sessions that would be deleted without removing them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"No session directory found at {root}")
        return 0

    removed = prune_sessions(root, older_than_hours=args.older_than, dry_run=args.dry_run)
    if args.dry_run:
        if removed:
            print("Sessions eligible for removal:")
            for path in removed:
                print(f"  {path}")
        else:
            print("No session directories meet the removal criteria.")
    else:
        if removed:
            print("Removed session directories:")
            for path in removed:
                print(f"  {path}")
        else:
            print("No session directories were removed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
