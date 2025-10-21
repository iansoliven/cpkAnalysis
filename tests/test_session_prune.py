from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis.session_prune import prune_sessions  # noqa: E402


def _make_session(root: Path, name: str, *, age_hours: float = 0.0) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    if age_hours > 0:
        past = time.time() - (age_hours * 3600.0)
        os.utime(path, (past, past))
    return path


def test_prune_sessions_removes_only_older_directories(tmp_path: Path) -> None:
    root = tmp_path / "temp"
    root.mkdir()
    old_session = _make_session(root, "session_1", age_hours=10)
    new_session = _make_session(root, "session_2", age_hours=1)

    removed = prune_sessions(root, older_than_hours=2.0, dry_run=False)

    assert old_session not in root.iterdir()
    assert new_session in root.iterdir()
    assert removed == [old_session]


def test_prune_sessions_dry_run_lists_candidates(tmp_path: Path) -> None:
    root = tmp_path / "temp"
    root.mkdir()
    target = _make_session(root, "session_old", age_hours=5)

    candidates = prune_sessions(root, older_than_hours=1.0, dry_run=True)

    assert candidates == [target]
    # Dry run should not delete anything.
    assert target.exists()
