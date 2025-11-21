from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_postprocess_actions


def _prepare_project(tmp_path: Path) -> Tuple[Path, Path]:
    workbook_path = tmp_path / "resume.xlsx"
    test_postprocess_actions._build_workbook(workbook_path)
    metadata_path = workbook_path.with_suffix(".json")
    metadata = {
        "output": str(workbook_path),
        "generated_at": "2024-01-01T00:00:00Z",
        "template_sheet": "Template",
        "analysis_options": {
            "generate_histogram": True,
            "generate_cdf": False,
            "generate_time_series": False,
            "generate_yield_pareto": False,
            "display_decimals": 4,
        },
        "post_processing": {
            "runs": [
                {
                    "timestamp": "2024-01-02T12:00:00Z",
                    "action": "calculate_proposed_limits",
                    "scope": "all",
                    "tests": ["lot1|TestA|1"],
                }
            ]
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return workbook_path, metadata_path


def _run_gui(args: list[str], *, input_text: str = "8\n\n") -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "cpkanalysis.gui", *args]
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        cwd=repo_root,
    )


def test_gui_resume_opens_menu(tmp_path: Path) -> None:
    workbook_path, _ = _prepare_project(tmp_path)
    result = _run_gui(["--resume", str(workbook_path)])
    assert result.returncode == 0, result.stderr
    assert "Resuming post-processing session" in result.stdout
    assert "Post-Processing Menu" in result.stdout
    assert "post>" in result.stdout or "Enter 'post' to reopen the post-processing menu" in result.stdout


def test_gui_resume_missing_workbook_fails(tmp_path: Path) -> None:
    missing = tmp_path / "absent.xlsx"
    result = _run_gui(["--resume", str(missing)], input_text="")
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "Workbook not found" in combined or "Unable to resume session" in combined


def test_gui_resume_missing_metadata_strict_fails(tmp_path: Path) -> None:
    workbook_path = tmp_path / "resume.xlsx"
    test_postprocess_actions._build_workbook(workbook_path)
    result = _run_gui(["--resume", str(workbook_path)], input_text="")
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "Metadata JSON not found" in combined or "Unable to resume session" in combined


def test_gui_resume_missing_metadata_lax_succeeds(tmp_path: Path) -> None:
    workbook_path = tmp_path / "resume.xlsx"
    test_postprocess_actions._build_workbook(workbook_path)
    result = _run_gui(["--resume", str(workbook_path), "--resume-lax"])
    assert result.returncode == 0, result.stderr
    assert "Resuming post-processing session" in result.stdout
    assert "Post-Processing Menu" in result.stdout


def test_gui_resume_corrupt_metadata_lax_succeeds(tmp_path: Path) -> None:
    workbook_path, metadata_path = _prepare_project(tmp_path)
    metadata_path.write_text("{ not-json", encoding="utf-8")
    result = _run_gui(["--resume", str(workbook_path), "--resume-lax"])
    assert result.returncode == 0, result.stderr
    assert "Resuming post-processing session" in result.stdout
    assert "Post-Processing Menu" in result.stdout
