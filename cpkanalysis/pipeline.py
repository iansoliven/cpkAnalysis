from __future__ import annotations

import json
import shutil
import time
import itertools
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pandas as pd

from . import ingest, outliers, stats, workbook_builder
from .models import AnalysisInputs, IngestResult
from .move_to_template import run as move_to_template


def run_analysis(config: AnalysisInputs) -> dict[str, Any]:
    """Execute the end-to-end analysis pipeline."""
    session_dir = _create_session_dir()
    template_sheet_used: str | None = None
    try:
        with _spinner("Ingesting STDF sources"):
            ingest_result = ingest.ingest_sources(config.sources, session_dir)

        with _spinner("Applying outlier filters"):
            filtered_frame, outlier_summary = outliers.apply_outlier_filter(
                ingest_result.frame, config.outliers.method, config.outliers.k
            )
            filtered_path = session_dir / "filtered_measurements.parquet"
            filtered_frame.to_parquet(filtered_path, engine="pyarrow", index=False)

        with _spinner("Computing summary statistics"):
            summary_df, limit_sources = stats.compute_summary(filtered_frame, ingest_result.test_catalog)

        with _spinner("Building Excel workbook"):
            workbook_builder.build_workbook(
                summary=summary_df,
                measurements=filtered_frame,
                test_limits=ingest_result.test_catalog,
                limit_sources=limit_sources,
                outlier_summary=outlier_summary,
                per_file_stats=ingest_result.per_file_stats,
                output_path=config.output,
                template_path=config.template,
                include_histogram=config.generate_histogram,
                include_cdf=config.generate_cdf,
                include_time_series=config.generate_time_series,
                temp_dir=session_dir,
            )

        if config.template or config.template_sheet:
            with _spinner("Updating template sheet"):
                template_sheet_used = move_to_template(config.output, config.template_sheet)

        with _spinner("Writing metadata sidecar"):
            metadata = _build_metadata(
                config=config,
                ingest_result=ingest_result,
                outlier_summary=outlier_summary,
                limit_sources=limit_sources,
                summary=summary_df,
                template_sheet=template_sheet_used,
            )
            metadata_path = config.output.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "output": str(config.output),
            "metadata": str(metadata_path),
            "summary_rows": int(len(summary_df)),
            "measurement_rows": int(len(filtered_frame)),
            "outlier_removed": outlier_summary.get("removed", 0),
            "template_sheet": template_sheet_used,
        }
    finally:
        _cleanup_session_dir(session_dir)


def _build_metadata(
    config: AnalysisInputs,
    ingest_result: IngestResult,
    outlier_summary: dict[str, Any],
    limit_sources: dict[tuple[str, str, str], dict[str, str]],
    summary: pd.DataFrame,
    template_sheet: str | None = None,
) -> dict[str, Any]:
    return {
        "output": str(config.output),
        "template": str(config.template) if config.template else None,
        "sources": ingest_result.per_file_stats,
        "outlier_filter": outlier_summary,
        "template_sheet": template_sheet,
        "limit_sources": {
            f"{file}|{test}|{number}": sources
            for (file, test, number), sources in limit_sources.items()
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary_counts": {
            "rows": int(len(summary)),
            "tests": int(summary["Test Name"].nunique()) if not summary.empty else 0,
        },
    }


def _create_session_dir() -> Path:
    root = Path("temp")
    root.mkdir(exist_ok=True)
    session = root / f"session_{int(time.time() * 1000)}"
    session.mkdir()
    return session


def _cleanup_session_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


class _Spinner:
    BRAILLE_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
    ASCII_FRAMES = ["-", "\\", "|", "/"]

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._line_length = 0
        self._unicode_supported = _supports_output("".join(self.BRAILLE_FRAMES) + "\u2714\u2716")
        self._frames = self.BRAILLE_FRAMES if self._unicode_supported else self.ASCII_FRAMES
        self._success_symbol = "\u2714" if self._unicode_supported else "[OK]"
        self._failure_symbol = "\u2716" if self._unicode_supported else "[FAIL]"

    def start(self) -> None:
        self._thread.start()

    def stop(self, final_message: str) -> None:
        self._stop.set()
        self._thread.join()
        text = final_message
        pad = max(self._line_length - len(text), 0)
        self._write(f"\r{text}{' ' * pad}\n")

    def success_text(self, message: str) -> str:
        return f"{self._success_symbol} Completed: {message}"

    def failure_text(self, message: str) -> str:
        return f"{self._failure_symbol} Failed: {message}"

    def _spin(self) -> None:
        for frame in itertools.cycle(self._frames):
            if self._stop.is_set():
                break
            text = f"{frame} {self.message}"
            self._line_length = len(text)
            self._write(f"\r{text}")
            time.sleep(0.1)

    @staticmethod
    def _write(text: str) -> None:
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:
            sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
        sys.stdout.flush()


@contextmanager
def _spinner(message: str):
    spinner = _Spinner(message)
    spinner.start()
    try:
        yield
    except Exception:
        spinner.stop(spinner.failure_text(message))
        raise
    else:
        spinner.stop(spinner.success_text(message))


def _supports_output(sample: str) -> bool:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        sample.encode(encoding)
        return True
    except Exception:
        return False
