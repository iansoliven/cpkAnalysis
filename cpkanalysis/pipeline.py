from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pandas as pd

from . import ingest, outliers, stats, workbook_builder
from .models import AnalysisInputs, IngestResult


def run_analysis(config: AnalysisInputs) -> dict[str, Any]:
    """Execute the end-to-end analysis pipeline."""
    session_dir = _create_session_dir()
    try:
        ingest_result = ingest.ingest_sources(config.sources, session_dir)
        filtered_frame, outlier_summary = outliers.apply_outlier_filter(
            ingest_result.frame, config.outliers.method, config.outliers.k
        )
        filtered_path = session_dir / "filtered_measurements.parquet"
        filtered_frame.to_parquet(filtered_path, engine="pyarrow", index=False)

        summary_df, limit_sources = stats.compute_summary(filtered_frame, ingest_result.test_catalog)

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

        metadata = _build_metadata(
            config=config,
            ingest_result=ingest_result,
            outlier_summary=outlier_summary,
            limit_sources=limit_sources,
            summary=summary_df,
        )
        metadata_path = config.output.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "output": str(config.output),
            "metadata": str(metadata_path),
            "summary_rows": int(len(summary_df)),
            "measurement_rows": int(len(filtered_frame)),
            "outlier_removed": outlier_summary.get("removed", 0),
        }
    finally:
        _cleanup_session_dir(session_dir)


def _build_metadata(
    config: AnalysisInputs,
    ingest_result: IngestResult,
    outlier_summary: dict[str, Any],
    limit_sources: dict[tuple[str, str, str], dict[str, str]],
    summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "output": str(config.output),
        "template": str(config.template) if config.template else None,
        "sources": ingest_result.per_file_stats,
        "outlier_filter": outlier_summary,
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
