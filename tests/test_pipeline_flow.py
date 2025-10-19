from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis import ingest, outliers, pipeline, stats, workbook_builder
from cpkanalysis.models import AnalysisInputs, IngestResult, SourceFile


def _make_source(tmp_path: Path, name: str) -> SourceFile:
    path = tmp_path / name
    path.write_bytes(b"stdf")
    return SourceFile(path)


def test_pipeline_runs_stages_with_expected_data_flow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"

    def fake_session_dir() -> Path:
        session_dir.mkdir(exist_ok=True)
        return session_dir

    monkeypatch.setattr(pipeline, "_create_session_dir", fake_session_dir)
    monkeypatch.setattr(pipeline, "_cleanup_session_dir", lambda path: None)
    monkeypatch.setattr(pipeline, "_spinner", lambda message: contextlib.nullcontext())
    monkeypatch.setattr(pipeline, "apply_template", lambda workbook, sheet_name: sheet_name)

    raw_measurements = pd.DataFrame(
        [
            {
                "file": "first.stdf",
                "file_path": "first.stdf",
                "device_id": "D1",
                "device_sequence": 1,
                "test_name": "VDD",
                "test_number": "1",
                "units": "V",
                "value": 1.0,
                "stdf_lower": 0.5,
                "stdf_upper": 1.5,
                "stdf_result_format": None,
                "stdf_lower_format": None,
                "stdf_upper_format": None,
                "stdf_result_scale": None,
                "stdf_lower_scale": None,
                "stdf_upper_scale": None,
                "timestamp": 0.0,
                "measurement_index": 1,
                "part_status": "PASS",
            },
            {
                "file": "first.stdf",
                "file_path": "first.stdf",
                "device_id": "D2",
                "device_sequence": 2,
                "test_name": "VDD",
                "test_number": "1",
                "units": "V",
                "value": 2.0,
                "stdf_lower": 0.5,
                "stdf_upper": 1.5,
                "stdf_result_format": None,
                "stdf_lower_format": None,
                "stdf_upper_format": None,
                "stdf_result_scale": None,
                "stdf_lower_scale": None,
                "stdf_upper_scale": None,
                "timestamp": 1.0,
                "measurement_index": 1,
                "part_status": "FAIL",
            },
        ]
    )
    filtered_measurements = raw_measurements.assign(value=lambda df: df["value"] + 100)
    test_catalog = pd.DataFrame(
        [
            {
                "test_name": "VDD",
                "test_number": "1",
                "unit": "V",
                "stdf_lower": 0.5,
                "stdf_upper": 1.5,
                "spec_lower": None,
                "spec_upper": None,
                "what_if_lower": None,
                "what_if_upper": None,
            }
        ]
    )
    raw_store_path = tmp_path / "raw_measurements.parquet"
    raw_store_path.write_bytes(b"parquet")  # placeholder to satisfy metadata expectations

    ingest_result = IngestResult(
        frame=raw_measurements,
        test_catalog=test_catalog,
        per_file_stats=[{"file": "first.stdf", "measurement_count": 2}],
        raw_store_path=raw_store_path,
    )

    captured: dict[str, Any] = {}

    def fake_ingest_sources(sources, temp_dir):
        captured["ingest_sources"] = list(sources)
        assert temp_dir == session_dir
        return ingest_result

    def fake_apply_outlier_filter(frame, method, k, **kwargs):
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), raw_measurements)
        captured["outlier_method"] = (method, k)
        return filtered_measurements, {"method": method, "k": k, "removed": 1}

    def fake_compute_summary(measurements, limits):
        pd.testing.assert_frame_equal(measurements.reset_index(drop=True), filtered_measurements)
        pd.testing.assert_frame_equal(limits.reset_index(drop=True), test_catalog)
        summary_row = {
            "File": "first.stdf",
            "Test Name": "VDD",
            "Test Number": "1",
            "Unit": "V",
            "COUNT": 2,
            "MEAN": 1.5,
            "STDEV": 0.5,
            "LL_2CPK": 0.3,
            "UL_2CPK": 2.7,
            "LL_3IQR": 0.2,
            "UL_3IQR": 2.8,
        }
        summary_df = pd.DataFrame([summary_row])
        limit_sources = {("first.stdf", "VDD", "1"): {"lower": "spec", "upper": "spec"}}
        captured["summary_calls"] = captured.get("summary_calls", 0) + 1
        return summary_df, limit_sources

    def fake_compute_yield_pareto(measurements, limits):
        pd.testing.assert_frame_equal(measurements.reset_index(drop=True), raw_measurements)
        pd.testing.assert_frame_equal(limits.reset_index(drop=True), test_catalog)
        yield_df = pd.DataFrame(
            [
                {"file": "first.stdf", "devices_total": 2, "devices_pass": 1, "devices_fail": 1, "yield_percent": 0.5}
            ]
        )
        pareto_df = pd.DataFrame(
            [
                {
                    "file": "first.stdf",
                    "test_name": "VDD",
                    "test_number": "1",
                    "devices_fail": 1,
                    "fail_rate_percent": 0.5,
                    "cumulative_percent": 0.5,
                    "lower_limit": 0.5,
                    "upper_limit": 1.5,
                }
            ]
        )
        captured["yield_invocations"] = captured.get("yield_invocations", 0) + 1
        return yield_df, pareto_df

    def fake_build_workbook(**kwargs):
        measurements = kwargs["measurements"]
        pd.testing.assert_frame_equal(measurements.reset_index(drop=True), filtered_measurements)
        kwargs["timing_collector"]["render"] = 0.01
        captured["workbook_kwargs"] = kwargs
        return Workbook()

    monkeypatch.setattr(ingest, "ingest_sources", fake_ingest_sources)
    monkeypatch.setattr(outliers, "apply_outlier_filter", fake_apply_outlier_filter)
    monkeypatch.setattr(stats, "compute_summary", fake_compute_summary)
    monkeypatch.setattr(stats, "compute_yield_pareto", fake_compute_yield_pareto)
    monkeypatch.setattr(workbook_builder, "build_workbook", fake_build_workbook)

    source = _make_source(tmp_path, "first.stdf")
    output_path = tmp_path / "analysis.xlsx"
    config = AnalysisInputs(
        sources=[source],
        output=output_path,
        template_sheet="Report",
        generate_histogram=False,
        generate_cdf=False,
        generate_time_series=False,
        generate_yield_pareto=True,
    )

    result = pipeline.Pipeline(config, registry=None).run()

    assert output_path.exists()
    metadata_path = output_path.with_suffix(".json")
    assert metadata_path.exists()

    assert result["measurement_rows"] == len(filtered_measurements)
    assert result["yield_rows"] == 1
    assert captured["yield_invocations"] == 1
    assert "workbook_kwargs" in captured
    assert captured["summary_calls"] == 1
    assert captured["outlier_method"] == (config.outliers.method, config.outliers.k)
    assert result["stage_timings"]["workbook"] >= 0.0


def test_pipeline_site_breakdown_generates_site_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session_dir = tmp_path / "session"

    def fake_session_dir() -> Path:
        session_dir.mkdir(exist_ok=True)
        return session_dir

    monkeypatch.setattr(pipeline, "_create_session_dir", fake_session_dir)
    monkeypatch.setattr(pipeline, "_cleanup_session_dir", lambda path: None)
    monkeypatch.setattr(pipeline, "_spinner", lambda message: contextlib.nullcontext())

    raw_measurements = pd.DataFrame(
        [
            {
                "file": "first.stdf",
                "file_path": "first.stdf",
                "device_id": "D1",
                "device_sequence": 1,
                "site": 1,
                "test_name": "VDD",
                "test_number": "1",
                "units": "V",
                "value": 1.0,
                "stdf_lower": 0.5,
                "stdf_upper": 1.5,
                "timestamp": 0.0,
                "measurement_index": 1,
                "part_status": "PASS",
            },
            {
                "file": "first.stdf",
                "file_path": "first.stdf",
                "device_id": "D2",
                "device_sequence": 2,
                "site": 2,
                "test_name": "VDD",
                "test_number": "1",
                "units": "V",
                "value": 2.0,
                "stdf_lower": 0.5,
                "stdf_upper": 1.5,
                "timestamp": 1.0,
                "measurement_index": 1,
                "part_status": "FAIL",
            },
        ]
    )
    filtered_measurements = raw_measurements.copy()
    test_catalog = pd.DataFrame(
        [
            {
                "test_name": "VDD",
                "test_number": "1",
                "unit": "V",
                "stdf_lower": 0.5,
                "stdf_upper": 1.5,
            }
        ]
    )
    raw_store_path = tmp_path / "raw_measurements.parquet"
    raw_store_path.write_text("parquet")
    ingest_result = IngestResult(
        frame=raw_measurements,
        test_catalog=test_catalog,
        per_file_stats=[{"file": "first.stdf", "measurement_count": 2}],
        raw_store_path=raw_store_path,
    )

    captured: dict[str, Any] = {}

    monkeypatch.setattr(pipeline.ingest, "ingest_sources", lambda sources, temp_dir: ingest_result)

    def fake_apply_outlier_filter(frame, method, k, **kwargs):
        captured["group_keys"] = kwargs.get("group_keys")
        return filtered_measurements, {"method": method, "k": k, "removed": 0}

    monkeypatch.setattr(pipeline.outliers, "apply_outlier_filter", fake_apply_outlier_filter)

    summary_df = pd.DataFrame(
        [
            {
                "File": "first.stdf",
                "Test Name": "VDD",
                "Test Number": "1",
                "Unit": "V",
                "COUNT": 2,
                "MEAN": 1.5,
                "MEDIAN": 1.5,
                "STDEV": 0.5,
                "IQR": 0.0,
                "CPL": 0.0,
                "CPU": 0.0,
                "CPK": 0.0,
                "%YLD LOSS": 0.5,
                "LL_2CPK": 0.0,
                "UL_2CPK": 0.0,
                "CPK_2.0": 0.0,
                "%YLD LOSS_2.0": 0.5,
                "LL_3IQR": 0.0,
                "UL_3IQR": 0.0,
                "CPK_3IQR": 0.0,
                "%YLD LOSS_3IQR": 0.5,
            }
        ]
    )
    site_summary_df = pd.DataFrame(
        [
            {
                "File": "first.stdf",
                "Site": 1,
                "Test Name": "VDD",
                "Test Number": "1",
                "Unit": "V",
                "COUNT": 1,
                "MEAN": 1.0,
                "MEDIAN": 1.0,
                "STDEV": 0.0,
                "IQR": 0.0,
                "CPL": 0.0,
                "CPU": 0.0,
                "CPK": 0.0,
                "%YLD LOSS": 0.0,
                "LL_2CPK": 0.0,
                "UL_2CPK": 0.0,
                "CPK_2.0": 0.0,
                "%YLD LOSS_2.0": 0.0,
                "LL_3IQR": 0.0,
                "UL_3IQR": 0.0,
                "CPK_3IQR": 0.0,
                "%YLD LOSS_3IQR": 0.0,
            },
            {
                "File": "first.stdf",
                "Site": 2,
                "Test Name": "VDD",
                "Test Number": "1",
                "Unit": "V",
                "COUNT": 1,
                "MEAN": 2.0,
                "MEDIAN": 2.0,
                "STDEV": 0.0,
                "IQR": 0.0,
                "CPL": 0.0,
                "CPU": 0.0,
                "CPK": 0.0,
                "%YLD LOSS": 1.0,
                "LL_2CPK": 0.0,
                "UL_2CPK": 0.0,
                "CPK_2.0": 0.0,
                "%YLD LOSS_2.0": 1.0,
                "LL_3IQR": 0.0,
                "UL_3IQR": 0.0,
                "CPK_3IQR": 0.0,
                "%YLD LOSS_3IQR": 1.0,
            },
        ]
    )
    site_limit_sources = {
        ("first.stdf", 1, "VDD", "1"): {"lower": "spec", "upper": "spec"},
        ("first.stdf", 2, "VDD", "1"): {"lower": "spec", "upper": "spec"},
    }
    yield_df = pd.DataFrame(
        [
            {"file": "first.stdf", "devices_total": 2, "devices_pass": 1, "devices_fail": 1, "yield_percent": 0.5}
        ]
    )
    site_yield_df = pd.DataFrame(
        [
            {"file": "first.stdf", "site": 1, "devices_total": 1, "devices_pass": 1, "devices_fail": 0, "yield_percent": 1.0},
            {"file": "first.stdf", "site": 2, "devices_total": 1, "devices_pass": 0, "devices_fail": 1, "yield_percent": 0.0},
        ]
    )
    pareto_df = pd.DataFrame(
        [
            {
                "file": "first.stdf",
                "test_name": "VDD",
                "test_number": "1",
                "devices_fail": 1,
                "fail_rate_percent": 0.5,
                "cumulative_percent": 0.5,
                "lower_limit": 0.5,
                "upper_limit": 1.5,
            }
        ]
    )
    site_pareto_df = pd.DataFrame(
        [
            {
                "file": "first.stdf",
                "site": 2,
                "test_name": "VDD",
                "test_number": "1",
                "devices_fail": 1,
                "fail_rate_percent": 1.0,
                "cumulative_percent": 1.0,
                "lower_limit": 0.5,
                "upper_limit": 1.5,
            }
        ]
    )

    monkeypatch.setattr(pipeline.stats, "compute_summary", lambda measurements, limits: (summary_df, {}))
    monkeypatch.setattr(
        pipeline.stats,
        "compute_summary_by_site",
        lambda measurements, limits: (site_summary_df, site_limit_sources),
    )
    monkeypatch.setattr(pipeline.stats, "compute_yield_pareto", lambda measurements, limits: (yield_df, pareto_df))
    monkeypatch.setattr(
        pipeline.stats,
        "compute_yield_pareto_by_site",
        lambda measurements, limits: (site_yield_df, site_pareto_df),
    )
    def fake_build_workbook(**kwargs):
        captured["build_kwargs"] = kwargs
        return Workbook()

    monkeypatch.setattr(workbook_builder, "build_workbook", fake_build_workbook)

    config = AnalysisInputs(
        sources=[_make_source(tmp_path, "first.stdf")],
        output=tmp_path / "analysis_site.xlsx",
        enable_site_breakdown=True,
        generate_yield_pareto=True,
    )

    result = pipeline.Pipeline(config, registry=None).run()

    assert captured["group_keys"] == ["file", "site", "test_name", "test_number"]
    assert result["site_breakdown_requested"] is True
    assert result["site_data_available"] is True
    assert result["site_summary_rows"] == len(site_summary_df)
    assert result["site_yield_rows"] == len(site_yield_df)
    assert result["site_pareto_rows"] == len(site_pareto_df)
    assert result["site_breakdown_generated"] is True
    metadata = json.loads((config.output.with_suffix(".json")).read_text())
    assert metadata["site_breakdown"]["generated"] is True
    assert metadata["analysis_options"]["enable_site_breakdown"] is True
    build_kwargs = captured["build_kwargs"]
    assert build_kwargs["site_enabled"] is True
    assert build_kwargs["site_summary"] is site_summary_df
    assert build_kwargs["site_yield_summary"] is site_yield_df
