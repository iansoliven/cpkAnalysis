from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from cpkanalysis.workbook_builder import build_workbook
from cpkanalysis.stats import SUMMARY_COLUMNS


def _make_summary_rows() -> pd.DataFrame:
    base = {column: 0 for column in SUMMARY_COLUMNS}
    rows = []
    row1 = base.copy()
    row1.update({"File": "lot1", "Test Name": "T1", "Test Number": "1", "Unit": "V", "COUNT": 3})
    row2 = base.copy()
    row2.update({"File": "lot2", "Test Name": "T1", "Test Number": "1", "Unit": "V", "COUNT": 1})
    rows.extend([row1, row2])
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def test_build_workbook_creates_yield_pareto_sheet(tmp_path: Path) -> None:
    summary_df = _make_summary_rows()

    measurements_df = pd.DataFrame(
        [
            {"file": "lot1", "device_id": "D1", "test_name": "T1", "test_number": "1", "value": 0.5, "units": "V", "timestamp": 0.0},
        ]
    )

    test_limits_df = pd.DataFrame(
        [
            {
                "test_name": "T1",
                "test_number": "1",
                "unit": "V",
                "stdf_lower": 0.0,
                "stdf_upper": 1.0,
                "spec_lower": None,
                "spec_upper": None,
                "what_if_lower": None,
                "what_if_upper": None,
            }
        ]
    )

    yield_summary = pd.DataFrame(
        [
            {"file": "lot1", "devices_total": 3, "devices_pass": 2, "devices_fail": 1, "yield_percent": 66.666},
            {"file": "lot2", "devices_total": 1, "devices_pass": 1, "devices_fail": 0, "yield_percent": 100.0},
        ]
    )

    pareto_summary = pd.DataFrame(
        [
            {
                "file": "lot1",
                "test_name": "T1",
                "test_number": "1",
                "devices_fail": 1,
                "fail_rate_percent": 33.333,
                "cumulative_percent": 33.333,
                "lower_limit": 0.0,
                "upper_limit": 1.0,
            }
        ]
    )

    output_path = tmp_path / "cpk.xlsx"

    build_workbook(
        summary=summary_df,
        measurements=measurements_df,
        test_limits=test_limits_df,
        limit_sources={},
        outlier_summary={"removed": 0},
        per_file_stats=[],
        output_path=output_path,
        template_path=None,
        include_histogram=False,
        include_cdf=False,
        include_time_series=False,
        include_yield_pareto=True,
        yield_summary=yield_summary,
        pareto_summary=pareto_summary,
        fallback_decimals=None,
        temp_dir=tmp_path,
    )

    workbook = load_workbook(output_path)
    assert "Yield and Pareto" in workbook.sheetnames
    sheet = workbook["Yield and Pareto"]

    assert sheet["A1"].value == "File"
    assert sheet["A2"].value == "lot1"

    assert sheet.cell(row=2, column=5).value == pytest.approx(0.666, rel=1e-3)

    found_file_header = any(cell.value == "File: lot1" for cell in sheet[
        "A"
    ])
    assert found_file_header

    table_names = {table.displayName for table in sheet._tables.values()}
    assert "YieldSummaryTable" in table_names
    assert any(name.startswith("YieldTable") for name in table_names)
    assert any(name.startswith("ParetoTable") for name in table_names)
