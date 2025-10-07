from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from .plotly_charts import render_cdf, render_histogram, render_time_series
from .stats import SUMMARY_COLUMNS

MEAS_COLUMNS = [
    ("file", "File"),
    ("device_id", "DeviceID"),
    ("test_name", "Test Name"),
    ("test_number", "Test Number"),
    ("value", "Value"),
    ("units", "Units"),
    ("timestamp", "Timestamp/Index"),
]

TEST_LIMIT_COLUMNS = [
    ("test_name", "Test name"),
    ("test_number", "Test number"),
    ("stdf_upper", "STDF Upper Limit"),
    ("stdf_lower", "STDF Lower Limit"),
    ("unit", "Unit"),
    ("spec_upper", "Spec Upper Limit"),
    ("spec_lower", "Spec Lower Limit"),
    ("what_if_upper", "User What-If Upper Limit"),
    ("what_if_lower", "User What-If Lower Limit"),
]

CPK_COLUMNS = [
    "File",
    "Test Name",
    "Test Number",
    "CPL",
    "CPU",
    "CPK",
    "%YLD LOSS",
    "LL_2CPK",
    "UL_2CPK",
    "CPK_2.0",
    "%YLD LOSS_2.0",
    "LL_3IQR",
    "UL_3IQR",
    "CPK_3IQR",
    "%YLD LOSS_3IQR",
    "PLOTS",
    "Proposal",
    "Lot Qual",
]

ROW_STRIDE = 28
IMAGE_ANCHOR_COLUMN = 2
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540


def build_workbook(
    *,
    summary: pd.DataFrame,
    measurements: pd.DataFrame,
    test_limits: pd.DataFrame,
    limit_sources: dict[tuple[str, str, str], dict[str, str]],
    outlier_summary: dict[str, Any],
    per_file_stats: list[dict[str, Any]],
    output_path: Path,
    template_path: Optional[Path],
    include_histogram: bool,
    include_cdf: bool,
    include_time_series: bool,
    temp_dir: Path,
) -> None:
    workbook = _load_base_workbook(template_path)
    _write_summary_sheet(workbook, summary)
    _write_measurements(workbook, measurements)
    _write_test_limits(workbook, test_limits)
    plot_links: dict[tuple[str, str, str], str] = {}
    if include_histogram or include_cdf or include_time_series:
        plot_links = _create_plot_sheets(
            workbook,
            measurements,
            test_limits,
            include_histogram=include_histogram,
            include_cdf=include_cdf,
            include_time_series=include_time_series,
        )
    _populate_cpk_report(workbook, summary, plot_links)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    workbook.close()


def _load_base_workbook(template_path: Optional[Path]) -> Workbook:
    if template_path and template_path.exists():
        return load_workbook(template_path)
    workbook = Workbook()
    default = workbook.active
    workbook.remove(default)
    return workbook


def _write_summary_sheet(workbook: Workbook, summary: pd.DataFrame) -> None:
    if "Summary" in workbook.sheetnames:
        del workbook["Summary"]
    ws = workbook.create_sheet("Summary", 0)
    ws.append(SUMMARY_COLUMNS)

    for _, row in summary.iterrows():
        ws.append([row.get(column) for column in SUMMARY_COLUMNS])

    last_row = ws.max_row
    table = Table(displayName="SummaryTable", ref=f"A1:{get_column_letter(len(SUMMARY_COLUMNS))}{last_row}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True, showColumnStripes=False)
    ws.add_table(table)
    ws.freeze_panes = "A2"

    widths = [18, 42, 18, 12, 10, 12, 12, 12, 12, 10, 10, 10, 14, 14, 14, 12, 16, 14, 14, 12, 18]
    for idx, width in enumerate(widths[: len(SUMMARY_COLUMNS)], start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width


def _write_measurements(workbook: Workbook, measurements: pd.DataFrame) -> None:
    for sheet in list(workbook.sheetnames):
        if sheet.startswith("Measurements"):
            del workbook[sheet]

    subset = measurements.copy()
    subset = subset.loc[:, [source for source, _ in MEAS_COLUMNS]]
    subset.columns = [target for _, target in MEAS_COLUMNS]

    max_rows = 1_048_576
    chunk_size = max_rows - 1
    total_rows = len(subset)
    for index, start in enumerate(range(0, total_rows, chunk_size)):
        end = min(start + chunk_size, total_rows)
        chunk = subset.iloc[start:end]
        suffix = "" if index == 0 else f"_{index + 1}"
        sheet_name = f"Measurements{suffix}"
        sheet_index = 1 + index
        ws = workbook.create_sheet(sheet_name, index=sheet_index)
        headers = list(chunk.columns)
        ws.append(headers)
        for row in chunk.itertuples(index=False):
            ws.append(list(row))
        last_row = ws.max_row
        table_ref = f"A1:{get_column_letter(len(headers))}{last_row}"
        table_name = f"MeasurementsTable{index + 1}"
        table = Table(displayName=table_name, ref=table_ref)
        table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True, showColumnStripes=False)
        ws.add_table(table)
        ws.freeze_panes = "A2"
        widths = [18, 20, 36, 18, 18, 16, 18]
        for idx, width in enumerate(widths[: len(headers)], start=1):
            ws.column_dimensions[get_column_letter(idx)].width = width


def _write_test_limits(workbook: Workbook, test_limits: pd.DataFrame) -> None:
    if "Test List and Limits" in workbook.sheetnames:
        del workbook["Test List and Limits"]
    ws = workbook.create_sheet("Test List and Limits", index=len(workbook.sheetnames))

    data_frame = test_limits.copy()
    if data_frame.empty:
        for _, header in TEST_LIMIT_COLUMNS:
            ws.append([header])
        return

    ordered_columns = [source for source, _ in TEST_LIMIT_COLUMNS if source in data_frame.columns]
    data_frame = data_frame.loc[:, ordered_columns]
    headers = [target for _, target in TEST_LIMIT_COLUMNS]
    ws.append(headers)
    for _, row in data_frame.iterrows():
        ws.append([row.get(source) for source, _ in TEST_LIMIT_COLUMNS])

    last_row = ws.max_row
    table = Table(displayName="LimitsTable", ref=f"A1:{get_column_letter(len(headers))}{last_row}")
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True, showColumnStripes=False)
    ws.add_table(table)
    ws.freeze_panes = "A2"
    widths = [32, 18, 18, 18, 12, 18, 18, 24, 24]
    for idx, width in enumerate(widths[: len(headers)], start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width


def _create_plot_sheets(
    workbook: Workbook,
    measurements: pd.DataFrame,
    test_limits: pd.DataFrame,
    *,
    include_histogram: bool,
    include_cdf: bool,
    include_time_series: bool,
) -> dict[tuple[str, str, str], str]:
    limit_map = _limit_lookup(test_limits)
    plot_links: dict[tuple[str, str, str], str] = {}
    grouped = measurements.groupby(["file", "test_name", "test_number"], sort=False)

    hist_sheets: dict[str, Any] = {}
    cdf_sheets: dict[str, Any] = {}
    time_sheets: dict[str, Any] = {}

    for (file_name, test_name, test_number), group in grouped:
        limit_info = limit_map.get((test_name, test_number), {})
        values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        timestamp = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy()
        x_range = (_finite_min(values), _finite_max(values))

        if include_histogram:
            anchor = _ensure_plot_anchor(hist_sheets, workbook, file_name, "Histogram")
            row, sheet = anchor["row"], anchor["sheet"]
            image_bytes = render_histogram(
                values,
                lower_limit=limit_info.get("active_lower"),
                upper_limit=limit_info.get("active_upper"),
                x_range=x_range,
            )
            cell = _place_image(sheet, image_bytes, row, label=f"{test_name} (Test {test_number})")
            plot_links[(file_name, test_name, test_number)] = f"'{sheet.title}'!{cell}"
            hist_sheets[file_name]["row"] += ROW_STRIDE

        if include_cdf:
            anchor_cdf = _ensure_plot_anchor(cdf_sheets, workbook, file_name, "CDF")
            image_bytes = render_cdf(
                values,
                lower_limit=limit_info.get("active_lower"),
                upper_limit=limit_info.get("active_upper"),
                x_range=x_range,
            )
            _place_image(anchor_cdf["sheet"], image_bytes, anchor_cdf["row"], label=f"{test_name} (Test {test_number})")
            cdf_sheets[file_name]["row"] += ROW_STRIDE

        if include_time_series:
            anchor_ts = _ensure_plot_anchor(time_sheets, workbook, file_name, "TimeSeries")
            image_bytes = render_time_series(
                x=timestamp,
                y=values,
                lower_limit=limit_info.get("active_lower"),
                upper_limit=limit_info.get("active_upper"),
            )
            _place_image(anchor_ts["sheet"], image_bytes, anchor_ts["row"], label=f"{test_name} (Test {test_number})")
            time_sheets[file_name]["row"] += ROW_STRIDE

    return plot_links


def _populate_cpk_report(
    workbook: Workbook,
    summary: pd.DataFrame,
    plot_links: dict[tuple[str, str, str], str],
) -> None:
    sheet_name = "CPK Report"
    if sheet_name not in workbook.sheetnames:
        ws = workbook.create_sheet(sheet_name)
        ws.append(CPK_COLUMNS)
    ws = workbook[sheet_name]

    header = [str(cell.value).strip() if cell.value else "" for cell in ws[1]]
    column_map: dict[str, int] = {}
    for idx, column in enumerate(header, start=1):
        if column:
            column_map[column] = idx

    for column in CPK_COLUMNS:
        if column not in column_map:
            header.append(column)
            column_map[column] = len(header)
            ws.cell(row=1, column=column_map[column], value=column)

    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)

    for _, row in summary.iterrows():
        excel_row_index = ws.max_row + 1
        ws.append([None] * len(header))
        for column in CPK_COLUMNS:
            cell_index = column_map[column]
            cell = ws.cell(row=excel_row_index, column=cell_index)
            if column == "PLOTS":
                link_key = (row["File"], row["Test Name"], row["Test Number"])
                hyperlink = plot_links.get(link_key)
                if hyperlink:
                    cell.value = "Histogram"
                    cell.hyperlink = hyperlink
                    cell.style = "Hyperlink"
                else:
                    cell.value = ""
            elif column == "Proposal" or column == "Lot Qual":
                cell.value = ""
            else:
                cell.value = row.get(column, "")

    for idx in range(1, len(header) + 1):
        ws.column_dimensions[get_column_letter(idx)].width = 18


def _ensure_plot_anchor(cache: dict[str, Any], workbook: Workbook, file_name: str, prefix: str) -> dict[str, Any]:
    entry = cache.get(file_name)
    if entry is not None:
        return entry
    sheet_name = _safe_sheet_name(f"{prefix}_{file_name}")
    sheet = workbook.create_sheet(sheet_name)
    sheet.sheet_view.showGridLines = False
    cache[file_name] = {"sheet": sheet, "row": 2}
    sheet.cell(row=1, column=1, value=f"{prefix} plots for {file_name}")
    return cache[file_name]


def _place_image(sheet, image_bytes: bytes, anchor_row: int, label: str) -> str:
    anchor_cell = f"A{anchor_row}"
    sheet.cell(row=anchor_row, column=1, value=label)
    image_stream = BytesIO(image_bytes)
    img = XLImage(image_stream)
    img.width = IMAGE_WIDTH
    img.height = IMAGE_HEIGHT
    target_cell = f"{get_column_letter(IMAGE_ANCHOR_COLUMN)}{anchor_row}"
    sheet.add_image(img, target_cell)
    return target_cell


def _safe_sheet_name(name: str) -> str:
    invalid = set("\\/:*?[]")
    sanitized = "".join("_" if ch in invalid else ch for ch in name)
    if len(sanitized) > 31:
        sanitized = sanitized[:28] + "..."
    return sanitized or "Sheet"


def _limit_lookup(test_limits: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    mapping: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in test_limits.iterrows():
        key = (str(row.get("test_name", "")), str(row.get("test_number", "")))
        lsl = _coerce_float(row.get("what_if_lower"))
        if lsl is None:
            lsl = _coerce_float(row.get("spec_lower"))
        if lsl is None:
            lsl = _coerce_float(row.get("stdf_lower"))
        usl = _coerce_float(row.get("what_if_upper"))
        if usl is None:
            usl = _coerce_float(row.get("spec_upper"))
        if usl is None:
            usl = _coerce_float(row.get("stdf_upper"))
        mapping[key] = {
            "active_lower": lsl,
            "active_upper": usl,
        }
    return mapping


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _finite_min(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.min())


def _finite_max(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.max())
