#!/usr/bin/env python3
"""Generate histogram charts per Test Name from the Measurements tables.

The script scans every worksheet whose name starts with "Measurements" inside the
combined workbook produced by mergeXlsx.py. For each distinct Test Name it builds a
histogram of the Measurement column, scales the X axis to the Low/High limits, draws
vertical limit lines, and embeds the rendered chart image into a dedicated Charts
worksheet sized approximately 10" x 5" (width x height).
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Always render off-screen
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.utils import get_column_letter

MEASUREMENTS_PREFIX = "Measurements"
INCH_TO_PIXELS = 96  # Excel assumes ~96 DPI for embedded images
CHART_WIDTH_IN = 10
CHART_HEIGHT_IN = 5
CHART_WIDTH_PX = int(CHART_WIDTH_IN * INCH_TO_PIXELS)
CHART_HEIGHT_PX = int(CHART_HEIGHT_IN * INCH_TO_PIXELS)
CHARTS_SHEET_NAME = "Charts"
CHARTS_PER_ROW = 2
ROW_STRIDE = 27  # heuristic rows to leave between chart anchors
COL_STRIDE = 15  # heuristic columns to leave between chart anchors


@dataclass
class TestData:
    name: str
    number: Optional[str]
    unit: Optional[str]
    measurements: List[float]
    low_limits: List[float]
    high_limits: List[float]

    @property
    def title(self) -> str:
        if self.number and str(self.number).strip():
            return f"{self.name} (Test {self.number})"
        return self.name

    def representative_limits(self) -> Tuple[Optional[float], Optional[float]]:
        low = _first_finite(self.low_limits)
        high = _first_finite(self.high_limits)
        return low, high


def _first_finite(values: Iterable[float]) -> Optional[float]:
    for value in values:
        if math.isfinite(value):
            return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add histogram charts for each Test Name in the combined workbook.")
    parser.add_argument(
        "workbook",
        nargs="?",
        type=Path,
        default=Path("combined_workbook.xlsx"),
        help="Path to the workbook that already contains Measurements sheets (default: combined_workbook.xlsx).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to save the augmented workbook. Defaults to overwriting the input.",
    )
    parser.add_argument(
        "--charts-per-row",
        type=int,
        default=CHARTS_PER_ROW,
        help="Number of charts per row in the output layout (default: 2).",
    )
    return parser.parse_args()


def load_measurement_tests(workbook_path: Path) -> Dict[Tuple[str, Optional[str]], TestData]:
    wb = load_workbook(workbook_path, data_only=True, read_only=True)
    try:
        measurement_sheet_names = [name for name in wb.sheetnames if name.startswith(MEASUREMENTS_PREFIX)]
        if not measurement_sheet_names:
            raise ValueError("No Measurements sheets were found in the workbook.")

        tests: Dict[Tuple[str, Optional[str]], TestData] = OrderedDict()
        for sheet_name in measurement_sheet_names:
            ws = wb[sheet_name]
            rows = ws.iter_rows(values_only=True)
            try:
                header = next(rows)
            except StopIteration:
                continue
            header_map = {str(value).strip(): idx for idx, value in enumerate(header) if value is not None}
            required_columns = [
                "Test Name",
                "Measurement",
            ]
            for column in required_columns:
                if column not in header_map:
                    raise ValueError(f"Column '{column}' was not found in sheet '{sheet_name}'.")

            idx_test_name = header_map["Test Name"]
            idx_measurement = header_map["Measurement"]
            idx_test_number = header_map.get("Test Number")
            idx_unit = header_map.get("Test Unit")
            idx_low_limit = header_map.get("Low Limit")
            idx_high_limit = header_map.get("High Limit")

            for row in rows:
                test_name = row[idx_test_name] if idx_test_name < len(row) else None
                if not isinstance(test_name, str) or not test_name.strip():
                    continue
                measurement_raw = row[idx_measurement] if idx_measurement < len(row) else None
                measurement = _coerce_float(measurement_raw)
                if measurement is None:
                    continue

                key = (test_name.strip(), _safe_string(row, idx_test_number))
                entry = tests.get(key)
                if entry is None:
                    entry = TestData(
                        name=test_name.strip(),
                        number=_safe_string(row, idx_test_number),
                        unit=_safe_string(row, idx_unit),
                        measurements=[],
                        low_limits=[],
                        high_limits=[],
                    )
                    tests[key] = entry

                entry.measurements.append(measurement)
                low_value = _coerce_float(row[idx_low_limit]) if idx_low_limit is not None else None
                high_value = _coerce_float(row[idx_high_limit]) if idx_high_limit is not None else None
                if low_value is not None:
                    entry.low_limits.append(low_value)
                if high_value is not None:
                    entry.high_limits.append(high_value)
        if not tests:
            raise ValueError("No measurement rows were found across the Measurements sheets.")
        return tests
    finally:
        wb.close()


def _coerce_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value_float = float(value)
        if math.isfinite(value_float):
            return value_float
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _safe_string(row: Tuple[object, ...], index: Optional[int]) -> Optional[str]:
    if index is None or index >= len(row):
        return None
    value = row[index]
    if value is None:
        return None
    return str(value).strip()


def determine_axis_limits(values: List[float], declared_low: Optional[float], declared_high: Optional[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("determine_axis_limits called with empty values")
    min_value = min(values)
    max_value = max(values)
    low = declared_low if declared_low is not None else min_value
    high = declared_high if declared_high is not None else max_value
    low = min(low, min_value)
    high = max(high, max_value)
    if math.isclose(low, high):
        span = max(abs(low) * 0.1, 1.0)
        low -= span
        high += span
    return low, high


def choose_bin_count(values: List[float], low: float, high: float) -> int:
    if not values:
        return 10
    span = high - low
    if span <= 0:
        span = max(abs(low) * 0.1, 1.0)
        high = low + span
    n = len(values)
    bins = max(10, min(60, int(round(math.sqrt(n) * 1.5))))
    return bins

def _next_charts_sheet_name(existing_names: Iterable[str]) -> str:
    if CHARTS_SHEET_NAME not in existing_names:
        return CHARTS_SHEET_NAME
    index = 1
    while True:
        candidate = f"{CHARTS_SHEET_NAME}_{index}"
        if candidate not in existing_names:
            return candidate
        index += 1
def build_chart_image(test: TestData) -> Optional[BytesIO]:
    if not test.measurements:
        return None
    low_declared, high_declared = test.representative_limits()
    axis_low_raw, axis_high_raw = determine_axis_limits(test.measurements, low_declared, high_declared)
    bin_count = choose_bin_count(test.measurements, axis_low_raw, axis_high_raw)
    bin_count = max(1, bin_count)

    span = axis_high_raw - axis_low_raw
    bin_width = span / bin_count if bin_count else 0.0
    padding = max(bin_width / 2.0, abs(span) * 0.05, 1e-6)
    if not math.isfinite(padding) or padding <= 0:
        padding = 0.5
    axis_low = axis_low_raw - padding
    axis_high = axis_high_raw + padding
    if not math.isfinite(axis_low):
        axis_low = axis_low_raw if math.isfinite(axis_low_raw) else -0.5
    if not math.isfinite(axis_high):
        axis_high = axis_high_raw if math.isfinite(axis_high_raw) else 0.5
    if axis_low >= axis_high:
        axis_low -= 0.5
        axis_high += 0.5

    bin_edges = np.linspace(axis_low, axis_high, bin_count + 1)

    fig, ax = plt.subplots(figsize=(CHART_WIDTH_IN, CHART_HEIGHT_IN))
    ax.hist(test.measurements, bins=bin_edges, color="#4F81BD", edgecolor="#1F497D", alpha=0.8)
    ax.set_title(test.title)
    axis_label = "Measurement"
    if test.unit:
        axis_label += f" ({test.unit})"
    ax.set_xlabel(axis_label)
    ax.set_ylabel("Count")
    ax.set_xlim(axis_low, axis_high)

    legend_entries: List[str] = []
    if low_declared is not None:
        ax.axvline(low_declared, color="#C0504D", linestyle="--", linewidth=2)
        legend_entries.append("Low Limit")
    if high_declared is not None:
        ax.axvline(high_declared, color="#9BBB59", linestyle="--", linewidth=2)
        legend_entries.append("High Limit")
    if legend_entries:
        ax.legend(legend_entries, loc="upper right")

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer

def add_charts_to_workbook(workbook_path: Path, output_path: Path, tests: Dict[Tuple[str, Optional[str]], TestData], charts_per_row: int) -> None:
    wb = load_workbook(workbook_path)
    sheet_name = _next_charts_sheet_name(wb.sheetnames)
    charts_ws = wb.create_sheet(sheet_name)
    charts_ws["A1"] = f"Charts generated by addCharts.py ({sheet_name})"
    charts_ws.freeze_panes = "A2"

    sorted_tests = sorted(tests.values(), key=lambda t: t.title.lower())
    charts_per_row = max(1, charts_per_row)

    for idx, test in enumerate(sorted_tests):
        image_stream = build_chart_image(test)
        if image_stream is None:
            continue
        img = XLImage(image_stream)
        img.width = CHART_WIDTH_PX
        img.height = CHART_HEIGHT_PX

        row_group = idx // charts_per_row
        col_group = idx % charts_per_row
        anchor_row = 2 + row_group * ROW_STRIDE
        anchor_col = 1 + col_group * COL_STRIDE
        anchor_cell = f"{get_column_letter(anchor_col)}{anchor_row}"
        charts_ws.add_image(img, anchor_cell)

    wb.save(output_path)
    wb.close()
def main() -> None:
    args = parse_args()
    workbook_path = args.workbook.expanduser().resolve()
    if not workbook_path.exists():
        raise SystemExit(f"Workbook not found: {workbook_path}")
    tests = load_measurement_tests(workbook_path)
    output_path = (args.output or workbook_path).expanduser().resolve()
    add_charts_to_workbook(workbook_path, output_path, tests, charts_per_row=args.charts_per_row)
    print(f"Added Charts sheet with {len(tests)} histogram(s) to {output_path}")


if __name__ == "__main__":
    main()


