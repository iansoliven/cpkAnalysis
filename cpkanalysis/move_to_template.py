"""Stage that copies CPK Report data into the template sheet.

The sheet provided by the template is expected to have a header row whose
labels match those found on the CPK Report sheet generated earlier in the
pipeline. Matching headers are copied column-by-column, bringing over values,
number formats, and hyperlinks while appending rows as needed.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Dict, Iterable, Optional

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

SOURCE_SHEET = "CPK Report"


def run(workbook_path: Path, sheet_name: Optional[str] = None) -> str:
    """Copy CPK Report contents into the template sheet and return the target sheet name."""
    resolved = workbook_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Workbook not found: {resolved}")

    workbook = load_workbook(resolved)
    if SOURCE_SHEET not in workbook.sheetnames:
        raise ValueError(f"Source sheet '{SOURCE_SHEET}' not found in {resolved}")
    source_ws = workbook[SOURCE_SHEET]

    target_ws = _select_target_sheet(workbook, sheet_name)
    target_sheetname = target_ws.title

    source_headers = _header_map(source_ws)
    target_headers = _header_map(target_ws)
    shared_headers = [header for header in source_headers if header in target_headers]

    if not shared_headers:
        raise ValueError("No matching headers between CPK Report and target sheet.")

    max_row = source_ws.max_row
    for header in shared_headers:
        src_col = source_headers[header]
        tgt_col = target_headers[header]
        _copy_column(source_ws, target_ws, src_col, tgt_col, max_row)

    workbook.save(resolved)
    workbook.close()
    return target_sheetname


def _select_target_sheet(workbook, sheet_name: Optional[str]) -> Worksheet:
    if sheet_name:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"Target sheet '{sheet_name}' not found in workbook.")
        return workbook[sheet_name]

    for candidate in workbook.sheetnames:
        if candidate != SOURCE_SHEET:
            return workbook[candidate]
    raise ValueError("No suitable target sheet found (workbook only contains CPK Report).")


def _header_map(sheet: Worksheet) -> Dict[str, int]:
    headers: Dict[str, int] = {}
    for cell in sheet[1]:
        value = str(cell.value).strip() if cell.value is not None else ""
        if value:
            headers[value] = cell.column
    return headers


def _copy_column(
    source: Worksheet,
    target: Worksheet,
    source_column: int,
    target_column: int,
    max_row: int,
) -> None:
    for row in range(2, max_row + 1):
        src_cell = source.cell(row=row, column=source_column)
        tgt_cell = target.cell(row=row, column=target_column)

        tgt_cell.value = src_cell.value
        tgt_cell.number_format = src_cell.number_format
        tgt_cell.font = copy(src_cell.font)
        tgt_cell.fill = copy(src_cell.fill)
        tgt_cell.border = copy(src_cell.border)
        tgt_cell.alignment = copy(src_cell.alignment)
        tgt_cell.protection = copy(src_cell.protection)
        tgt_cell.hyperlink = src_cell.hyperlink if src_cell.hyperlink else None
        tgt_cell.style = src_cell.style
