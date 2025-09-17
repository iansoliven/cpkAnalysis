#!/usr/bin/env python3
"""
Combine worksheets from multiple Excel workbooks into a single workbook.

The script copies cell values and, by default, basic formatting (fonts, fills, borders,
column widths, row heights, merged cells, and frozen panes). Advanced Excel-specific
features such as pivot tables, slicers, or macros are not preserved.
"""

from __future__ import annotations

import argparse
import sys
from copy import copy
from pathlib import Path
from typing import List, Sequence, Set

from openpyxl import Workbook, load_workbook

EXCEL_SHEET_NAME_LIMIT = 31
INVALID_SHEET_CHARS = set('[]:*?/\\')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine worksheets from multiple Excel workbooks into a single file."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional paths to the source .xlsx files. Defaults to all .xlsx files in the directory."
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=Path,
        default=Path.cwd(),
        help="Directory to search when no explicit input files are provided."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("combined_workbook.xlsx"),
        help="Path for the merged workbook (default: combined_workbook.xlsx)."
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden worksheets from the source workbooks."
    )
    parser.add_argument(
        "--values-only",
        action="store_true",
        help="Copy only cell values and skip formatting to speed up the merge."
    )
    return parser.parse_args()

def sanitize_sheet_title(name: str) -> str:
    sanitized = ''.join('_' if ch in INVALID_SHEET_CHARS else ch for ch in name)
    sanitized = sanitized.rstrip("'").strip()
    return sanitized or "Sheet"

def build_unique_sheet_title(source_file: Path, sheet_name: str, used_titles: Set[str]) -> str:
    base = sanitize_sheet_title(f"{source_file.stem}_{sheet_name}")
    base = base[:EXCEL_SHEET_NAME_LIMIT] or "Sheet"
    candidate = base
    index = 1
    while candidate in used_titles:
        suffix = f"_{index}"
        allowed_length = EXCEL_SHEET_NAME_LIMIT - len(suffix)
        trimmed = base[:allowed_length] if allowed_length > 0 else base[:EXCEL_SHEET_NAME_LIMIT]
        candidate = (trimmed or "Sheet") + suffix
        index += 1
    used_titles.add(candidate)
    return candidate

def gather_sources(inputs: Sequence[Path], directory: Path, output_path: Path) -> List[Path]:
    if inputs:
        source_candidates: List[Path] = []
        missing: List[str] = []
        for path in inputs:
            expanded = path.expanduser()
            if not expanded.exists():
                missing.append(str(expanded))
                continue
            if expanded.is_dir():
                raise ValueError(f"Input path '{expanded}' is a directory. Provide .xlsx files instead.")
            source_candidates.append(expanded.resolve())
        if missing:
            raise FileNotFoundError(f"Source file(s) not found: {', '.join(missing)}")
    else:
        search_dir = directory.expanduser()
        if not search_dir.exists():
            raise FileNotFoundError(f"Directory not found: {search_dir}")
        if not search_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {search_dir}")
        source_candidates = sorted(search_dir.glob("*.xlsx"))

    output_resolved = output_path.expanduser().resolve()
    filtered: List[Path] = []
    seen = set()
    for candidate in source_candidates:
        if candidate.name.startswith("~$"):
            continue
        if candidate.suffix.lower() != ".xlsx":
            continue
        resolved = candidate.resolve()
        if resolved == output_resolved:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        filtered.append(resolved)

    if not filtered:
        raise FileNotFoundError("No .xlsx source files were found to merge.")
    return filtered

def copy_sheet_contents(source_ws, target_ws, copy_formatting: bool) -> None:
    if copy_formatting:
        if source_ws.sheet_properties.tabColor is not None:
            target_ws.sheet_properties.tabColor = copy(source_ws.sheet_properties.tabColor)
        if source_ws.freeze_panes:
            target_ws.freeze_panes = source_ws.freeze_panes

        for idx, row_dimension in source_ws.row_dimensions.items():
            target_dimension = target_ws.row_dimensions[idx]
            if row_dimension.height is not None:
                target_dimension.height = row_dimension.height
            if row_dimension.hidden is not None:
                target_dimension.hidden = row_dimension.hidden
            if row_dimension.outlineLevel:
                target_dimension.outlineLevel = row_dimension.outlineLevel

        for key, column_dimension in source_ws.column_dimensions.items():
            target_dimension = target_ws.column_dimensions[key]
            if column_dimension.width is not None:
                target_dimension.width = column_dimension.width
            if column_dimension.hidden is not None:
                target_dimension.hidden = column_dimension.hidden
            if column_dimension.outlineLevel:
                target_dimension.outlineLevel = column_dimension.outlineLevel
            if column_dimension.bestFit is not None:
                target_dimension.bestFit = column_dimension.bestFit

        if source_ws.auto_filter and source_ws.auto_filter.ref:
            target_ws.auto_filter.ref = source_ws.auto_filter.ref

    for row in source_ws.iter_rows():
        for cell in row:
            target_cell = target_ws.cell(row=cell.row, column=cell.col_idx, value=cell.value)
            if copy_formatting and cell.has_style:
                target_cell.font = copy(cell.font)
                target_cell.border = copy(cell.border)
                target_cell.fill = copy(cell.fill)
                target_cell.number_format = cell.number_format
                target_cell.protection = copy(cell.protection)
                target_cell.alignment = copy(cell.alignment)
            if cell.hyperlink:
                target_cell.hyperlink = cell.hyperlink.target
            if copy_formatting and cell.comment:
                target_cell.comment = copy(cell.comment)

    for merged in source_ws.merged_cells.ranges:
        target_ws.merge_cells(str(merged))

def main() -> int:
    args = parse_args()
    try:
        sources = gather_sources(args.inputs, args.directory, args.output)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    destination_path = args.output.expanduser().resolve()
    workbook = Workbook()
    workbook.remove(workbook.active)
    used_titles: Set[str] = set()
    copied_sheets = 0
    skipped_hidden = 0
    failures: List[str] = []

    for source_file in sources:
        print(f"Processing {source_file}...")
        try:
            source_workbook = load_workbook(source_file, data_only=False, read_only=False)
        except Exception as exc:
            failures.append(f"{source_file}: {exc}")
            continue

        try:
            for sheet in source_workbook.worksheets:
                if not args.include_hidden and sheet.sheet_state != "visible":
                    skipped_hidden += 1
                    continue
                title = build_unique_sheet_title(source_file, sheet.title, used_titles)
                target_sheet = workbook.create_sheet(title=title)
                target_sheet.sheet_state = sheet.sheet_state
                copy_sheet_contents(sheet, target_sheet, copy_formatting=not args.values_only)
                copied_sheets += 1
        finally:
            source_workbook.close()

    if copied_sheets == 0:
        print("No worksheets were copied. Nothing to merge.", file=sys.stderr)
        if failures:
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
        return 1

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(destination_path)
    except PermissionError:
        print(f"Permission denied when writing {destination_path}. Close the file if it is open and retry.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to save workbook: {exc}", file=sys.stderr)
        return 1

    print(f"Merged {copied_sheets} worksheet(s) into {destination_path}")
    if skipped_hidden and not args.include_hidden:
        print(f"Skipped {skipped_hidden} hidden worksheet(s). Use --include-hidden to copy them.", file=sys.stderr)
    if failures:
        print("Some workbooks could not be processed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
