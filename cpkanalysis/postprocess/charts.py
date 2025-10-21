"""Utilities for regenerating charts after post-processing updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter

from .. import mpl_charts
from ..workbook_builder import (
    AXIS_META_SHEET,
    IMAGE_ANCHOR_COLUMN,
    ROW_STRIDE,
    _compute_axis_bounds,
    _ensure_row_height,
    _place_image,
    _safe_sheet_name,
    _remove_axis_tracking_sheet,
    _write_axis_ranges,
)
from .context import PostProcessContext
from .sheet_utils import build_header_map, find_rows_by_test, get_cell, normalize_header

__all__ = ["refresh_tests"]

SPEC_COLOR = "#4B6CB7"
WHAT_IF_COLOR = "#FFA500"
PROPOSED_COLOR = "#008B8B"
CHART_PREFIXES = ("Histogram", "CDF", "TimeSeries")


@dataclass
class LimitInfo:
    stdf_lower: float | None = None
    stdf_upper: float | None = None
    spec_lower: float | None = None
    spec_upper: float | None = None
    what_lower: float | None = None
    what_upper: float | None = None
    proposed_lower: float | None = None
    proposed_upper: float | None = None


def _ensure_chart_state(metadata: Dict[str, Any]) -> Dict[str, Any]:
    state = metadata.setdefault("post_processing_state", {})
    charts = state.setdefault("chart_positions", {})
    return charts


def _make_test_key(file_key: str, test_name: str, test_number: str) -> str:
    return f"{file_key}|{test_name}|{test_number}"


def _resolve_test_index(chart_state: Dict[str, Any], file_key: str, test_key: str) -> int:
    order_map: Dict[str, Dict[str, int]] = chart_state.setdefault("_order", {})
    file_order = order_map.setdefault(file_key, {})
    if test_key not in file_order:
        file_order[test_key] = len(file_order)
    return file_order[test_key]


def _record_prefix_index(chart_state: Dict[str, Any], prefix: str, file_key: str, test_key: str, index: int) -> None:
    prefix_map: Dict[str, Dict[str, int]] = chart_state.setdefault(prefix, {})
    file_map = prefix_map.setdefault(file_key, {})
    file_map[test_key] = index


def _anchor_row_for_index(index: int) -> int:
    return 2 + (index * ROW_STRIDE)


def _get_plot_sheet(workbook, file_key: str, prefix: str):
    sheet_name = _safe_sheet_name(f"{prefix}_{file_key}")
    if sheet_name in workbook.sheetnames:
        sheet = workbook[sheet_name]
    else:
        sheet = workbook.create_sheet(sheet_name)
    sheet.sheet_view.showGridLines = False
    header_text = f"{prefix} plots for {file_key}"
    if sheet.cell(row=1, column=1).value != header_text:
        sheet.cell(row=1, column=1, value=header_text)
    label_col = get_column_letter(1)
    image_col = get_column_letter(IMAGE_ANCHOR_COLUMN)
    if sheet.column_dimensions[label_col].width is None or sheet.column_dimensions[label_col].width < 28:
        sheet.column_dimensions[label_col].width = 28
    if sheet.column_dimensions[image_col].width is None or sheet.column_dimensions[image_col].width < 60:
        sheet.column_dimensions[image_col].width = 60
    return sheet


def _load_axis_ranges(workbook) -> Dict[Tuple[str, str, str], Dict[str, float | None]]:
    if AXIS_META_SHEET not in workbook.sheetnames:
        return {}
    ws = workbook[AXIS_META_SHEET]
    ranges: Dict[Tuple[str, str, str], Dict[str, float | None]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or all(cell is None for cell in row):
            continue
        file_key = _safe_text(row[0])
        test_name = _safe_text(row[1])
        test_number = _safe_text(row[2])
        key = (file_key, test_name, test_number)
        ranges[key] = {
            "data_min": _safe_float(row[3]),
            "data_max": _safe_float(row[4]),
            "lower_limit": _safe_float(row[5]),
            "upper_limit": _safe_float(row[6]),
            "axis_min": _safe_float(row[7]),
            "axis_max": _safe_float(row[8]),
        }
    return ranges


def _rewrite_axis_ranges(workbook, axis_ranges: Dict[Tuple[str, str, str], Dict[str, float | None]]) -> None:
    _remove_axis_tracking_sheet(workbook)
    if not axis_ranges:
        return
    _write_axis_ranges(workbook, axis_ranges)


def _replace_chart_image(sheet, index: int, anchor_row: int, label: str, image_bytes: bytes) -> None:
    if index < len(sheet._images):
        sheet._images.pop(index)
    _place_image(sheet, image_bytes, anchor_row, label, position=index)
    _ensure_row_height(sheet, anchor_row)


def refresh_tests(
    context: PostProcessContext,
    tests: Sequence,
    *,
    include_spec: bool = False,
    include_proposed: bool = False,
) -> None:
    """Regenerate plot sheets to reflect updated limits."""
    workbook = context.workbook()
    summary_df = context.summary_frame(refresh=True)
    measurements_df = context.measurements_frame(refresh=True)
    limits_df = context.limits_frame(refresh=True)
    template_ws = context.template_sheet()

    limit_lookup = _collect_limit_info(limits_df, template_ws)
    summary_lookup = {
        (
            _safe_text(row.get("File")),
            _safe_text(row.get("Test Name")),
            _safe_text(row.get("Test Number")),
        ): row
        for _, row in summary_df.iterrows()
    }

    chart_state = _ensure_chart_state(context.metadata)

    target_keys: Optional[Set[Tuple[str, str, str]]] = None
    if tests:
        target_keys = {
            (
                _safe_text(getattr(test, "file", "")),
                _safe_text(getattr(test, "test_name", "")),
                _safe_text(getattr(test, "test_number", "")),
            )
            for test in tests
        }

    if measurements_df.empty:
        existing_axis = _load_axis_ranges(workbook) if target_keys else {}
        _rewrite_axis_ranges(workbook, existing_axis)
        context.invalidate_frames("measurements")
        return

    if not target_keys:
        _refresh_all_tests(
            context,
            workbook,
            summary_lookup,
            measurements_df,
            limit_lookup,
            include_spec,
            include_proposed,
            chart_state,
        )
    else:
        _refresh_subset_tests(
            context,
            workbook,
            summary_lookup,
            measurements_df,
            limit_lookup,
            include_spec,
            include_proposed,
            chart_state,
            target_keys,
        )

    context.invalidate_frames("measurements")


def _refresh_all_tests(
    context: PostProcessContext,
    workbook,
    summary_lookup: Dict[Tuple[str, str, str], pd.Series],
    measurements_df: pd.DataFrame,
    limit_lookup: Dict[Tuple[str, str], LimitInfo],
    include_spec: bool,
    include_proposed: bool,
    chart_state: Dict[str, Any],
) -> None:
    _prune_plot_sheets(workbook)
    chart_state.clear()
    order_map: Dict[str, Dict[str, int]] = chart_state.setdefault("_order", {})
    axis_ranges: Dict[Tuple[str, str, str], Dict[str, float | None]] = {}
    next_index: Dict[str, int] = {}

    grouped = measurements_df.groupby(["File", "Test Name", "Test Number"], sort=False, dropna=False)
    for (file_name, test_name, test_number), group in grouped:
        file_key = _safe_text(file_name)
        safe_test_name = _safe_text(test_name)
        safe_test_number = _safe_text(test_number)
        test_key = (file_key, safe_test_name, safe_test_number)
        test_key_str = _make_test_key(file_key, safe_test_name, safe_test_number)

        numeric_values = pd.to_numeric(group["Value"], errors="coerce")
        valid_mask = numeric_values.notna() & np.isfinite(numeric_values)
        if not bool(valid_mask.any()):
            continue
        filtered_group = group.loc[valid_mask].copy()

        serial_numbers = np.arange(1, len(filtered_group) + 1, dtype=float)
        filtered_group["_serial_number"] = serial_numbers
        sort_keys = ["_serial_number"]
        if "Measurement Index" in filtered_group.columns:
            sort_keys.append("Measurement Index")
        elif "measurement_index" in filtered_group.columns:
            sort_keys.append("measurement_index")
        filtered_group.sort_values(by=sort_keys, kind="mergesort", inplace=True)
        serial_numbers = filtered_group["_serial_number"].to_numpy()
        values = pd.to_numeric(filtered_group["Value"], errors="coerce").to_numpy()

        limit_info = limit_lookup.get((safe_test_name, safe_test_number), LimitInfo())
        markers = _build_markers(limit_info, include_spec=include_spec, include_proposed=include_proposed)

        data_limits = [m.value for m in markers if m.value is not None]
        if include_spec:
            data_limits.extend(
                value
                for value in (
                    limit_info.spec_lower,
                    limit_info.spec_upper,
                    limit_info.what_lower,
                    limit_info.what_upper,
                )
                if value is not None
            )
        if include_proposed:
            data_limits.extend(
                value
                for value in (
                    limit_info.proposed_lower,
                    limit_info.proposed_upper,
                )
                if value is not None
            )
        lower_limit = limit_info.stdf_lower
        upper_limit = limit_info.stdf_upper

        axis_min, axis_max, data_min, data_max = _bounds_with_markers(values, lower_limit, upper_limit, data_limits)
        x_range = (axis_min, axis_max)
        y_range = (axis_min, axis_max)

        summary_row = summary_lookup.get(test_key)
        cpk = None
        unit_label = ""
        if summary_row is not None:
            cpk = _safe_float(summary_row.get("CPK"))
            unit_label = _safe_text(summary_row.get("Unit"))

        test_label = safe_test_name if not safe_test_number else f"{safe_test_name} (Test {safe_test_number})"
        index = next_index.get(file_key, 0)
        next_index[file_key] = index + 1

        order_map.setdefault(file_key, {})[test_key_str] = index
        anchor_row = _anchor_row_for_index(index)

        if context.analysis_inputs.generate_histogram:
            sheet = _get_plot_sheet(workbook, file_key, "Histogram")
            image_bytes = mpl_charts.render_histogram(
                values,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                x_range=x_range,
                test_label=test_label,
                cpk=cpk,
                unit_label=unit_label,
                extra_markers=markers,
                title_font_size=10,
                cpk_font_size=8,
            )
            _replace_chart_image(sheet, index, anchor_row, test_label, image_bytes)
            _record_prefix_index(chart_state, "Histogram", file_key, test_key_str, index)

        if context.analysis_inputs.generate_cdf:
            sheet_cdf = _get_plot_sheet(workbook, file_key, "CDF")
            image_bytes = mpl_charts.render_cdf(
                values,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                x_range=x_range,
                test_label=test_label,
                cpk=cpk,
                unit_label=unit_label,
                extra_markers=markers,
                title_font_size=10,
                cpk_font_size=8,
            )
            _replace_chart_image(sheet_cdf, index, anchor_row, test_label, image_bytes)
            _record_prefix_index(chart_state, "CDF", file_key, test_key_str, index)

        if context.analysis_inputs.generate_time_series:
            sheet_ts = _get_plot_sheet(workbook, file_key, "TimeSeries")
            image_bytes = mpl_charts.render_time_series(
                x=serial_numbers,
                y=values,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                y_range=y_range,
                test_label=test_label,
                cpk=cpk,
                unit_label=unit_label,
                extra_markers=_horizontalised_markers(markers),
                title_font_size=10,
                cpk_font_size=8,
            )
            _replace_chart_image(sheet_ts, index, anchor_row, test_label, image_bytes)
            _record_prefix_index(chart_state, "TimeSeries", file_key, test_key_str, index)

        axis_ranges[test_key] = {
            "data_min": _safe_float(data_min),
            "data_max": _safe_float(data_max),
            "lower_limit": _safe_float(lower_limit),
            "upper_limit": _safe_float(upper_limit),
            "axis_min": _safe_float(axis_min),
            "axis_max": _safe_float(axis_max),
        }

    _rewrite_axis_ranges(workbook, axis_ranges)


def _refresh_subset_tests(
    context: PostProcessContext,
    workbook,
    summary_lookup: Dict[Tuple[str, str, str], pd.Series],
    measurements_df: pd.DataFrame,
    limit_lookup: Dict[Tuple[str, str], LimitInfo],
    include_spec: bool,
    include_proposed: bool,
    chart_state: Dict[str, Any],
    target_keys: Set[Tuple[str, str, str]],
) -> None:
    existing_axis = _load_axis_ranges(workbook)
    axis_ranges = dict(existing_axis)
    order_map: Dict[str, Dict[str, int]] = chart_state.setdefault("_order", {})
    updated = False

    grouped = measurements_df.groupby(["File", "Test Name", "Test Number"], sort=False, dropna=False)
    for (file_name, test_name, test_number), group in grouped:
        file_key = _safe_text(file_name)
        safe_test_name = _safe_text(test_name)
        safe_test_number = _safe_text(test_number)
        test_key = (file_key, safe_test_name, safe_test_number)
        if test_key not in target_keys:
            continue

        numeric_values = pd.to_numeric(group["Value"], errors="coerce")
        valid_mask = numeric_values.notna() & np.isfinite(numeric_values)
        if not bool(valid_mask.any()):
            continue
        filtered_group = group.loc[valid_mask].copy()

        serial_numbers = np.arange(1, len(filtered_group) + 1, dtype=float)
        filtered_group["_serial_number"] = serial_numbers
        sort_keys = ["_serial_number"]
        if "Measurement Index" in filtered_group.columns:
            sort_keys.append("Measurement Index")
        elif "measurement_index" in filtered_group.columns:
            sort_keys.append("measurement_index")
        filtered_group.sort_values(by=sort_keys, kind="mergesort", inplace=True)
        serial_numbers = filtered_group["_serial_number"].to_numpy()
        values = pd.to_numeric(filtered_group["Value"], errors="coerce").to_numpy()

        limit_info = limit_lookup.get((safe_test_name, safe_test_number), LimitInfo())
        markers = _build_markers(limit_info, include_spec=include_spec, include_proposed=include_proposed)

        data_limits = [m.value for m in markers if m.value is not None]
        if include_spec:
            data_limits.extend(
                value
                for value in (
                    limit_info.spec_lower,
                    limit_info.spec_upper,
                    limit_info.what_lower,
                    limit_info.what_upper,
                )
                if value is not None
            )
        if include_proposed:
            data_limits.extend(
                value
                for value in (
                    limit_info.proposed_lower,
                    limit_info.proposed_upper,
                )
                if value is not None
            )
        lower_limit = limit_info.stdf_lower
        upper_limit = limit_info.stdf_upper

        axis_min, axis_max, data_min, data_max = _bounds_with_markers(values, lower_limit, upper_limit, data_limits)
        x_range = (axis_min, axis_max)
        y_range = (axis_min, axis_max)

        summary_row = summary_lookup.get(test_key)
        cpk = None
        unit_label = ""
        if summary_row is not None:
            cpk = _safe_float(summary_row.get("CPK"))
            unit_label = _safe_text(summary_row.get("Unit"))

        test_label = safe_test_name if not safe_test_number else f"{safe_test_name} (Test {safe_test_number})"
        test_key_str = _make_test_key(file_key, safe_test_name, safe_test_number)
        index = _resolve_test_index(chart_state, file_key, test_key_str)
        order_map.setdefault(file_key, {})[test_key_str] = index
        anchor_row = _anchor_row_for_index(index)

        if context.analysis_inputs.generate_histogram:
            sheet = _get_plot_sheet(workbook, file_key, "Histogram")
            image_bytes = mpl_charts.render_histogram(
                values,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                x_range=x_range,
                test_label=test_label,
                cpk=cpk,
                unit_label=unit_label,
                extra_markers=markers,
                title_font_size=10,
                cpk_font_size=8,
            )
            _replace_chart_image(sheet, index, anchor_row, test_label, image_bytes)
            _record_prefix_index(chart_state, "Histogram", file_key, test_key_str, index)

        if context.analysis_inputs.generate_cdf:
            sheet_cdf = _get_plot_sheet(workbook, file_key, "CDF")
            image_bytes = mpl_charts.render_cdf(
                values,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                x_range=x_range,
                test_label=test_label,
                cpk=cpk,
                unit_label=unit_label,
                extra_markers=markers,
                title_font_size=10,
                cpk_font_size=8,
            )
            _replace_chart_image(sheet_cdf, index, anchor_row, test_label, image_bytes)
            _record_prefix_index(chart_state, "CDF", file_key, test_key_str, index)

        if context.analysis_inputs.generate_time_series:
            sheet_ts = _get_plot_sheet(workbook, file_key, "TimeSeries")
            image_bytes = mpl_charts.render_time_series(
                x=serial_numbers,
                y=values,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                y_range=y_range,
                test_label=test_label,
                cpk=cpk,
                unit_label=unit_label,
                extra_markers=_horizontalised_markers(markers),
                title_font_size=10,
                cpk_font_size=8,
            )
            _replace_chart_image(sheet_ts, index, anchor_row, test_label, image_bytes)
            _record_prefix_index(chart_state, "TimeSeries", file_key, test_key_str, index)

        axis_ranges[test_key] = {
            "data_min": _safe_float(data_min),
            "data_max": _safe_float(data_max),
            "lower_limit": _safe_float(lower_limit),
            "upper_limit": _safe_float(upper_limit),
            "axis_min": _safe_float(axis_min),
            "axis_max": _safe_float(axis_max),
        }
        updated = True

    if updated:
        _rewrite_axis_ranges(workbook, axis_ranges)


def _collect_limit_info(limits_df: pd.DataFrame, template_ws) -> Dict[Tuple[str, str], LimitInfo]:
    lookup: Dict[Tuple[str, str], LimitInfo] = {}
    if not limits_df.empty:
        for _, row in limits_df.iterrows():
            key = (
                _safe_text(_row_value(row, ["Test name", "test_name", "Test Name"])),
                _safe_text(_row_value(row, ["Test number", "test_number", "Test Number"])),
            )
            info = lookup.setdefault(key, LimitInfo())
            info.stdf_lower = _safe_float(_row_value(row, ["STDF Lower Limit", "stdf_lower"]))
            info.stdf_upper = _safe_float(_row_value(row, ["STDF Upper Limit", "stdf_upper"]))
            info.spec_lower = _safe_float(_row_value(row, ["Spec Lower Limit", "spec_lower"]))
            info.spec_upper = _safe_float(_row_value(row, ["Spec Upper Limit", "spec_upper"]))
            info.what_lower = _safe_float(_row_value(row, ["User What-If Lower Limit", "what_if_lower"]))
            info.what_upper = _safe_float(_row_value(row, ["User What-If Upper Limit", "what_if_upper"]))

    template_header_row, template_headers = build_header_map(template_ws)
    ll_ate_col = _resolve_template_column(template_headers, ["LL_ATE", "LL ATE", "Lower ATE"])
    ul_ate_col = _resolve_template_column(template_headers, ["UL_ATE", "UL ATE", "Upper ATE"])
    spec_lower_col = _resolve_template_column(template_headers, ["Spec Lower", "Spec Lower Limit"])
    spec_upper_col = _resolve_template_column(template_headers, ["Spec Upper", "Spec Upper Limit"])
    what_lower_col = _resolve_template_column(template_headers, ["What-if Lower", "What If Lower", "What-If Lower"])
    what_upper_col = _resolve_template_column(template_headers, ["What-if Upper", "What If Upper", "What-If Upper"])
    prop_lower_col = _resolve_template_column(template_headers, ["LL_PROP", "LL Proposed"])
    prop_upper_col = _resolve_template_column(template_headers, ["UL_PROP", "UL Proposed"])
    test_name_col = _resolve_template_column(template_headers, ["TEST NAME", "Test Name"])
    test_number_col = _resolve_template_column(template_headers, ["TEST NUM", "Test Number"])

    for row_idx in range(template_header_row + 1, template_ws.max_row + 1):
        test_name = _safe_text(get_cell(template_ws, row_idx, test_name_col)) if test_name_col else ""
        test_number = _safe_text(get_cell(template_ws, row_idx, test_number_col)) if test_number_col else ""
        if not test_name and not test_number:
            continue
        key = (test_name, test_number)
        info = lookup.setdefault(key, LimitInfo())
        if ll_ate_col:
            value = _safe_float(get_cell(template_ws, row_idx, ll_ate_col))
            if value is not None:
                info.stdf_lower = value
        if ul_ate_col:
            value = _safe_float(get_cell(template_ws, row_idx, ul_ate_col))
            if value is not None:
                info.stdf_upper = value
        if spec_lower_col:
            value = _safe_float(get_cell(template_ws, row_idx, spec_lower_col))
            if value is not None:
                info.spec_lower = value
        if spec_upper_col:
            value = _safe_float(get_cell(template_ws, row_idx, spec_upper_col))
            if value is not None:
                info.spec_upper = value
        if what_lower_col:
            value = _safe_float(get_cell(template_ws, row_idx, what_lower_col))
            if value is not None:
                info.what_lower = value
        if what_upper_col:
            value = _safe_float(get_cell(template_ws, row_idx, what_upper_col))
            if value is not None:
                info.what_upper = value
        if prop_lower_col:
            value = _safe_float(get_cell(template_ws, row_idx, prop_lower_col))
            if value is not None:
                info.proposed_lower = value
        if prop_upper_col:
            value = _safe_float(get_cell(template_ws, row_idx, prop_upper_col))
            if value is not None:
                info.proposed_upper = value

    return lookup


def _resolve_template_column(header_map: Dict[str, int], aliases: Sequence[str]) -> Optional[int]:
    for alias in aliases:
        normalized = normalize_header(alias)
        if normalized in header_map:
            return header_map[normalized]
    return None


def _prune_plot_sheets(workbook) -> None:
    for sheet_name in list(workbook.sheetnames):
        if sheet_name.startswith(("Histogram", "CDF", "TimeSeries")):
            del workbook[sheet_name]
    _remove_axis_tracking_sheet(workbook)


def _build_markers(
    info: LimitInfo,
    *,
    include_spec: bool,
    include_proposed: bool,
) -> List[mpl_charts.ChartMarker]:
    markers: List[mpl_charts.ChartMarker] = []
    if include_spec:
        if info.spec_lower is not None:
            markers.append(mpl_charts.ChartMarker("Spec Lower", info.spec_lower, "vertical", SPEC_COLOR))
        if info.spec_upper is not None:
            markers.append(mpl_charts.ChartMarker("Spec Upper", info.spec_upper, "vertical", SPEC_COLOR))
        if info.what_lower is not None:
            markers.append(mpl_charts.ChartMarker("What-If Lower", info.what_lower, "vertical", WHAT_IF_COLOR))
        if info.what_upper is not None:
            markers.append(mpl_charts.ChartMarker("What-If Upper", info.what_upper, "vertical", WHAT_IF_COLOR))
    if include_proposed:
        if info.proposed_lower is not None:
            markers.append(mpl_charts.ChartMarker("Proposed Lower", info.proposed_lower, "vertical", PROPOSED_COLOR, linestyle="-"))
        if info.proposed_upper is not None:
            markers.append(mpl_charts.ChartMarker("Proposed Upper", info.proposed_upper, "vertical", PROPOSED_COLOR, linestyle="-"))
    return markers


def _horizontalised_markers(markers: Sequence[mpl_charts.ChartMarker]) -> List[mpl_charts.ChartMarker]:
    converted = []
    for marker in markers:
        if marker.orientation == "horizontal":
            converted.append(marker)
        else:
            converted.append(mpl_charts.ChartMarker(marker.label, marker.value, "horizontal", marker.color, marker.linestyle, marker.linewidth))
    return converted


def _bounds_with_markers(values: np.ndarray, lower, upper, markers: Sequence[float]) -> Tuple[float, float, float, float]:
    axis_min, axis_max, data_min, data_max = _compute_axis_bounds(values, lower, upper, desired_ticks=10)
    candidates = [m for m in markers if m is not None and np.isfinite(m)]
    if candidates:
        min_marker = float(min(candidates))
        max_marker = float(max(candidates))
        axis_min = float(min(axis_min, min_marker))
        axis_max = float(max(axis_max, max_marker))
        span = axis_max - axis_min
        if not np.isfinite(span) or span <= 0:
            span = max(abs(axis_min), abs(axis_max), 1.0)
        padding = max(span * 0.02, 1e-6)
        axis_min = float(min(axis_min, min_marker - padding))
        axis_max = float(max(axis_max, max_marker + padding))
    return axis_min, axis_max, data_min, data_max


def _safe_float(value) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result):
        return None
    return result


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        return "" if text.lower() == "nan" else text
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _row_value(row: pd.Series, candidates: Sequence[str]):
    for candidate in candidates:
        if candidate in row.index:
            value = row.get(candidate)
            if pd.notna(value):
                return value
    return None
