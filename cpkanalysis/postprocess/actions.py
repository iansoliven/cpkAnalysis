"""Action implementations for post-processing menu."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .context import PostProcessContext
from .io_adapters import PostProcessIO
from . import sheet_utils
from . import charts

__all__ = [
    "ActionCancelled",
    "update_stdf_limits",
    "apply_spec_limits",
    "calculate_proposed_limits",
]


class ActionCancelled(RuntimeError):
    """Raised when the user aborts an action."""


@dataclass(frozen=True)
class TestDescriptor:
    file: str
    test_name: str
    test_number: str
    unit: str
    mean: float | None
    stdev: float | None
    cpk: float | None

    def key(self) -> str:
        return f"{self.file}|{self.test_name}|{self.test_number}"

    def label(self) -> str:
        suffix = f" (Test {self.test_number})" if self.test_number else ""
        file_part = f" [{self.file}]" if self.file else ""
        return f"{self.test_name}{suffix}{file_part}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_test_descriptors(context: PostProcessContext) -> List[TestDescriptor]:
    summary = context.summary_frame()
    if summary.empty:
        return []
    descriptors: List[TestDescriptor] = []
    for _, row in summary.iterrows():
        descriptors.append(
            TestDescriptor(
                file=str(row.get("File", "") or ""),
                test_name=str(row.get("Test Name", "") or ""),
                test_number=str(row.get("Test Number", "") or ""),
                unit=str(row.get("Unit", "") or ""),
                mean=_safe_float(row.get("MEAN")),
                stdev=_safe_float(row.get("STDEV")),
                cpk=_safe_float(row.get("CPK")),
            )
        )
    return descriptors


def _safe_float(value) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _prompt_scope(
    io: PostProcessIO,
    params: Optional[dict],
    *,
    allow_single: bool = True,
    require_target: bool = False,
    default_target: float | None = None,
) -> dict:
    scope = (params or {}).get("scope")
    if scope not in {"all", "single"}:
        options = ["All tests"]
        if allow_single:
            options.append("Single test")
        choice = io.prompt_choice("Apply to:", options)
        scope = "all" if choice == 0 else "single"

    target_cpk = (params or {}).get("target_cpk")
    if require_target or (params is None):
        prompt = "Target CPK (blank to skip):" if not require_target else "Target CPK:"
        default_text = "" if default_target is None else str(default_target)
        if require_target:
            while target_cpk is None:
                value = io.prompt(prompt, default=default_text)
                value = value.strip()
                if not value:
                    io.warn("Target CPK is required.")
                    continue
                try:
                    target_cpk = float(value)
                    if target_cpk <= 0:
                        io.warn("Target CPK must be positive.")
                        target_cpk = None
                except ValueError:
                    io.warn("Enter a numeric value.")
        else:
            if target_cpk is None:
                value = io.prompt(prompt, default=default_text)
                value = value.strip()
                if value:
                    try:
                        target_cpk = float(value)
                        if target_cpk <= 0:
                            io.warn("Target CPK must be positive.")
                            target_cpk = None
                    except ValueError:
                        io.warn("Ignoring invalid CPK entry.")
                        target_cpk = None

    selection = (params or {}).get("test_key")

    return {"scope": scope, "target_cpk": target_cpk, "test_key": selection}


def _resolve_tests(
    descriptors: Sequence[TestDescriptor],
    context: PostProcessContext,
    io: PostProcessIO,
    resolved: dict,
) -> List[TestDescriptor]:
    if resolved["scope"] == "all":
        return list(descriptors)
    if not descriptors:
        raise ActionCancelled("No tests available.")
    test_map = {descriptor.key(): descriptor for descriptor in descriptors}
    if resolved.get("test_key") in test_map:
        return [test_map[resolved["test_key"]]]

    labels = [desc.label() for desc in descriptors]
    choice = io.prompt_choice("Select test:", labels)
    descriptor = descriptors[choice]
    resolved["test_key"] = descriptor.key()
    return [descriptor]


def _summaries_by_key(summary: pd.DataFrame) -> Dict[str, pd.Series]:
    mapping: Dict[str, pd.Series] = {}
    for _, row in summary.iterrows():
        key = f"{row.get('File','')}|{row.get('Test Name','')}|{row.get('Test Number','')}"
        mapping[key] = row
    return mapping


def _warn_if_missing(io: PostProcessIO, warnings: List[str], message: str) -> None:
    warnings.append(message)
    io.warn(message)


def _resolve_column(header_map: Dict[str, int], aliases: Iterable[str]) -> Optional[int]:
    for alias in aliases:
        normalized = sheet_utils.normalize_header(alias)
        if normalized in header_map:
            return header_map[normalized]
    return None


def _tests_to_strings(tests: Iterable[TestDescriptor]) -> List[str]:
    return [test.key() for test in tests]


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def update_stdf_limits(context: PostProcessContext, io: PostProcessIO, params: Optional[dict]) -> dict:
    descriptors = _build_test_descriptors(context)
    if not descriptors:
        raise ActionCancelled("Summary sheet is empty.")

    resolved = _prompt_scope(io, params, allow_single=True, require_target=False)
    selected_tests = _resolve_tests(descriptors, context, io, resolved)

    summary_df = context.summary_frame()
    summary_lookup = _summaries_by_key(summary_df)

    template_ws = context.template_sheet()
    template_header_row, template_headers = sheet_utils.build_header_map(template_ws)
    ll_column = _resolve_column(template_headers, ["LL_ATE", "LL ATE", "Lower ATE", "LL"])
    ul_column = _resolve_column(template_headers, ["UL_ATE", "UL ATE", "Upper ATE", "UL"])
    if ll_column is None or ul_column is None:
        raise ActionCancelled("Template sheet missing LL_ATE or UL_ATE columns.")

    limits_ws = context.workbook()["Test List and Limits"]
    limits_header_row, limits_headers = sheet_utils.build_header_map(limits_ws)
    stdf_lower_col = _resolve_column(limits_headers, ["STDF Lower Limit", "STDF Lower"])
    stdf_upper_col = _resolve_column(limits_headers, ["STDF Upper Limit", "STDF Upper"])

    warnings: List[str] = []
    updated_tests: List[TestDescriptor] = []
    for descriptor in selected_tests:
        row = summary_lookup.get(descriptor.key())
        if row is None:
            _warn_if_missing(
                io,
                warnings,
                f"Summary entry not found for test {descriptor.label()} – skipping.",
            )
            continue

        mean = _safe_float(row.get("MEAN"))
        stdev = _safe_float(row.get("STDEV"))
        if stdev is None or stdev <= 0:
            _warn_if_missing(
                io,
                warnings,
                f"Cannot compute limits for {descriptor.label()} (non-positive STDEV).",
            )
            continue

        target_cpk = resolved.get("target_cpk")
        if target_cpk and target_cpk > 0:
            half_width = 3.0 * target_cpk * stdev
            lower_limit = mean - half_width if mean is not None else None
            upper_limit = mean + half_width if mean is not None else None
        else:
            lower_limit = _safe_float(row.get("LL_2CPK")) or _safe_float(row.get("LL_3IQR"))
            upper_limit = _safe_float(row.get("UL_2CPK")) or _safe_float(row.get("UL_3IQR"))

        if lower_limit is None or upper_limit is None:
            _warn_if_missing(
                io,
                warnings,
                f"Unable to determine LL/UL for {descriptor.label()} – skipping.",
            )
            continue

        template_rows = sheet_utils.find_rows_by_test(
            template_ws,
            template_header_row,
            template_headers,
            test_name=descriptor.test_name,
            test_number=descriptor.test_number,
        )
        if not template_rows:
            _warn_if_missing(
                io,
                warnings,
                f"Template row not found for {descriptor.label()} – skipping.",
            )
            continue

        for row_idx in template_rows:
            sheet_utils.set_cell(template_ws, row_idx, ll_column, lower_limit)
            sheet_utils.set_cell(template_ws, row_idx, ul_column, upper_limit)

        if stdf_lower_col is not None and stdf_upper_col is not None:
            limit_rows = sheet_utils.find_rows_by_test(
                limits_ws,
                limits_header_row,
                limits_headers,
                test_name=descriptor.test_name,
                test_number=descriptor.test_number,
            )
            for row_idx in limit_rows or []:
                sheet_utils.set_cell(limits_ws, row_idx, stdf_lower_col, lower_limit)
                sheet_utils.set_cell(limits_ws, row_idx, stdf_upper_col, upper_limit)

        updated_tests.append(descriptor)

    if not updated_tests:
        raise ActionCancelled("No tests updated.")

    context.invalidate_frames("limits")
    charts.refresh_tests(context, updated_tests)

    summary_text = f"Updated STDF limits for {len(updated_tests)} test(s)."
    return {
        "summary": summary_text,
        "warnings": warnings,
        "audit": {
            "scope": resolved["scope"],
            "tests": _tests_to_strings(updated_tests),
            "parameters": {"target_cpk": resolved.get("target_cpk")},
        },
        "replay_params": resolved,
        "mark_dirty": True,
    }


def apply_spec_limits(context: PostProcessContext, io: PostProcessIO, params: Optional[dict]) -> dict:
    descriptors = _build_test_descriptors(context)
    if not descriptors:
        raise ActionCancelled("Summary sheet is empty.")

    resolved = _prompt_scope(io, params, allow_single=True, require_target=False)
    selected_tests = _resolve_tests(descriptors, context, io, resolved)

    template_ws = context.template_sheet()
    template_header_row, template_headers = sheet_utils.build_header_map(template_ws)
    spec_lower_col = _resolve_column(template_headers, ["Spec Lower", "Spec LL", "Spec Lower Limit"])
    spec_upper_col = _resolve_column(template_headers, ["Spec Upper", "Spec UL", "Spec Upper Limit"])
    what_lower_col = _resolve_column(template_headers, ["What-if Lower", "What If Lower", "What-If Lower"])
    what_upper_col = _resolve_column(template_headers, ["What-if Upper", "What If Upper", "What-If Upper"])

    limits_ws = context.workbook()["Test List and Limits"]
    limits_header_row, limits_headers = sheet_utils.build_header_map(limits_ws)
    limits_spec_lower_col = _resolve_column(limits_headers, ["Spec Lower Limit", "Spec Lower"])
    limits_spec_upper_col = _resolve_column(limits_headers, ["Spec Upper Limit", "Spec Upper"])
    limits_what_lower_col = _resolve_column(limits_headers, ["User What-If Lower Limit", "What-if Lower"])
    limits_what_upper_col = _resolve_column(limits_headers, ["User What-If Upper Limit", "What-if Upper"])

    if not any([spec_lower_col, spec_upper_col, what_lower_col, what_upper_col]):
        raise ActionCancelled("Template sheet missing Spec / What-If columns.")

    summary_df = context.summary_frame()
    summary_lookup = _summaries_by_key(summary_df)

    warnings: List[str] = []
    updated_tests: List[TestDescriptor] = []

    for descriptor in selected_tests:
        template_rows = sheet_utils.find_rows_by_test(
            template_ws,
            template_header_row,
            template_headers,
            test_name=descriptor.test_name,
            test_number=descriptor.test_number,
        )
        if not template_rows:
            _warn_if_missing(io, warnings, f"Template row not found for {descriptor.label()} – skipping.")
            continue

        row = summary_lookup.get(descriptor.key())
        mean = _safe_float(row.get("MEAN")) if row is not None else None
        stdev = _safe_float(row.get("STDEV")) if row is not None else None

        spec_lower = spec_upper = what_lower = what_upper = None
        if resolved.get("target_cpk"):
            target = resolved["target_cpk"]
            if target and target > 0 and stdev and stdev > 0 and mean is not None:
                width = 3.0 * target * stdev
                spec_lower = mean - width
                spec_upper = mean + width
                what_lower = spec_lower
                what_upper = spec_upper
            else:
                _warn_if_missing(io, warnings, f"Cannot compute target-based limits for {descriptor.label()}.")

        for row_idx in template_rows:
            if spec_lower_col:
                spec_lower = spec_lower if spec_lower is not None else _safe_float(
                    sheet_utils.get_cell(template_ws, row_idx, spec_lower_col)
                )
            if spec_upper_col:
                spec_upper = spec_upper if spec_upper is not None else _safe_float(
                    sheet_utils.get_cell(template_ws, row_idx, spec_upper_col)
                )
            if what_lower_col:
                what_lower = what_lower if what_lower is not None else _safe_float(
                    sheet_utils.get_cell(template_ws, row_idx, what_lower_col)
                )
            if what_upper_col:
                what_upper = what_upper if what_upper is not None else _safe_float(
                    sheet_utils.get_cell(template_ws, row_idx, what_upper_col)
                )

        def _write_if_available(column, value):
            if column and value is not None:
                for row_idx in template_rows:
                    sheet_utils.set_cell(template_ws, row_idx, column, value)

        _write_if_available(spec_lower_col, spec_lower)
        _write_if_available(spec_upper_col, spec_upper)
        _write_if_available(what_lower_col, what_lower)
        _write_if_available(what_upper_col, what_upper)

        limit_rows = sheet_utils.find_rows_by_test(
            limits_ws,
            limits_header_row,
            limits_headers,
            test_name=descriptor.test_name,
            test_number=descriptor.test_number,
        )
        for row_idx in limit_rows or []:
            if limits_spec_lower_col and spec_lower is not None:
                sheet_utils.set_cell(limits_ws, row_idx, limits_spec_lower_col, spec_lower)
            if limits_spec_upper_col and spec_upper is not None:
                sheet_utils.set_cell(limits_ws, row_idx, limits_spec_upper_col, spec_upper)
            if limits_what_lower_col and what_lower is not None:
                sheet_utils.set_cell(limits_ws, row_idx, limits_what_lower_col, what_lower)
            if limits_what_upper_col and what_upper is not None:
                sheet_utils.set_cell(limits_ws, row_idx, limits_what_upper_col, what_upper)

        updated_tests.append(descriptor)

    if not updated_tests:
        raise ActionCancelled("No tests updated.")

    context.invalidate_frames("limits")
    charts.refresh_tests(context, updated_tests, include_spec=True)

    summary_text = f"Applied Spec/What-If limits for {len(updated_tests)} test(s)."
    return {
        "summary": summary_text,
        "warnings": warnings,
        "audit": {
            "scope": resolved["scope"],
            "tests": _tests_to_strings(updated_tests),
            "parameters": {"target_cpk": resolved.get("target_cpk")},
        },
        "replay_params": resolved,
        "mark_dirty": True,
    }


def calculate_proposed_limits(context: PostProcessContext, io: PostProcessIO, params: Optional[dict]) -> dict:
    descriptors = _build_test_descriptors(context)
    if not descriptors:
        raise ActionCancelled("Summary sheet is empty.")

    resolved = _prompt_scope(io, params, allow_single=True, require_target=True)
    selected_tests = _resolve_tests(descriptors, context, io, resolved)
    target_cpk = resolved.get("target_cpk") or 0.0

    template_ws = context.template_sheet()
    template_header_row, template_headers = sheet_utils.build_header_map(template_ws)
    ll_prop_col = _resolve_column(template_headers, ["LL_PROP", "LL Proposal", "LL Proposed"])
    ul_prop_col = _resolve_column(template_headers, ["UL_PROP", "UL Proposal", "UL Proposed"])
    cpk_prop_col = _resolve_column(template_headers, ["CPK_PROP", "CPK Proposed"])
    yld_prop_col = _resolve_column(template_headers, ["%YLD LOSS_PROP", "%YLD Loss Proposed"])

    if not all([ll_prop_col, ul_prop_col, cpk_prop_col, yld_prop_col]):
        raise ActionCancelled("Template sheet missing proposed-limit columns.")

    summary_df = context.summary_frame()
    summary_lookup = _summaries_by_key(summary_df)
    measurements_df = context.measurements_frame()

    warnings: List[str] = []
    updated_tests: List[TestDescriptor] = []

    for descriptor in selected_tests:
        row = summary_lookup.get(descriptor.key())
        if row is None:
            _warn_if_missing(io, warnings, f"Summary entry not found for {descriptor.label()} – skipping.")
            continue

        mean = _safe_float(row.get("MEAN"))
        stdev = _safe_float(row.get("STDEV"))
        if mean is None or stdev is None or stdev <= 0:
            _warn_if_missing(io, warnings, f"Cannot compute proposals for {descriptor.label()} – invalid stats.")
            continue

        width = 3.0 * target_cpk * stdev
        proposed_ll = mean - width
        proposed_ul = mean + width

        # Determine yield loss using measurement data
        failures = _compute_yield_loss(
            measurements_df,
            descriptor.test_name,
            descriptor.test_number,
            proposed_ll,
            proposed_ul,
        )

        template_rows = sheet_utils.find_rows_by_test(
            template_ws,
            template_header_row,
            template_headers,
            test_name=descriptor.test_name,
            test_number=descriptor.test_number,
        )
        if not template_rows:
            _warn_if_missing(io, warnings, f"Template row not found for {descriptor.label()} – skipping.")
            continue

        for row_idx in template_rows:
            sheet_utils.set_cell(template_ws, row_idx, ll_prop_col, proposed_ll)
            sheet_utils.set_cell(template_ws, row_idx, ul_prop_col, proposed_ul)
            sheet_utils.set_cell(template_ws, row_idx, cpk_prop_col, target_cpk)
            sheet_utils.set_cell(template_ws, row_idx, yld_prop_col, failures)

        updated_tests.append(descriptor)

    if not updated_tests:
        raise ActionCancelled("No tests updated.")

    charts.refresh_tests(context, updated_tests, include_spec=True, include_proposed=True)

    summary_text = f"Calculated proposed limits for {len(updated_tests)} test(s) with CPK target {target_cpk:.3f}."
    return {
        "summary": summary_text,
        "warnings": warnings,
        "audit": {
            "scope": resolved["scope"],
            "tests": _tests_to_strings(updated_tests),
            "parameters": {"target_cpk": target_cpk},
        },
        "replay_params": resolved,
        "mark_dirty": True,
    }


# ---------------------------------------------------------------------------
# Supporting calculations
# ---------------------------------------------------------------------------


def _compute_yield_loss(
    measurements: pd.DataFrame,
    test_name: str,
    test_number: str,
    lower: float,
    upper: float,
) -> float:
    if measurements.empty:
        return float("nan")
    filtered = measurements.copy()
    if "Test Name" in filtered.columns:
        filtered = filtered[filtered["Test Name"].astype(str) == str(test_name)]
    elif "TEST NAME" in filtered.columns:
        filtered = filtered[filtered["TEST NAME"].astype(str) == str(test_name)]
    if "Test Number" in filtered.columns:
        filtered = filtered[filtered["Test Number"].astype(str) == str(test_number)]
    elif "TEST NUM" in filtered.columns:
        filtered = filtered[filtered["TEST NUM"].astype(str) == str(test_number)]

    if filtered.empty or "Value" not in filtered.columns:
        return float("nan")

    values = pd.to_numeric(filtered["Value"], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    failures = 0
    if lower is not None:
        failures += int(np.sum(values < lower))
    if upper is not None:
        failures += int(np.sum(values > upper))
    return (failures / len(values)) * 100.0
