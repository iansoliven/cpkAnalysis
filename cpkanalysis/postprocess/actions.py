"""Action implementations for post-processing menu."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .context import PostProcessContext
from .io_adapters import PostProcessIO
from . import sheet_utils
from . import charts
from . import proposed_limits_grr

__all__ = [
    "ActionCancelled",
    "update_stdf_limits",
    "apply_spec_limits",
    "calculate_proposed_limits",
    "calculate_proposed_limits_grr",
]

PROPOSAL_TOLERANCE = 1e-6


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
                file=_safe_str(row.get("File")),
                test_name=_safe_str(row.get("Test Name")),
                test_number=_safe_str(row.get("Test Number")),
                unit=_safe_str(row.get("Unit")),
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
    prompt_target: bool = True,
) -> dict:
    scope = (params or {}).get("scope")
    if scope not in {"all", "single"}:
        options = ["All tests"]
        if allow_single:
            options.append("Single test")
        choice = io.prompt_choice("Select scope for this action:", options)
        scope = "all" if choice == 0 else "single"

    target_cpk = None
    if prompt_target:
        target_cpk = (params or {}).get("target_cpk")
        if require_target and target_cpk is not None:
            try:
                target_cpk = float(target_cpk)
            except (TypeError, ValueError):
                io.warn("Target CPK must be numeric.")
                target_cpk = None
            else:
                if target_cpk <= 0:
                    io.warn("Target CPK must be positive.")
                    target_cpk = None

    if prompt_target and (require_target or (params is None)):
        prompt = (
            "Enter target CPK (blank to keep existing limits):"
            if not require_target
            else "Enter target CPK (must be greater than zero):"
        )
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

    io.info("Select the test you want to update:")
    labels = [desc.label() for desc in descriptors]
    choice = io.prompt_choice("Test selection:", labels)
    descriptor = descriptors[choice]
    resolved["test_key"] = descriptor.key()
    return [descriptor]


def _summaries_by_key(summary: pd.DataFrame) -> Dict[str, pd.Series]:
    mapping: Dict[str, pd.Series] = {}
    for _, row in summary.iterrows():
        key = "|".join(
            [
                _safe_str(row.get("File")),
                _safe_str(row.get("Test Name")),
                _safe_str(row.get("Test Number")),
            ]
        )
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


def _safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text.strip() if text is not None else ""


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _read_first_from_rows(sheet, rows: List[int], column: int | None) -> float | None:
    if column is None:
        return None
    for row_idx in rows:
        value = _safe_float(sheet_utils.get_cell(sheet, row_idx, column))
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Proposed limit helpers
# ---------------------------------------------------------------------------


def _almost_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= PROPOSAL_TOLERANCE


def _ensure_proposal_state(metadata: Dict[str, object]) -> Dict[str, Dict[str, float | None]]:
    state = metadata.setdefault("post_processing_state", {})
    if not isinstance(state, dict):
        state = {}
        metadata["post_processing_state"] = state
    proposals = state.setdefault("proposed_limits", {})
    if not isinstance(proposals, dict):
        proposals = {}
        state["proposed_limits"] = proposals
    return proposals


def _ensure_proposed_spec_columns(
    template_ws,
    header_row: int,
    header_map: Dict[str, int],
) -> dict[str, int]:
    yld_col = _resolve_column(header_map, ["%YLD LOSS_PROP", "%YLD Loss Proposed"])
    if yld_col is None:
        raise ActionCancelled("Template sheet missing %YLD LOSS_PROP column required for proposed spec output.")

    start_col = yld_col + 5  # keep four blank columns for readability
    headers = [
        ("Proposed Spec Lower", start_col),
        ("Proposed Spec Upper", start_col + 1),
        ("CPK Proposed Spec", start_col + 2),
        ("Guardband Selection", start_col + 3),
    ]

    for title, column in headers:
        cell = template_ws.cell(row=header_row, column=column)
        if _safe_str(cell.value) == "":
            cell.value = title
        header_map[sheet_utils.normalize_header(title)] = column

    return {
        "spec_lower": start_col,
        "spec_upper": start_col + 1,
        "spec_cpk": start_col + 2,
        "guardband": start_col + 3,
    }


def _resolve_unit_override(
    io: PostProcessIO,
    cache: Dict[Tuple[str, str], bool],
    descriptor_label: str,
    workbook_unit: str,
    grr_unit: str,
) -> bool:
    key = (workbook_unit or "", grr_unit or "")
    if key in cache:
        return cache[key]
    message = (
        f"Unit mismatch for {descriptor_label}: workbook '{workbook_unit or 'n/a'}' "
        f"vs GRR '{grr_unit or 'n/a'}'. Proceed using workbook unit?"
    )
    decision = io.confirm(message, default=False)
    cache[key] = decision
    return decision


def _coerce_positive_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0 or math.isnan(number) or math.isinf(number):
        return None
    return number


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def update_stdf_limits(context: PostProcessContext, io: PostProcessIO, params: Optional[dict]) -> dict:
    descriptors = _build_test_descriptors(context)
    if not descriptors:
        raise ActionCancelled("Summary sheet is empty.")

    resolved = _prompt_scope(io, params, allow_single=True, require_target=False, prompt_target=False)
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

        lower_limit = _first_not_none(_safe_float(row.get("LL_2CPK")), _safe_float(row.get("LL_3IQR")))
        upper_limit = _first_not_none(_safe_float(row.get("UL_2CPK")), _safe_float(row.get("UL_3IQR")))

        if lower_limit is None and upper_limit is None:
            _warn_if_missing(
                io,
                warnings,
                f"Unable to determine limits for {descriptor.label()} – skipping.",
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
            if lower_limit is not None:
                sheet_utils.set_cell(template_ws, row_idx, ll_column, lower_limit)
            if upper_limit is not None:
                sheet_utils.set_cell(template_ws, row_idx, ul_column, upper_limit)

        if stdf_lower_col is not None or stdf_upper_col is not None:
            limit_rows = sheet_utils.find_rows_by_test(
                limits_ws,
                limits_header_row,
                limits_headers,
                test_name=descriptor.test_name,
                test_number=descriptor.test_number,
            )
            for row_idx in limit_rows or []:
                if stdf_lower_col is not None and lower_limit is not None:
                    sheet_utils.set_cell(limits_ws, row_idx, stdf_lower_col, lower_limit)
                if stdf_upper_col is not None and upper_limit is not None:
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
        missing = []
        if not ll_prop_col:
            missing.append("LL_PROP")
        if not ul_prop_col:
            missing.append("UL_PROP")
        if not cpk_prop_col:
            missing.append("CPK_PROP")
        if not yld_prop_col:
            missing.append("%YLD LOSS_PROP")
        raise ActionCancelled(
            f"Template sheet missing proposed-limit columns: {', '.join(missing)}. "
            "Please add these columns to your template sheet to use this feature."
        )

    summary_df = context.summary_frame()
    summary_lookup = _summaries_by_key(summary_df)
    measurements_df = context.measurements_frame()
    metadata_proposals = _ensure_proposal_state(context.metadata)
    timestamp = datetime.now(timezone.utc).isoformat()

    warnings: List[str] = []
    updated_tests: List[TestDescriptor] = []
    metrics_updates = 0
    audit_details: List[dict] = []
    mark_dirty = False

    def _read_first(rows: List[int], column: int | None) -> float | None:
        if column is None:
            return None
        for row_idx in rows:
            value = _safe_float(sheet_utils.get_cell(template_ws, row_idx, column))
            if value is not None:
                return value
        return None

    for descriptor in selected_tests:
        row = summary_lookup.get(descriptor.key())
        if row is None:
            _warn_if_missing(io, warnings, f"Summary entry not found for {descriptor.label()} – skipping.")
            continue

        mean = _safe_float(row.get("MEAN"))
        stdev = _safe_float(row.get("STDEV"))
        width = None
        if mean is not None and stdev is not None and stdev > 0:
            width = 3.0 * target_cpk * stdev

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

        prior_entry = metadata_proposals.get(descriptor.key(), {})
        prev_lower = _safe_float(prior_entry.get("ll")) if isinstance(prior_entry, dict) else None
        prev_upper = _safe_float(prior_entry.get("ul")) if isinstance(prior_entry, dict) else None

        existing_lower = _read_first(template_rows, ll_prop_col)
        existing_upper = _read_first(template_rows, ul_prop_col)
        current_cpk = _read_first(template_rows, cpk_prop_col)
        current_yield = _read_first(template_rows, yld_prop_col)

        lower_origin = "unchanged"
        upper_origin = "unchanged"
        proposal_changed = False

        final_lower = existing_lower
        if existing_lower is not None:
            if prev_lower is None or not _almost_equal(existing_lower, prev_lower):
                lower_origin = "user"
                proposal_changed = True
        else:
            if width is None or mean is None:
                _warn_if_missing(io, warnings, f"Cannot compute lower proposal for {descriptor.label()} – missing stats.")
            else:
                final_lower = mean - width
                lower_origin = "computed"
                proposal_changed = True

        final_upper = existing_upper
        if existing_upper is not None:
            if prev_upper is None or not _almost_equal(existing_upper, prev_upper):
                upper_origin = "user"
                proposal_changed = True
        else:
            if width is None or mean is None:
                _warn_if_missing(io, warnings, f"Cannot compute upper proposal for {descriptor.label()} – missing stats.")
            else:
                final_upper = mean + width
                upper_origin = "computed"
                proposal_changed = True

        if final_lower is None and final_upper is None:
            _warn_if_missing(io, warnings, f"No proposed limits determined for {descriptor.label()} – skipping.")
            continue

        metrics_blank = current_cpk is None or current_yield is None
        if not proposal_changed and not metrics_blank:
            continue

        for row_idx in template_rows:
            if final_lower is not None and lower_origin in {"user", "computed"}:
                sheet_utils.set_cell(template_ws, row_idx, ll_prop_col, final_lower)
                mark_dirty = True
            if final_upper is not None and upper_origin in {"user", "computed"}:
                sheet_utils.set_cell(template_ws, row_idx, ul_prop_col, final_upper)
                mark_dirty = True

        cpk_updated = False
        yield_updated = False

        if proposal_changed or current_cpk is None:
            if stdev is None or stdev <= 0 or mean is None:
                _warn_if_missing(io, warnings, f"Cannot compute CPK for {descriptor.label()} – missing stats.")
            else:
                candidates: List[float] = []
                if final_upper is not None:
                    candidates.append((final_upper - mean) / (3.0 * stdev))
                if final_lower is not None:
                    candidates.append((mean - final_lower) / (3.0 * stdev))
                new_cpk_value = min(candidates) if candidates else None
                if new_cpk_value is not None and not np.isfinite(new_cpk_value):
                    new_cpk_value = None
                if new_cpk_value is not None:
                    if current_cpk is None or not _almost_equal(new_cpk_value, current_cpk):
                        for row_idx in template_rows:
                            sheet_utils.set_cell(template_ws, row_idx, cpk_prop_col, new_cpk_value)
                        cpk_updated = True
                        mark_dirty = True
                elif current_cpk is None:
                    pass  # Nothing to write

        if proposal_changed or current_yield is None:
            if measurements_df.empty:
                _warn_if_missing(io, warnings, f"No measurements available to compute yield for {descriptor.label()}.")
            else:
                new_yield_value = _compute_yield_loss(
                    measurements_df,
                    descriptor.test_name,
                    descriptor.test_number,
                    final_lower,
                    final_upper,
                )
                if new_yield_value is not None and not np.isfinite(new_yield_value):
                    new_yield_value = None
                if new_yield_value is not None:
                    if current_yield is None or not _almost_equal(new_yield_value, current_yield):
                        for row_idx in template_rows:
                            sheet_utils.set_cell(template_ws, row_idx, yld_prop_col, new_yield_value)
                        yield_updated = True
                        mark_dirty = True
                elif current_yield is None:
                    pass

        test_changed = proposal_changed or cpk_updated or yield_updated
        if not test_changed:
            continue

        metadata_proposals[descriptor.key()] = {"ll": final_lower, "ul": final_upper, "timestamp": timestamp}
        mark_dirty = True

        updated_tests.append(descriptor)
        if cpk_updated or yield_updated:
            metrics_updates += 1
        audit_details.append(
            {
                "test": descriptor.key(),
                "lower_origin": lower_origin,
                "upper_origin": upper_origin,
                "cpk_updated": cpk_updated,
                "yield_updated": yield_updated,
            }
        )

    if not updated_tests:
        raise ActionCancelled("No tests updated.")

    charts.refresh_tests(context, updated_tests, include_spec=True, include_proposed=True)

    summary_parts = [f"Updated proposed limits for {len(updated_tests)} test(s)."]
    if metrics_updates:
        summary_parts.append(f"Recomputed metrics for {metrics_updates} test(s).")
    summary_text = " ".join(summary_parts)

    return {
        "summary": summary_text,
        "warnings": warnings,
        "audit": {
            "scope": resolved["scope"],
            "tests": _tests_to_strings(updated_tests),
            "parameters": {
                "target_cpk": target_cpk,
                "per_test": audit_details,
            },
        },
        "replay_params": resolved,
        "mark_dirty": mark_dirty,
    }


def calculate_proposed_limits_grr(context: PostProcessContext, io: PostProcessIO, params: Optional[dict]) -> dict:
    descriptors = _build_test_descriptors(context)
    if not descriptors:
        raise ActionCancelled("Summary sheet is empty.")

    params = dict(params or {})
    grr_available = params.get("grr_available")
    if grr_available not in {True, False}:
        grr_available = io.confirm("Is GRR data available for proposed limit calculation?", default=True)
    if not grr_available:
        raise ActionCancelled("GRR data is required to compute guardbanded proposed limits.")
    params["grr_available"] = True

    def _prompt_grr_path(previous: str | None) -> Path:
        attempt = _safe_str(previous)
        while True:
            if not attempt:
                attempt = _safe_str(io.prompt("Enter directory containing Total_GRR.xlsx:", default=""))
            if not attempt:
                raise ActionCancelled("GRR directory not provided.")
            candidate = Path(attempt).expanduser()
            workbook_path = candidate if candidate.is_file() else candidate / "Total_GRR.xlsx"
            if workbook_path.exists():
                return candidate
            io.warn(f"GRR workbook not found at {workbook_path}.")
            if not io.confirm("Try another location?", default=True):
                raise ActionCancelled("Unable to locate GRR workbook.")
            attempt = ""

    grr_root = _prompt_grr_path(params.get("grr_path"))
    params["grr_path"] = str(grr_root)

    try:
        grr_table = proposed_limits_grr.load_grr_table(grr_root)
    except (FileNotFoundError, ValueError) as exc:
        raise ActionCancelled(str(exc))

    min_cpk = _coerce_positive_float(params.get("cpk_min"))
    while min_cpk is None:
        value = io.prompt_float("Enter minimum FT CPK:", default=None)
        if value is None:
            io.warn("Minimum FT CPK is required.")
            continue
        if value <= 0:
            io.warn("Minimum FT CPK must be positive.")
            continue
        min_cpk = value
    params["cpk_min"] = min_cpk

    max_cpk = _coerce_positive_float(params.get("cpk_max"))
    while max_cpk is None:
        value = io.prompt_float("Enter maximum FT CPK:", default=None)
        if value is None:
            io.warn("Maximum FT CPK is required.")
            continue
        if value <= 0:
            io.warn("Maximum FT CPK must be positive.")
            continue
        max_cpk = value
    while max_cpk < min_cpk:
        io.warn("Maximum FT CPK must be greater than or equal to the minimum.")
        value = io.prompt_float("Enter maximum FT CPK:", default=max_cpk)
        if value is None or value <= 0:
            io.warn("Maximum FT CPK must be positive.")
            continue
        max_cpk = value
    params["cpk_max"] = max_cpk

    resolved = _prompt_scope(io, params, allow_single=True, prompt_target=False)
    selected_tests = _resolve_tests(descriptors, context, io, resolved)

    template_ws = context.template_sheet()
    template_header_row, template_headers = sheet_utils.build_header_map(template_ws)
    ll_prop_col = _resolve_column(template_headers, ["LL_PROP", "LL Proposal", "LL Proposed"])
    ul_prop_col = _resolve_column(template_headers, ["UL_PROP", "UL Proposal", "UL Proposed"])
    cpk_prop_col = _resolve_column(template_headers, ["CPK_PROP", "CPK Proposed"])
    yld_prop_col = _resolve_column(template_headers, ["%YLD LOSS_PROP", "%YLD Loss Proposed"])
    if not all([ll_prop_col, ul_prop_col, cpk_prop_col, yld_prop_col]):
        missing = []
        if not ll_prop_col:
            missing.append("LL_PROP")
        if not ul_prop_col:
            missing.append("UL_PROP")
        if not cpk_prop_col:
            missing.append("CPK_PROP")
        if not yld_prop_col:
            missing.append("%YLD LOSS_PROP")
        raise ActionCancelled(
            "Template sheet missing proposed-limit columns: " + ", ".join(missing) + "."
        )

    spec_columns = _ensure_proposed_spec_columns(template_ws, template_header_row, template_headers)
    spec_prop_lower_col = spec_columns["spec_lower"]
    spec_prop_upper_col = spec_columns["spec_upper"]
    spec_prop_cpk_col = spec_columns["spec_cpk"]
    guardband_col = spec_columns["guardband"]

    spec_lower_col = _resolve_column(template_headers, ["Spec Lower", "Spec Lower Limit"])
    spec_upper_col = _resolve_column(template_headers, ["Spec Upper", "Spec Upper Limit"])

    summary_df = context.summary_frame()
    summary_lookup = _summaries_by_key(summary_df)
    measurements_df = context.measurements_frame()
    metadata_proposals = _ensure_proposal_state(context.metadata)
    timestamp = datetime.now(timezone.utc).isoformat()

    warnings: List[str] = []
    updated_tests: List[TestDescriptor] = []
    metrics_updates = 0
    audit_details: List[dict] = []
    unit_override_cache: Dict[Tuple[str, str], bool] = {}
    mark_dirty = False

    for descriptor in selected_tests:
        row = summary_lookup.get(descriptor.key())
        if row is None:
            _warn_if_missing(io, warnings, f"Summary entry not found for {descriptor.label()} - skipping.")
            continue

        mean = _safe_float(row.get("MEAN"))
        stdev = _safe_float(row.get("STDEV"))
        if mean is None or stdev is None or stdev <= 0:
            _warn_if_missing(io, warnings, f"Cannot compute limits for {descriptor.label()} - missing stats.")
            continue

        template_rows = sheet_utils.find_rows_by_test(
            template_ws,
            template_header_row,
            template_headers,
            test_name=descriptor.test_name,
            test_number=descriptor.test_number,
        )
        if not template_rows:
            _warn_if_missing(io, warnings, f"Template row not found for {descriptor.label()} - skipping.")
            continue

        grr_record = grr_table.find(descriptor.test_number, descriptor.test_name)
        if grr_record is None or grr_record.guardband_full is None or grr_record.guardband_full <= 0:
            _warn_if_missing(io, warnings, f"GRR data missing for {descriptor.label()} - skipping.")
            continue

        workbook_unit = proposed_limits_grr.normalize_unit(descriptor.unit)
        grr_unit = grr_record.unit_normalized
        if workbook_unit and grr_unit and workbook_unit != grr_unit:
            proceed = _resolve_unit_override(
                io,
                unit_override_cache,
                descriptor.label(),
                descriptor.unit,
                grr_record.unit_raw,
            )
            if not proceed:
                warnings.append(
                    f"Skipped {descriptor.label()} due to unmatched units: workbook '{descriptor.unit or 'n/a'}' "
                    f"vs GRR '{grr_record.unit_raw or 'n/a'}'."
                )
                continue

        spec_lower_value = grr_record.spec_lower
        spec_upper_value = grr_record.spec_upper
        if spec_lower_value is None or spec_upper_value is None:
            spec_lower_value = _read_first_from_rows(template_ws, template_rows, spec_lower_col)
            spec_upper_value = _read_first_from_rows(template_ws, template_rows, spec_upper_col)
        if spec_lower_value is None or spec_upper_value is None:
            _warn_if_missing(io, warnings, f"Spec limits missing for {descriptor.label()} - skipping.")
            continue

        try:
            computation = proposed_limits_grr.compute_proposed_limits(
                mean=float(mean),
                stdev=float(stdev),
                spec_lower=float(spec_lower_value),
                spec_upper=float(spec_upper_value),
                guardband_full=float(grr_record.guardband_full),
                cpk_min=float(min_cpk),
                cpk_max=float(max_cpk),
            )
        except proposed_limits_grr.ComputationError as exc:
            _warn_if_missing(io, warnings, f"{descriptor.label()}: {exc}")
            continue

        for row_idx in template_rows:
            sheet_utils.set_cell(template_ws, row_idx, ll_prop_col, computation.ft_lower)
            sheet_utils.set_cell(template_ws, row_idx, ul_prop_col, computation.ft_upper)
            sheet_utils.set_cell(template_ws, row_idx, spec_prop_lower_col, computation.spec_lower)
            sheet_utils.set_cell(template_ws, row_idx, spec_prop_upper_col, computation.spec_upper)
            sheet_utils.set_cell(template_ws, row_idx, guardband_col, computation.guardband_label)
            if computation.ft_cpk is not None:
                sheet_utils.set_cell(template_ws, row_idx, cpk_prop_col, computation.ft_cpk)
            if computation.spec_cpk is not None:
                sheet_utils.set_cell(template_ws, row_idx, spec_prop_cpk_col, computation.spec_cpk)
        mark_dirty = True

        yield_value = None
        if measurements_df.empty:
            _warn_if_missing(io, warnings, f"No measurements available to compute yield for {descriptor.label()}.")
        else:
            yield_value = _compute_yield_loss(
                measurements_df,
                descriptor.test_name,
                descriptor.test_number,
                computation.ft_lower,
                computation.ft_upper,
            )
            if yield_value is not None and not np.isfinite(yield_value):
                yield_value = None
        if yield_value is not None:
            for row_idx in template_rows:
                sheet_utils.set_cell(template_ws, row_idx, yld_prop_col, yield_value)
            metrics_updates += 1

        if computation.ft_cpk is not None:
            metrics_updates += 1

        metadata_proposals[descriptor.key()] = {
            "ll": computation.ft_lower,
            "ul": computation.ft_upper,
            "spec_ll": computation.spec_lower,
            "spec_ul": computation.spec_upper,
            "guardband": computation.guardband_label,
            "timestamp": timestamp,
        }

        audit_entry = {
            "test": descriptor.key(),
            "guardband": computation.guardband_label,
            "guardband_value": computation.guardband_value,
            "ft_lower": computation.ft_lower,
            "ft_upper": computation.ft_upper,
            "ft_cpk": computation.ft_cpk,
            "spec_lower": computation.spec_lower,
            "spec_upper": computation.spec_upper,
            "spec_cpk": computation.spec_cpk,
            "spec_widened": computation.spec_widened,
            "notes": computation.notes,
        }
        if grr_record.guardband_full is not None:
            audit_entry["grr_full"] = grr_record.guardband_full
        audit_details.append(audit_entry)
        updated_tests.append(descriptor)

    if not updated_tests:
        raise ActionCancelled("No tests updated.")

    charts.refresh_tests(
        context,
        updated_tests,
        include_spec=True,
        include_proposed=True,
        include_proposed_spec=True,
        build_proposed_sheets=True,
    )

    summary_parts = [f"Calculated GRR-based proposed limits for {len(updated_tests)} test(s)."]
    if metrics_updates:
        summary_parts.append(f"Updated metrics for {metrics_updates} test(s).")
    summary_text = " ".join(summary_parts)

    return {
        "summary": summary_text,
        "warnings": warnings,
        "audit": {
            "scope": resolved["scope"],
            "tests": _tests_to_strings(updated_tests),
            "parameters": {
                "grr_path": params.get("grr_path"),
                "cpk_min": min_cpk,
                "cpk_max": max_cpk,
                "per_test": audit_details,
            },
        },
        "replay_params": {
            "scope": resolved["scope"],
            "test_key": resolved.get("test_key"),
            "grr_available": True,
            "grr_path": params.get("grr_path"),
            "cpk_min": min_cpk,
            "cpk_max": max_cpk,
        },
        "mark_dirty": mark_dirty,
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
    target_name = _safe_str(test_name)
    target_number = _safe_str(test_number)
    if target_name:
        for column in ("Test Name", "TEST NAME"):
            if column in filtered.columns:
                mask = filtered[column].map(_safe_str)
                filtered = filtered[mask == target_name]
                break
    if target_number:
        for column in ("Test Number", "TEST NUM"):
            if column in filtered.columns:
                mask = filtered[column].map(_safe_str)
                filtered = filtered[mask == target_number]
                break

    if filtered.empty or "Value" not in filtered.columns:
        return float("nan")

    values = pd.to_numeric(filtered["Value"], errors="coerce").dropna()
    if values.empty:
        return float("nan")

    # Filter out Inf/-Inf values (dropna() only removes NaN).
    # Rationale: ±Inf typically indicates instrumentation artefacts (saturation,
    # divide-by-zero, etc.) rather than true out-of-spec parts, so counting them
    # as failures would distort the yield metric.
    finite_values = values[np.isfinite(values)]
    if finite_values.empty:
        return float("nan")

    failures = 0
    if lower is not None:
        failures += int(np.sum(finite_values < lower))
    if upper is not None:
        failures += int(np.sum(finite_values > upper))
    return failures / len(finite_values)
