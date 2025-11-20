from __future__ import annotations

import math
import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import LimitSource

GROUP_KEYS = ["file", "test_name", "test_number"]
GROUP_KEYS_SITE = ["file", "site", "test_name", "test_number"]

SUMMARY_COLUMNS = [
    "File",
    "Test Name",
    "Test Number",
    "Unit",
    "COUNT",
    "MEAN",
    "MEDIAN",
    "STDEV",
    "IQR",
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
]

SUMMARY_COLUMNS_SITE = [
    "File",
    "Site",
    "Test Name",
    "Test Number",
    "Unit",
    "COUNT",
    "MEAN",
    "MEDIAN",
    "STDEV",
    "IQR",
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
]

YIELD_SUMMARY_COLUMNS = [
    "file",
    "devices_total",
    "devices_pass",
    "devices_fail",
    "yield_percent",
]

YIELD_SUMMARY_COLUMNS_SITE = [
    "file",
    "site",
    "devices_total",
    "devices_pass",
    "devices_fail",
    "yield_percent",
]

PARETO_COLUMNS = [
    "file",
    "test_name",
    "test_number",
    "devices_fail",
    "fail_rate_percent",
    "cumulative_percent",
    "lower_limit",
    "upper_limit",
]

PARETO_COLUMNS_SITE = [
    "file",
    "site",
    "test_name",
    "test_number",
    "devices_fail",
    "fail_rate_percent",
    "cumulative_percent",
    "lower_limit",
    "upper_limit",
]


def compute_summary(
    measurements: pd.DataFrame,
    limits: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, LimitSource]]]:
    summary, limit_sources = _compute_summary_table(
        measurements,
        limits,
        group_keys=GROUP_KEYS,
        include_site=False,
    )
    return summary, limit_sources


def compute_yield_pareto(
    measurements: pd.DataFrame,
    limits: pd.DataFrame,
    *,
    first_failure_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    yield_frame = _compute_yield_summary(measurements, ["file"], limits)
    pareto_frame = _compute_pareto_details(
        measurements,
        limits,
        yield_frame,
        ["file"],
        first_failure_only,
    )
    return yield_frame, pareto_frame


def compute_summary_by_site(
    measurements: pd.DataFrame,
    limits: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[Any, Any, Any, Any], dict[str, LimitSource]]]:
    if "site" not in measurements.columns:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS_SITE)
        return empty, {}
    return _compute_summary_table(
        measurements,
        limits,
        group_keys=GROUP_KEYS_SITE,
        include_site=True,
    )


def compute_yield_pareto_by_site(
    measurements: pd.DataFrame,
    limits: pd.DataFrame,
    *,
    first_failure_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "site" not in measurements.columns:
        empty_yield = pd.DataFrame(columns=YIELD_SUMMARY_COLUMNS_SITE)
        empty_pareto = pd.DataFrame(columns=PARETO_COLUMNS_SITE)
        return empty_yield, empty_pareto
    yield_frame = _compute_yield_summary(measurements, ["file", "site"], limits)
    pareto_frame = _compute_pareto_details(
        measurements,
        limits,
        yield_frame,
        ["file", "site"],
        first_failure_only,
    )
    return yield_frame, pareto_frame


def _compute_summary_table(
    measurements: pd.DataFrame,
    limits: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    include_site: bool,
) -> tuple[pd.DataFrame, dict[tuple[Any, ...], dict[str, LimitSource]]]:
    columns = SUMMARY_COLUMNS_SITE if include_site else SUMMARY_COLUMNS
    required_columns = set(group_keys) | {"value"}
    if measurements.empty or not required_columns.issubset(measurements.columns):
        return pd.DataFrame(columns=columns), {}

    limit_map = _build_limit_map(limits)
    records: list[dict[str, Any]] = []
    limit_sources: dict[tuple[Any, ...], dict[str, LimitSource]] = {}

    grouped = measurements.groupby(list(group_keys), dropna=False, sort=False)
    for key_tuple, group in grouped:
        if not isinstance(key_tuple, tuple):
            key_tuple = (key_tuple,)
        key_map = {name: value for name, value in zip(group_keys, key_tuple)}
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        unit = _first_not_empty(group["units"]) if "units" in group.columns else None
        test_name = key_map.get("test_name")
        test_number = key_map.get("test_number")
        limit_info = _lookup_limit(limit_map, key_map.get("file"), test_name, test_number)
        stats_row = _compute_group_statistics(values.to_numpy(), limit_info)
        stats_row["File"] = key_map.get("file")
        if include_site:
            stats_row["Site"] = key_map.get("site")
        stats_row["Test Name"] = test_name
        stats_row["Test Number"] = test_number
        stats_row["Unit"] = unit or limit_info.get("unit", "")
        records.append(stats_row)
        limit_key = tuple(key_map.get(name) for name in group_keys)
        limit_sources[limit_key] = {
            "lower": stats_row.get("_LOWER_SRC", "unset"),
            "upper": stats_row.get("_UPPER_SRC", "unset"),
        }

    summary = pd.DataFrame(records, columns=columns + ["_LOWER_SRC", "_UPPER_SRC"])
    if "_LOWER_SRC" in summary.columns:
        summary = summary[columns]
    return summary, limit_sources


def _build_limit_map(limits: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[Optional[str], str, str], dict[str, Any]] = {}
    if limits is None or limits.empty:
        return result
    rows = limits.fillna(value=pd.NA)
    for _, row in rows.iterrows():
        key = (_coerce_str(row.get("file")), str(row.get("test_name", "")), str(row.get("test_number", "")))
        result[key] = {
            "unit": row.get("unit"),
            "stdf_lower": _coerce_float(row.get("stdf_lower")),
            "stdf_upper": _coerce_float(row.get("stdf_upper")),
            "spec_lower": _coerce_float(row.get("spec_lower")),
            "spec_upper": _coerce_float(row.get("spec_upper")),
            "what_if_lower": _coerce_float(row.get("what_if_lower")),
            "what_if_upper": _coerce_float(row.get("what_if_upper")),
        }
    return result


def _lookup_limit(
    limit_map: dict[tuple[Optional[str], str, str], dict[str, Any]],
    file_value: Any,
    test_name: Any,
    test_number: Any,
) -> dict[str, Any]:
    name = str(test_name)
    number = str(test_number)
    file_key = _coerce_str(file_value)
    info = limit_map.get((file_key, name, number))
    if info is None and file_key is not None:
        info = limit_map.get((None, name, number))
    if info is None:
        info = limit_map.get(("", name, number))
    return info or {}


def _compute_group_statistics(values: np.ndarray, limit_info: dict[str, Any]) -> dict[str, Any]:
    clean = values[np.isfinite(values)]
    n = clean.size
    if n == 0:
        return {name: math.nan for name in SUMMARY_COLUMNS[4:]}

    mean = float(clean.mean())
    median = float(np.median(clean))
    stdev = float(np.std(clean, ddof=1)) if n > 1 else 0.0
    if math.isnan(stdev):
        stdev = 0.0
    q1 = float(np.percentile(clean, 25))
    q3 = float(np.percentile(clean, 75))
    iqr = q3 - q1

    lsl, lsl_src = _select_limit("lower", limit_info)
    usl, usl_src = _select_limit("upper", limit_info)

    cpl = _process_capability_lower(mean, stdev, lsl)
    cpu = _process_capability_upper(mean, stdev, usl)
    cpk = _merge_capabilities(cpl, cpu)
    yield_loss = _yield_loss(clean, lsl, usl)

    ll_2cpk = mean - 6 * stdev if lsl is not None else math.nan
    ul_2cpk = mean + 6 * stdev if usl is not None else math.nan
    cpk_2 = _compute_cpk(mean, stdev, ll_2cpk if not math.isnan(ll_2cpk) else None, ul_2cpk if not math.isnan(ul_2cpk) else None)
    yield_loss_2 = _yield_loss(clean, None if math.isnan(ll_2cpk) else ll_2cpk, None if math.isnan(ul_2cpk) else ul_2cpk)

    ll_3iqr = median - 3 * iqr if iqr > 0 else median
    ul_3iqr = median + 3 * iqr if iqr > 0 else median
    robust_sigma = (iqr / 1.349) if iqr > 0 else stdev
    cpk_3iqr = _compute_cpk(mean, stdev, ll_3iqr, ul_3iqr)
    yield_loss_3iqr = _yield_loss(clean, ll_3iqr, ul_3iqr)

    return {
        "COUNT": int(n),
        "MEAN": mean,
        "MEDIAN": median,
        "STDEV": stdev,
        "IQR": iqr,
        "CPL": cpl,
        "CPU": cpu,
        "CPK": cpk,
        "%YLD LOSS": yield_loss,
        "LL_2CPK": ll_2cpk,
        "UL_2CPK": ul_2cpk,
        "CPK_2.0": cpk_2,
        "%YLD LOSS_2.0": yield_loss_2,
        "LL_3IQR": ll_3iqr,
        "UL_3IQR": ul_3iqr,
        "CPK_3IQR": cpk_3iqr,
        "%YLD LOSS_3IQR": yield_loss_3iqr,
        "_LOWER_SRC": lsl_src,
        "_UPPER_SRC": usl_src,
    }


def _yield_columns(group_keys: Sequence[str]) -> list[str]:
    if list(group_keys) == ["file", "site"]:
        return YIELD_SUMMARY_COLUMNS_SITE
    if list(group_keys) == ["file"]:
        return YIELD_SUMMARY_COLUMNS
    return list(group_keys) + ["devices_total", "devices_pass", "devices_fail", "yield_percent"]


def _pareto_columns(group_keys: Sequence[str]) -> list[str]:
    if list(group_keys) == ["file", "site"]:
        return PARETO_COLUMNS_SITE
    if list(group_keys) == ["file"]:
        return PARETO_COLUMNS
    return list(group_keys) + [
        "test_name",
        "test_number",
        "devices_fail",
        "fail_rate_percent",
        "cumulative_percent",
        "lower_limit",
        "upper_limit",
    ]


def _resolve_limits_for_rows(
    limit_map: dict[tuple[Optional[str], str, str], dict[str, Any]],
    files: Iterable[Any],
    test_names: Iterable[Any],
    test_numbers: Iterable[Any],
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    lower_limits: list[Optional[float]] = []
    upper_limits: list[Optional[float]] = []
    for file_value, test_name, test_number in zip(files, test_names, test_numbers):
        limit_info = _lookup_limit(limit_map, file_value, test_name, test_number)
        lower, _ = _select_limit("lower", limit_info)
        upper, _ = _select_limit("upper", limit_info)
        lower_limits.append(lower if lower is not None else np.nan)
        upper_limits.append(upper if upper is not None else np.nan)
    return lower_limits, upper_limits


def _device_limit_fail_flags(
    measurements: pd.DataFrame,
    group_keys: Sequence[str],
    limit_map: dict[tuple[Optional[str], str, str], dict[str, Any]],
) -> pd.DataFrame:
    required = set(group_keys) | {"test_name", "test_number", "value"}
    if measurements.empty or not required.issubset(measurements.columns):
        return pd.DataFrame(columns=list(group_keys) + ["device_key", "device_limit_fail"])

    subset_columns = list(group_keys) + ["test_name", "test_number", "value"]
    subset = measurements[subset_columns].copy()
    subset["test_name"] = subset["test_name"].astype(str)
    subset["test_number"] = subset["test_number"].astype(str)
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset["device_key"] = _normalise_device_id(measurements)

    files = subset["file"] if "file" in subset.columns else itertools.repeat(None)
    lower_limits, upper_limits = _resolve_limits_for_rows(
        limit_map,
        files,
        subset["test_name"],
        subset["test_number"],
    )
    subset["lower_limit"] = lower_limits
    subset["upper_limit"] = upper_limits

    lower_mask = subset["lower_limit"].notna() & (subset["value"] < subset["lower_limit"])
    upper_mask = subset["upper_limit"].notna() & (subset["value"] > subset["upper_limit"])
    subset["fail_by_limits"] = lower_mask | upper_mask

    grouped = (
        subset.groupby(list(group_keys) + ["device_key"], dropna=False, sort=False)["fail_by_limits"]
        .any()
        .rename("device_limit_fail")
        .reset_index()
    )
    return grouped


def _select_failure_rows(
    failing_subset: pd.DataFrame,
    group_keys: Sequence[str],
    first_failure_only: bool,
) -> pd.DataFrame:
    if failing_subset.empty:
        return pd.DataFrame(columns=failing_subset.columns)

    selected: list[pd.DataFrame] = []
    for _, device_rows in failing_subset.groupby(list(group_keys) + ["device_key"], dropna=False, sort=False):
        ordered = device_rows.sort_values(by="row_order")
        fail_rows = ordered[ordered["fail_by_limits"]]
        if fail_rows.empty:
            chosen = ordered.iloc[[0]]
        elif first_failure_only:
            chosen = fail_rows.iloc[[0]]
        else:
            chosen = fail_rows.drop_duplicates(subset=["test_name", "test_number"])
        selected.append(chosen)

    if not selected:
        return pd.DataFrame(columns=failing_subset.columns)
    return pd.concat(selected, ignore_index=True)


def _compute_yield_summary(
    measurements: pd.DataFrame,
    group_keys: Sequence[str],
    limits: pd.DataFrame | None,
) -> pd.DataFrame:
    required = set(group_keys)
    if measurements.empty or not required.issubset(measurements.columns):
        return pd.DataFrame(columns=_yield_columns(group_keys))

    device_keys = _normalise_device_id(measurements)
    limit_map = _build_limit_map(limits)
    status_series = measurements.get("part_status")
    has_status = status_series is not None

    device_frame = pd.DataFrame(index=measurements.index)
    for key in group_keys:
        device_frame[key] = measurements[key]
    device_frame["device_key"] = device_keys

    status_group = pd.DataFrame(columns=list(group_keys) + ["device_key", "fail_status"])
    if has_status:
        status_values = status_series.astype(str).str.upper()
        device_frame["fail_status"] = status_values != "PASS"
        status_group = (
            device_frame.groupby(list(group_keys) + ["device_key"], dropna=False, sort=False)["fail_status"]
            .any()
            .rename("fail_status")
            .reset_index()
        )

    limit_fail_group = _device_limit_fail_flags(measurements, group_keys, limit_map)

    merged = status_group.merge(
        limit_fail_group,
        on=list(group_keys) + ["device_key"],
        how="outer",
    )
    if "fail_status" not in merged:
        merged["fail_status"] = False
    if "device_limit_fail" not in merged:
        merged["device_limit_fail"] = False
    merged["fail_status"] = merged["fail_status"].where(merged["fail_status"].notna(), False).astype(bool)
    merged["device_limit_fail"] = merged["device_limit_fail"].where(
        merged["device_limit_fail"].notna(), False
    ).astype(bool)
    if has_status:
        merged["is_pass"] = ~merged["fail_status"].astype(bool)
    else:
        merged["is_pass"] = ~merged["device_limit_fail"].astype(bool)

    grouping_keys = list(group_keys)
    if merged.empty:
        return pd.DataFrame(columns=_yield_columns(group_keys))

    totals = merged.groupby(grouping_keys, dropna=False, sort=False)["device_key"].nunique().rename("devices_total")
    passes = (
        merged[merged["is_pass"]]
        .groupby(grouping_keys, dropna=False, sort=False)["device_key"]
        .nunique()
        .rename("devices_pass")
    )

    result = totals.to_frame().join(passes, how="left").fillna({"devices_pass": 0})
    result["devices_total"] = result["devices_total"].astype(int)
    result["devices_pass"] = result["devices_pass"].astype(int)
    result["devices_fail"] = (result["devices_total"] - result["devices_pass"]).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["yield_percent"] = np.where(
            result["devices_total"] > 0,
            result["devices_pass"] / result["devices_total"],
            np.nan,
        )
    result.reset_index(inplace=True)
    column_order = list(group_keys) + ["devices_total", "devices_pass", "devices_fail", "yield_percent"]
    result = result[column_order]
    result.sort_values(by=list(group_keys), inplace=True, ignore_index=True)
    return result


def _compute_pareto_details(
    measurements: pd.DataFrame,
    limits: pd.DataFrame,
    yield_summary: pd.DataFrame,
    group_keys: Sequence[str],
    first_failure_only: bool,
) -> pd.DataFrame:
    required_columns = set(group_keys) | {"test_name", "test_number", "value"}
    if not required_columns.issubset(measurements.columns):
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    limit_map = _build_limit_map(limits)
    subset_columns = list(group_keys) + ["test_name", "test_number", "value"]
    subset = measurements[subset_columns].copy()
    subset["test_name"] = subset["test_name"].astype(str)
    subset["test_number"] = subset["test_number"].astype(str)
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset["device_key"] = _normalise_device_id(measurements)
    subset["row_order"] = np.arange(len(subset))

    files = subset["file"] if "file" in subset.columns else [None] * len(subset)
    lower_limits, upper_limits = _resolve_limits_for_rows(
        limit_map,
        files,
        subset["test_name"],
        subset["test_number"],
    )
    subset["lower_limit"] = lower_limits
    subset["upper_limit"] = upper_limits

    lower_mask = subset["lower_limit"].notna() & (subset["value"] < subset["lower_limit"])
    upper_mask = subset["upper_limit"].notna() & (subset["value"] > subset["upper_limit"])
    subset["fail_by_limits"] = lower_mask | upper_mask

    device_limit_fail = _device_limit_fail_flags(measurements, group_keys, limit_map)
    status_series = measurements.get("part_status")
    status_group = pd.DataFrame(columns=list(group_keys) + ["device_key", "fail_status"])
    if status_series is not None:
        status_values = status_series.astype(str).str.upper()
        status_frame = pd.DataFrame(index=measurements.index)
        for key in group_keys:
            status_frame[key] = measurements[key]
        status_frame["device_key"] = subset["device_key"]
        status_frame["fail_status"] = status_values != "PASS"
        status_group = (
            status_frame.groupby(list(group_keys) + ["device_key"], dropna=False, sort=False)["fail_status"]
            .any()
            .rename("fail_status")
            .reset_index()
        )

    device_flags = status_group.merge(
        device_limit_fail,
        on=list(group_keys) + ["device_key"],
        how="outer",
    )
    device_flags["fail_status"] = device_flags.get("fail_status", False).where(
        device_flags.get("fail_status", False).notna(), False
    ).astype(bool)
    device_flags["device_limit_fail"] = device_flags.get("device_limit_fail", False).where(
        device_flags.get("device_limit_fail", False).notna(), False
    ).astype(bool)
    if status_series is not None:
        device_flags["is_device_fail"] = device_flags["fail_status"].astype(bool)
    else:
        device_flags["is_device_fail"] = device_flags["device_limit_fail"].astype(bool)

    subset = subset.merge(device_flags, on=list(group_keys) + ["device_key"], how="left")
    subset["is_device_fail"] = subset["is_device_fail"].fillna(False)
    subset["fail_status"] = subset["fail_status"].fillna(False)
    subset["device_limit_fail"] = subset["device_limit_fail"].fillna(False)

    failing_subset = subset[subset["is_device_fail"]]
    if failing_subset.empty:
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    selected = _select_failure_rows(failing_subset, group_keys, first_failure_only)
    if selected.empty:
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    dedupe_columns = list(group_keys) + ["test_name", "test_number", "device_key"]
    selected = selected.drop_duplicates(subset=dedupe_columns)

    agg_keys = list(group_keys) + ["test_name", "test_number"]
    grouped = selected.groupby(agg_keys, dropna=False, sort=False).agg(
        devices_fail=("device_key", "nunique"),
        lower_limit=("lower_limit", "first"),
        upper_limit=("upper_limit", "first"),
    )
    grouped.reset_index(inplace=True)

    if not yield_summary.empty:
        totals_frame = yield_summary[list(group_keys) + ["devices_total"]].copy()
        grouped = grouped.merge(totals_frame, on=list(group_keys), how="left")
    else:
        grouped["devices_total"] = 0
    grouped["devices_total"] = grouped["devices_total"].fillna(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        grouped["fail_rate_percent"] = np.where(
            grouped["devices_total"] > 0,
            grouped["devices_fail"] / grouped["devices_total"],
            np.nan,
        )
    sort_columns = list(group_keys) + ["devices_fail", "test_name", "test_number"]
    sort_order = [True] * len(group_keys) + [False, True, True]
    grouped.sort_values(by=sort_columns, ascending=sort_order, inplace=True)
    grouped["cumulative_percent"] = grouped.groupby(list(group_keys))["fail_rate_percent"].cumsum()

    result_columns = list(group_keys) + [
        "test_name",
        "test_number",
        "devices_fail",
        "fail_rate_percent",
        "cumulative_percent",
        "lower_limit",
        "upper_limit",
    ]
    result = grouped[result_columns].reset_index(drop=True)
    return result


def _select_limit(side: str, info: dict[str, Any]) -> tuple[Optional[float], LimitSource]:
    candidates = [
        (info.get(f"what_if_{side}"), "what_if"),
        (info.get(f"spec_{side}"), "spec"),
        (info.get(f"stdf_{side}"), "stdf"),
    ]
    for value, source in candidates:
        value = _coerce_float(value)
        if value is not None and math.isfinite(value):
            return float(value), source  # type: ignore[return-value]
    return None, "unset"


def _process_capability_lower(mean: float, stdev: float, lsl: Optional[float]) -> float | None:
    if lsl is None:
        return None
    diff = mean - lsl
    return _capability_ratio(diff, stdev)


def _process_capability_upper(mean: float, stdev: float, usl: Optional[float]) -> float | None:
    if usl is None:
        return None
    diff = usl - mean
    return _capability_ratio(diff, stdev)


def _capability_ratio(diff: float, stdev: float) -> float:
    if stdev > 0:
        return diff / (3 * stdev)
    if diff > 0:
        return math.inf
    if diff < 0:
        return -math.inf
    return math.inf


def _merge_capabilities(cpl: Optional[float], cpu: Optional[float]) -> Optional[float]:
    if cpl is None and cpu is None:
        return None
    if cpl is None:
        return cpu
    if cpu is None:
        return cpl
    return cpl if cpl < cpu else cpu


def _compute_cpk(center: float, sigma: float, lower: Optional[float], upper: Optional[float]) -> Optional[float]:
    if lower is None and upper is None:
        return None
    cpl = _capability_ratio(center - lower, sigma) if lower is not None else None
    cpu = _capability_ratio(upper - center, sigma) if upper is not None else None
    return _merge_capabilities(cpl, cpu)


def _normalise_device_id(frame: pd.DataFrame, column: str = "device_id") -> pd.Series:
    if column not in frame.columns:
        values = ["__row_" + str(idx) for idx in frame.index]
        return pd.Series(values, index=frame.index, dtype="object")
    series = frame[column]
    text = series.astype(str).str.strip()
    mask = pd.isna(series) | (text == "") | (text.str.lower() == "nan")
    if not mask.any():
        return text
    fallback = "__row_" + frame.index.astype(str)
    combined = np.where(mask.to_numpy(), fallback.to_numpy(), text.to_numpy())
    return pd.Series(combined, index=frame.index, dtype="object")


def _yield_loss(values: np.ndarray, lower: Optional[float], upper: Optional[float]) -> float:
    if lower is None and upper is None:
        return math.nan
    failures = 0
    if lower is not None:
        failures += int(np.sum(values < lower))
    if upper is not None:
        failures += int(np.sum(values > upper))
    return (failures / values.size) if values.size else math.nan


def _first_not_empty(series: Iterable[Any]) -> Optional[str]:
    for value in series:
        text = _coerce_str(value)
        if text:
            return text
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    try:
        stripped = str(value).strip()
    except (TypeError, ValueError):
        return None
    return stripped or None
