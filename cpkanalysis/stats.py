from __future__ import annotations

import math
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    yield_frame = _compute_yield_summary(measurements, ["file"])
    pareto_frame = _compute_pareto_details(measurements, limits, yield_frame, ["file"])
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "site" not in measurements.columns:
        empty_yield = pd.DataFrame(columns=YIELD_SUMMARY_COLUMNS_SITE)
        empty_pareto = pd.DataFrame(columns=PARETO_COLUMNS_SITE)
        return empty_yield, empty_pareto
    yield_frame = _compute_yield_summary(measurements, ["file", "site"])
    pareto_frame = _compute_pareto_details(
        measurements,
        limits,
        yield_frame,
        ["file", "site"],
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
        limit_info = limit_map.get((str(test_name), str(test_number)), {})
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
    result: dict[tuple[str, str], dict[str, Any]] = {}
    if limits is None or limits.empty:
        return result
    rows = limits.fillna(value=pd.NA)
    for _, row in rows.iterrows():
        key = (str(row.get("test_name", "")), str(row.get("test_number", "")))
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
    cpk_3iqr = _compute_cpk(median, robust_sigma, ll_3iqr, ul_3iqr)
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


def _compute_yield_summary(measurements: pd.DataFrame, group_keys: Sequence[str]) -> pd.DataFrame:
    required = set(group_keys)
    if measurements.empty or not required.issubset(measurements.columns):
        return pd.DataFrame(columns=_yield_columns(group_keys))

    device_keys = _normalise_device_id(measurements)
    status_series = measurements.get("part_status")
    if status_series is None:
        status_series = pd.Series(data="PASS", index=measurements.index, dtype="object")
    status_values = status_series.astype(str).str.upper()
    is_pass = status_values == "PASS"

    device_status = pd.DataFrame(index=measurements.index)
    for key in group_keys:
        device_status[key] = measurements[key]
    device_status["device_key"] = device_keys
    device_status["is_pass"] = is_pass

    grouping_keys = list(group_keys) + ["device_key"]
    per_device = device_status.groupby(grouping_keys, dropna=False, sort=False)["is_pass"].all().reset_index()
    if per_device.empty:
        return pd.DataFrame(columns=_yield_columns(group_keys))

    base_group = per_device.groupby(list(group_keys), dropna=False, sort=False)
    totals = base_group["device_key"].count().rename("devices_total")
    passes = (
        per_device[per_device["is_pass"]]
        .groupby(list(group_keys), dropna=False, sort=False)["device_key"]
        .count()
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
) -> pd.DataFrame:
    required_columns = set(group_keys) | {"test_name", "test_number", "value"}
    if not required_columns.issubset(measurements.columns):
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    limit_map = _build_limit_map(limits)
    if not limit_map:
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    limit_records: list[dict[str, Any]] = []
    for (test_name, test_number), info in limit_map.items():
        lower, _ = _select_limit("lower", info)
        upper, _ = _select_limit("upper", info)
        if lower is None and upper is None:
            continue
        limit_records.append(
            {
                "test_name": test_name,
                "test_number": test_number,
                "lower_limit": lower,
                "upper_limit": upper,
            }
        )

    if not limit_records:
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    limits_frame = pd.DataFrame(limit_records)
    limits_frame["test_name"] = limits_frame["test_name"].astype(str)
    limits_frame["test_number"] = limits_frame["test_number"].astype(str)

    subset_columns = list(group_keys) + ["test_name", "test_number", "value"]
    subset = measurements[subset_columns].copy()
    subset["test_name"] = subset["test_name"].astype(str)
    subset["test_number"] = subset["test_number"].astype(str)
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset["device_key"] = _normalise_device_id(measurements)

    merged = subset.merge(limits_frame, on=["test_name", "test_number"], how="left")
    if merged.empty:
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    valid_limits = merged["lower_limit"].notna() | merged["upper_limit"].notna()
    merged = merged.loc[valid_limits]
    if merged.empty:
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    lower_mask = merged["lower_limit"].notna() & (merged["value"] < merged["lower_limit"])
    upper_mask = merged["upper_limit"].notna() & (merged["value"] > merged["upper_limit"])
    fail_mask = lower_mask | upper_mask
    selected_columns = list(group_keys) + ["test_name", "test_number", "device_key", "lower_limit", "upper_limit"]
    failed = merged.loc[fail_mask, selected_columns]
    if failed.empty:
        return pd.DataFrame(columns=_pareto_columns(group_keys))

    dedupe_columns = list(group_keys) + ["test_name", "test_number", "device_key"]
    failed = failed.drop_duplicates(subset=dedupe_columns)
    agg_keys = list(group_keys) + ["test_name", "test_number"]
    grouped = failed.groupby(agg_keys, dropna=False, sort=False).agg(
        devices_fail=("device_key", "size"),
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
