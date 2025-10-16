from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import LimitSource

GROUP_KEYS = ["file", "test_name", "test_number"]

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


def compute_summary(
    measurements: pd.DataFrame,
    limits: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, LimitSource]]]:
    if measurements.empty:
        empty = pd.DataFrame(columns=SUMMARY_COLUMNS)
        return empty, {}

    limit_map = _build_limit_map(limits)
    records: list[dict[str, Any]] = []
    limit_sources: dict[tuple[str, str, str], dict[str, LimitSource]] = {}

    grouped = measurements.groupby(GROUP_KEYS, dropna=False, sort=False)
    for (file_name, test_name, test_number), group in grouped:
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        unit = _first_not_empty(group["units"])
        limit_info = limit_map.get((test_name, test_number), {})
        stats = _compute_group_statistics(values.to_numpy(), limit_info)
        stats["File"] = file_name
        stats["Test Name"] = test_name
        stats["Test Number"] = test_number
        stats["Unit"] = unit or limit_info.get("unit", "")
        records.append(stats)
        limit_sources[(file_name, test_name, test_number)] = {
            "lower": stats.get("_LOWER_SRC", "unset"),
            "upper": stats.get("_UPPER_SRC", "unset"),
        }

    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS + ["_LOWER_SRC", "_UPPER_SRC"])
    if "_LOWER_SRC" in summary.columns:
        summary = summary[SUMMARY_COLUMNS]
    summary.sort_values(by=["File", "Test Name", "Test Number"], inplace=True, ignore_index=True)
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
    except Exception:
        return None
    return stripped or None
