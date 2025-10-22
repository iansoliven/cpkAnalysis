"""Helpers for GRR-driven proposed limit calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

__all__ = [
    "GRRTable",
    "GRRRecord",
    "ComputationResult",
    "ComputationError",
    "load_grr_table",
    "normalize_test_number",
    "normalize_test_name",
    "normalize_unit",
    "compute_proposed_limits",
]

UNIT_ALIASES: Dict[str, str] = {
    "V": "VOLTS",
    "VOLTS": "VOLTS",
    "VOLT": "VOLTS",
    "MV": "MILLIVOLTS",
    "MILLIVOLTS": "MILLIVOLTS",
    "A": "AMPS",
    "AMP": "AMPS",
    "AMPS": "AMPS",
    "MA": "MILLIAMPS",
    "MILLIAMPS": "MILLIAMPS",
    "UA": "UAMPS",
    "UAMP": "UAMPS",
    "UAMPS": "UAMPS",
    "NA": "NAMPS",
    "NAMPS": "NAMPS",
    "OHM": "OHMS",
    "OHMS": "OHMS",
    "KOHM": "KOHM",
    "KOHMS": "KOHM",
    "NS": "NSECONDS",
    "NSEC": "NSECONDS",
    "NSECONDS": "NSECONDS",
}

_EPSILON = 1e-9


def _safe_float(value) -> Optional[float]:
    if value in (None, "", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _safe_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text.strip()


def _normalize_header(text: str) -> str:
    parts = str(text).strip().lower().split()
    return " ".join(parts)


def normalize_test_number(text: str) -> str:
    raw = _safe_text(text).upper()
    if not raw:
        return ""
    if raw.startswith("T"):
        raw = raw[1:]
    raw = raw.lstrip("0")
    return raw


def normalize_test_name(text: str) -> str:
    raw = _safe_text(text).lower()
    return " ".join(raw.split())


def normalize_unit(text: str) -> str:
    raw = "".join(ch for ch in _safe_text(text).upper() if not ch.isspace())
    return UNIT_ALIASES.get(raw, raw)


@dataclass(frozen=True)
class GRRRecord:
    test_number: str
    test_name: str
    unit_raw: str
    unit_normalized: str
    spec_lower: Optional[float]
    spec_upper: Optional[float]
    guardband_full: Optional[float]


class GRRTable:
    """Lookup helper for GRR records keyed by test number/name."""

    def __init__(self, records: Iterable[GRRRecord]):
        by_number: Dict[Tuple[str, str], GRRRecord] = {}
        by_name: Dict[str, GRRRecord] = {}
        for record in records:
            key = (record.test_number, record.test_name)
            if record.test_number:
                by_number[key] = record
            if record.test_name:
                by_name.setdefault(record.test_name, record)
        self._by_number = by_number
        self._by_name = by_name

    def find(self, test_number: str, test_name: str) -> Optional[GRRRecord]:
        number_key = normalize_test_number(test_number)
        name_key = normalize_test_name(test_name)
        if number_key:
            record = self._by_number.get((number_key, name_key))
            if record:
                return record
        return self._by_name.get(name_key)


class ComputationError(RuntimeError):
    """Raised when proposed-limit calculation cannot proceed."""


@dataclass
class ComputationResult:
    guardband_label: str
    guardband_value: float
    guardband_cpk: Optional[float]
    ft_lower: float
    ft_upper: float
    ft_cpk: Optional[float]
    spec_lower: float
    spec_upper: float
    spec_cpk: Optional[float]
    spec_widened: bool
    notes: List[str]


def _find_column(df: pd.DataFrame, aliases: Iterable[str]) -> Optional[str]:
    headers = {_normalize_header(col): col for col in df.columns}
    for alias in aliases:
        key = _normalize_header(alias)
        if key in headers:
            return headers[key]
    return None


def load_grr_table(path: Path) -> GRRTable:
    candidate = path
    if candidate.is_dir():
        candidate = candidate / "Total_GRR.xlsx"
    if not candidate.exists():
        raise FileNotFoundError(f"GRR workbook not found: {candidate}")

    xls = pd.ExcelFile(candidate)
    sheet_name = "GRR data" if "GRR data" in xls.sheet_names else xls.sheet_names[0]
    raw = xls.parse(sheet_name, header=None)
    header_index: Optional[int] = None
    for idx, row in raw.iterrows():
        normalized = [_normalize_header(value) for value in row.tolist()]
        if "test #" in normalized and (
            "test name" in normalized or "description" in normalized
        ):
            header_index = idx
            break

    if header_index is not None:
        header_values = raw.iloc[header_index].tolist()
        frame = raw.iloc[header_index + 1 :].copy()
        frame.columns = header_values
    else:
        frame = raw.copy()

    frame.dropna(how="all", inplace=True)
    if frame.empty:
        raise ValueError(f"GRR sheet '{sheet_name}' is empty.")

    test_number_col = _find_column(frame, ["Test #", "Test Number", "Test Num", "Test"])
    test_name_col = _find_column(frame, ["Test Name", "Description"])
    unit_col = _find_column(frame, ["Unit", "Units"])
    spec_lower_col = _find_column(frame, ["Min Spec", "Min\nSpec", "Spec Min", "Lower Spec", "Spec Lower"])
    spec_upper_col = _find_column(frame, ["Max Spec", "Max\nSpec", "Spec Max", "Upper Spec", "Spec Upper"])
    grr_col = _find_column(frame, ["Worst R&R", "R&R", "Total R&R", "GRR"])

    records: List[GRRRecord] = []
    for _, row in frame.iterrows():
        name = normalize_test_name(row.get(test_name_col, "")) if test_name_col else ""
        number = normalize_test_number(row.get(test_number_col, "")) if test_number_col else ""
        if not name and not number:
            continue
        raw_unit = _safe_text(row.get(unit_col)) if unit_col else ""
        record = GRRRecord(
            test_number=number,
            test_name=name,
            unit_raw=raw_unit,
            unit_normalized=normalize_unit(raw_unit),
            spec_lower=_safe_float(row.get(spec_lower_col)) if spec_lower_col else None,
            spec_upper=_safe_float(row.get(spec_upper_col)) if spec_upper_col else None,
            guardband_full=_safe_float(row.get(grr_col)) if grr_col else None,
        )
        records.append(record)

    return GRRTable(records)


def _calc_cpk(mean: float, stdev: float, lower: Optional[float], upper: Optional[float]) -> Optional[float]:
    if stdev <= 0:
        return None
    candidates: List[float] = []
    if upper is not None:
        candidates.append((upper - mean) / (3.0 * stdev))
    if lower is not None:
        candidates.append((mean - lower) / (3.0 * stdev))
    if not candidates:
        return None
    value = min(candidates)
    if not math.isfinite(value):
        return None
    return float(value)


def compute_proposed_limits(
    *,
    mean: float,
    stdev: float,
    spec_lower: float,
    spec_upper: float,
    guardband_full: float,
    cpk_min: float,
    cpk_max: float,
) -> ComputationResult:
    if stdev <= 0:
        raise ComputationError("Standard deviation must be positive.")
    if spec_lower is None or spec_upper is None:
        raise ComputationError("Specification limits are required.")
    if spec_upper <= spec_lower:
        raise ComputationError("Specification limits are invalid.")
    if guardband_full is None or guardband_full <= 0:
        raise ComputationError("GRR guardband must be positive.")

    guardbands: List[Tuple[str, float]] = [
        ("100% GRR", guardband_full),
        ("50% GRR", guardband_full / 2.0),
    ]

    best_choice: Optional[Tuple[str, float, float]] = None
    selected_choice: Optional[Tuple[str, float, float]] = None

    for label, guardband in guardbands:
        if guardband <= 0:
            continue
        ft_lower = spec_lower + guardband
        ft_upper = spec_upper - guardband
        if ft_lower >= ft_upper:
            continue
        cpk_value = _calc_cpk(mean, stdev, ft_lower, ft_upper)
        if cpk_value is None:
            continue
        if cpk_value >= cpk_min:
            selected_choice = (label, guardband, cpk_value)
            break
        if best_choice is None or cpk_value > best_choice[2]:
            best_choice = (label, guardband, cpk_value)

    if selected_choice is None:
        if best_choice is None:
            raise ComputationError("Unable to derive guardband from GRR data.")
        selected_choice = best_choice

    selected_label, guardband, selected_cpk = selected_choice
    spec_lower_prop = float(spec_lower)
    spec_upper_prop = float(spec_upper)
    spec_widened = False
    notes: List[str] = []

    if spec_upper_prop - spec_lower_prop <= (2.0 * guardband + _EPSILON):
        delta = (2.0 * guardband + _EPSILON) - (spec_upper_prop - spec_lower_prop)
        spec_lower_prop -= delta / 2.0
        spec_upper_prop += delta / 2.0
        spec_widened = True
        notes.append("Specification widened to accommodate guardband width.")

    if cpk_min > 0:
        required_lower = mean - guardband - 3.0 * stdev * cpk_min
        required_upper = mean + guardband + 3.0 * stdev * cpk_min
        if required_lower < spec_lower_prop - _EPSILON:
            spec_lower_prop = required_lower
            spec_widened = True
        if required_upper > spec_upper_prop + _EPSILON:
            spec_upper_prop = required_upper
            spec_widened = True

    ft_lower = spec_lower_prop + guardband
    ft_upper = spec_upper_prop - guardband
    ft_cpk = _calc_cpk(mean, stdev, ft_lower, ft_upper)

    if cpk_min > 0 and (ft_cpk is None or ft_cpk + _EPSILON < cpk_min):
        required_lower = mean - 3.0 * stdev * cpk_min
        required_upper = mean + 3.0 * stdev * cpk_min
        if ft_lower > required_lower:
            adjustment = ft_lower - required_lower
            spec_lower_prop -= adjustment
            spec_widened = True
        if ft_upper < required_upper:
            adjustment = required_upper - ft_upper
            spec_upper_prop += adjustment
            spec_widened = True
        ft_lower = spec_lower_prop + guardband
        ft_upper = spec_upper_prop - guardband
        ft_cpk = _calc_cpk(mean, stdev, ft_lower, ft_upper)

    if ft_lower >= ft_upper:
        mid = (spec_lower_prop + spec_upper_prop) / 2.0
        span = max(6.0 * stdev, 2.0 * guardband + _EPSILON)
        spec_lower_prop = mid - span / 2.0
        spec_upper_prop = mid + span / 2.0
        spec_widened = True
        ft_lower = spec_lower_prop + guardband
        ft_upper = spec_upper_prop - guardband
        ft_cpk = _calc_cpk(mean, stdev, ft_lower, ft_upper)

    if ft_cpk is not None and cpk_max > 0 and ft_cpk > cpk_max + _EPSILON:
        required_half = 3.0 * stdev * cpk_max
        candidate_lower = mean - required_half
        candidate_upper = mean + required_half
        ft_lower = max(candidate_lower, spec_lower_prop + guardband)
        ft_upper = min(candidate_upper, spec_upper_prop - guardband)
        ft_cpk = _calc_cpk(mean, stdev, ft_lower, ft_upper)
        notes.append("FT guardband tightened to respect CPK ceiling.")

    spec_cpk = _calc_cpk(mean, stdev, spec_lower_prop, spec_upper_prop)

    return ComputationResult(
        guardband_label=selected_label,
        guardband_value=guardband,
        guardband_cpk=selected_cpk,
        ft_lower=ft_lower,
        ft_upper=ft_upper,
        ft_cpk=ft_cpk,
        spec_lower=spec_lower_prop,
        spec_upper=spec_upper_prop,
        spec_cpk=spec_cpk,
        spec_widened=spec_widened,
        notes=notes,
    )
