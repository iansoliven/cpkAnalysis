from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .models import IngestResult, SourceFile

ISTDF_SRC = Path(__file__).resolve().parents[1] / "Submodules" / "istdf" / "src"
if ISTDF_SRC.exists() and str(ISTDF_SRC) not in sys.path:  # pragma: no cover - runtime path injection
    sys.path.insert(0, str(ISTDF_SRC))

try:  # pragma: no cover - dependency injection based on submodule availability
    from istdf import STDFReader  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    STDFReader = None  # type: ignore[assignment]


@dataclass
class _TestMetadata:
    test_name: str = ""
    unit: str = ""
    low_limit: float | None = None
    high_limit: float | None = None
    scale: int | None = None


EXPECTED_COLUMNS = [
    "file",
    "file_path",
    "device_id",
    "device_sequence",
    "test_name",
    "test_number",
    "units",
    "value",
    "stdf_lower",
    "stdf_upper",
    "timestamp",
    "measurement_index",
    "part_status",
]


def ingest_sources(sources: Sequence[SourceFile], temp_dir: Path) -> IngestResult:
    """Load STDF measurements into a columnar store and return metadata."""
    if not sources:
        raise ValueError("No STDF sources supplied.")
    if STDFReader is None:
        raise RuntimeError(
            "STDF support unavailable - ensure Submodules/istdf is initialised and importable."
        )

    temp_dir.mkdir(parents=True, exist_ok=True)
    raw_store_path = temp_dir / "raw_measurements.parquet"
    if raw_store_path.exists():
        raw_store_path.unlink()

    per_file_stats: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []
    test_catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for index, source in enumerate(sources, start=1):
        frame, test_meta, stats = _parse_stdf_file(source, index)
        if not frame.empty:
            all_frames.append(frame)
        per_file_stats.append(stats)
        for key, info in test_meta.items():
            existing = test_catalog.get(key)
            if existing is None:
                test_catalog[key] = info
                continue
            if existing.get("unit") in ("", None) and info.get("unit"):
                existing["unit"] = info["unit"]
            if existing.get("stdf_lower") is None and info.get("stdf_lower") is not None:
                existing["stdf_lower"] = info["stdf_lower"]
            if existing.get("stdf_upper") is None and info.get("stdf_upper") is not None:
                existing["stdf_upper"] = info["stdf_upper"]

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=EXPECTED_COLUMNS)

    table = pa.Table.from_pandas(combined, preserve_index=False)
    pq.write_table(table, raw_store_path, compression="snappy")

    catalog_rows = [
        {
            "test_name": key[0],
            "test_number": key[1],
            "unit": info.get("unit") or "",
            "stdf_lower": info.get("stdf_lower"),
            "stdf_upper": info.get("stdf_upper"),
            "spec_lower": None,
            "spec_upper": None,
            "what_if_lower": None,
            "what_if_upper": None,
        }
        for key, info in sorted(test_catalog.items())
    ]
    catalog_frame = pd.DataFrame(catalog_rows)

    return IngestResult(
        frame=combined,
        test_catalog=catalog_frame,
        per_file_stats=per_file_stats,
        raw_store_path=raw_store_path,
    )


def _parse_stdf_file(source: SourceFile, file_index: int) -> tuple[pd.DataFrame, Dict[Tuple[str, str], Dict[str, Any]], dict[str, Any]]:
    """Parse a single STDF file into a DataFrame plus metadata."""
    fallback_index = 0
    device_sequence = 0
    current_serial: Optional[str] = None
    current_site: Optional[int] = None
    current_measurements: list[dict[str, Any]] = []
    test_metadata: Dict[str, _TestMetadata] = {}
    test_catalog: Dict[Tuple[str, str], Dict[str, Any]] = {}
    device_test_counters: Dict[Tuple[str, str, str], int] = defaultdict(int)
    rows: list[dict[str, Any]] = []

    with STDFReader(source.path, ignore_unknown=True) as reader:
        for record in reader:
            name = record.name
            data = record.to_dict()
            if name == "PIR":
                current_measurements.clear()
                current_serial = _coerce_str(data.get("PART_ID"))
                current_site = _coerce_int(data.get("SITE_NUM"))
                device_sequence += 1
            elif name == "PTR":
                measurement = _extract_measurement(data, test_metadata)
                if measurement is not None:
                    current_measurements.append(measurement)
            elif name == "PRR":
                status = "FAIL" if _is_part_fail(data.get("PART_FLG")) else "PASS"
                part_id = _coerce_str(data.get("PART_ID")) or current_serial
                if part_id is None:
                    part_id = f"SITE{current_site or 0}_{fallback_index}"
                    fallback_index += 1
                for measurement in current_measurements:
                    key = (part_id, measurement["test_name"], measurement["test_number"])
                    device_test_counters[key] += 1
                    measurement_index = device_test_counters[key]
                    timestamp = measurement.get("test_time")
                    rows.append(
                        {
                            "file": source.file_name,
                            "file_path": str(source.path),
                            "device_id": part_id,
                            "device_sequence": device_sequence,
                            "test_name": measurement["test_name"],
                            "test_number": measurement["test_number"],
                            "units": measurement["test_unit"],
                            "value": measurement["measurement"],
                            "stdf_lower": measurement["low_limit"],
                            "stdf_upper": measurement["high_limit"],
                            "timestamp": timestamp if timestamp is not None else float(measurement_index),
                            "measurement_index": measurement_index,
                            "part_status": status,
                        }
                    )
                    catalog_key = (measurement["test_name"], measurement["test_number"])
                    catalog_entry = test_catalog.setdefault(
                        catalog_key,
                        {
                            "unit": measurement["test_unit"],
                            "stdf_lower": measurement["low_limit"],
                            "stdf_upper": measurement["high_limit"],
                        },
                    )
                    if catalog_entry.get("unit") in ("", None) and measurement["test_unit"]:
                        catalog_entry["unit"] = measurement["test_unit"]
                    if catalog_entry.get("stdf_lower") is None and measurement["low_limit"] is not None:
                        catalog_entry["stdf_lower"] = measurement["low_limit"]
                    if catalog_entry.get("stdf_upper") is None and measurement["high_limit"] is not None:
                        catalog_entry["stdf_upper"] = measurement["high_limit"]
                current_measurements.clear()
                current_serial = None
                current_site = None

    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    stats = {
        "file": source.file_name,
        "path": str(source.path),
        "measurement_count": int(len(rows)),
        "device_count": int(device_sequence),
    }
    return frame, test_catalog, stats


def _extract_measurement(data: Dict[str, Any], cache: Dict[str, _TestMetadata]) -> Optional[dict[str, Any]]:
    test_num = data.get("TEST_NUM")
    test_name_raw = data.get("TEST_TXT")
    if test_num is None and not test_name_raw:
        return None

    key = str(test_num) if test_num is not None else f"name:{test_name_raw or ''}"
    metadata = cache.get(key, _TestMetadata())

    test_name = _coerce_str(test_name_raw) or metadata.test_name
    if test_name:
        metadata.test_name = test_name

    unit_raw = data.get("UNITS")
    unit_value = _coerce_str(unit_raw) or metadata.unit
    if unit_value:
        metadata.unit = unit_value

    res_scale = _select_scale(data.get("RES_SCAL"), metadata.scale)
    measurement_value = _apply_scale(_coerce_float(data.get("RESULT")), res_scale)

    llm_scale = _select_scale(data.get("LLM_SCAL"), res_scale, metadata.scale)
    low_limit = _apply_scale(_coerce_float(data.get("LO_LIMIT")), llm_scale)
    if low_limit is not None:
        metadata.low_limit = low_limit
    else:
        low_limit = metadata.low_limit

    hlm_scale = _select_scale(data.get("HLM_SCAL"), res_scale, metadata.scale)
    high_limit = _apply_scale(_coerce_float(data.get("HI_LIMIT")), hlm_scale)
    if high_limit is not None:
        metadata.high_limit = high_limit
    else:
        high_limit = metadata.high_limit

    metadata.scale = res_scale
    cache[key] = metadata

    catalog_name = metadata.test_name or test_name or ""
    catalog_number = str(test_num) if test_num is not None else ""
    return {
        "test_number": catalog_number,
        "test_name": catalog_name,
        "test_unit": _compose_unit(metadata.unit, metadata.scale),
        "low_limit": low_limit,
        "high_limit": high_limit,
        "measurement": measurement_value,
        "test_time": _coerce_float(data.get("TEST_TIM")),
    }


def _apply_scale(value: Optional[float], scale: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    if scale in (None, 0):
        return value
    try:
        return value * (10 ** scale)
    except Exception:
        return value


def _select_scale(*candidates: Optional[int]) -> Optional[int]:
    for candidate in candidates:
        if candidate not in (None, 0):
            return candidate
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


_PREFIX_BY_EXPONENT = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "u",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}


def _compose_unit(base_unit: str, scale: Optional[int]) -> str:
    base = (base_unit or "").strip()
    if not base:
        if scale in (2, -2):
            return "%"
        return ""
    if scale in (None, 0):
        return base
    prefix = _PREFIX_BY_EXPONENT.get(-scale)
    if prefix is None:
        return base
    return f"{prefix}{base}"


def _is_part_fail(flag: Any) -> bool:
    if flag is None:
        return False
    numeric: int
    if isinstance(flag, bytes):
        if not flag:
            return False
        numeric = flag[0]
    else:
        try:
            numeric = int(flag)
        except (TypeError, ValueError):
            return False
    return bool(numeric & 0x08)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
