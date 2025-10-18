from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Tuple, TypedDict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .models import IngestResult, SourceFile

ISTDF_SRC = Path(__file__).resolve().parents[1] / "Submodules" / "istdf" / "src"


def _load_local_istdf() -> ModuleType | None:
    """Attempt to import the bundled istdf package without modifying sys.path."""
    package_root = ISTDF_SRC / "istdf"
    init_path = package_root / "__init__.py"
    if not init_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "istdf",
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["istdf"] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - defensive: third-party import
        sys.modules.pop("istdf", None)
        warnings.warn(
            f"Unable to initialise the 'istdf' submodule: {exc}. STDF ingestion features will be unavailable.",
            RuntimeWarning,
        )
        return None
    return module


def _resolve_stdf_reader() -> Any:
    try:  # pragma: no cover - dependency injection based on submodule availability
        from istdf import STDFReader as reader  # type: ignore
        return reader
    except ImportError:
        if not ISTDF_SRC.exists():
            warnings.warn(
                "istdf submodule not available; STDF ingestion features will be unavailable.",
                RuntimeWarning,
            )
            return None
        module = _load_local_istdf()
        if module is None:
            return None
        reader = getattr(module, "STDFReader", None)
        if reader is None:
            warnings.warn(
                "istdf submodule loaded but STDFReader not found; STDF ingestion features will be unavailable.",
                RuntimeWarning,
            )
        return reader


STDFReader = _resolve_stdf_reader()


@dataclass
class MeasurementRecord(TypedDict, total=False):
    test_number: str
    test_name: str
    test_unit: str
    low_limit: Optional[float]
    high_limit: Optional[float]
    measurement: Optional[float]
    test_time: Optional[float]
    stdf_result_format: Optional[str]
    stdf_lower_format: Optional[str]
    stdf_upper_format: Optional[str]
    stdf_result_scale: Optional[int]
    stdf_lower_scale: Optional[int]
    stdf_upper_scale: Optional[int]


class _TestMetadata:
    test_name: str = ""
    unit: str = ""
    low_limit: float | None = None
    high_limit: float | None = None
    has_low_limit: bool = False
    has_high_limit: bool = False
    scale: int | None = None
    result_format: Optional[str] = None
    low_format: Optional[str] = None
    high_format: Optional[str] = None
    result_scale: Optional[int] = None
    low_scale: Optional[int] = None
    high_scale: Optional[int] = None


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
    "stdf_result_format",
    "stdf_lower_format",
    "stdf_upper_format",
    "stdf_result_scale",
    "stdf_lower_scale",
    "stdf_upper_scale",
    "timestamp",
    "measurement_index",
    "part_status",
]

# STDF flag constants
RESULT_INVALID = 0x04  # PARM_FLG bit 2: measurement result is invalid

# TEST_FLG constants - reject measurements with these flags (corrected per STDF spec)
TEST_RESULT_NOT_VALID = 0x02       # Bit 1 (2): Result is not valid
TEST_RESULT_UNRELIABLE = 0x04      # Bit 2 (4): Test result is unreliable  
TEST_TIMEOUT_OCCURRED = 0x08       # Bit 3 (8): Timeout occurred
TEST_NOT_EXECUTED = 0x10           # Bit 4 (16): Test not executed
TEST_ABORTED = 0x20                # Bit 5 (32): Test aborted
TEST_COMPLETED_NO_PF = 0x40        # Bit 6 (64): Test completed without P/F indication

# Combined mask for TEST_FLG rejection criteria
TEST_FLG_REJECT_MASK = TEST_RESULT_NOT_VALID | TEST_RESULT_UNRELIABLE | TEST_TIMEOUT_OCCURRED | TEST_NOT_EXECUTED | TEST_ABORTED | TEST_COMPLETED_NO_PF


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
            existing.setdefault("has_stdf_lower", existing.get("stdf_lower") is not None)
            existing.setdefault("has_stdf_upper", existing.get("stdf_upper") is not None)
            info_low_flag = info.get("has_stdf_lower")
            info_high_flag = info.get("has_stdf_upper")
            # Update limits: allow explicit updates and clears based on has_stdf_* flags
            # If has_stdf_lower is True, update (even if value is None - explicit clear)
            # If has_stdf_lower is False and value is None, it's an explicit "no limit"
            if info_low_flag is True:
                existing["stdf_lower"] = info["stdf_lower"]
                existing["has_stdf_lower"] = info["stdf_lower"] is not None
            elif info_low_flag is False and info.get("stdf_lower") is None:
                # Explicit "no limit" case
                existing["stdf_lower"] = None
                existing["has_stdf_lower"] = False
            if info_high_flag is True:
                existing["stdf_upper"] = info["stdf_upper"]
                existing["has_stdf_upper"] = info["stdf_upper"] is not None
            elif info_high_flag is False and info.get("stdf_upper") is None:
                # Explicit "no limit" case
                existing["stdf_upper"] = None
                existing["has_stdf_upper"] = False

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
            "stdf_lower": info.get("stdf_lower") if info.get("has_stdf_lower", info.get("stdf_lower") is not None) else None,
            "stdf_upper": info.get("stdf_upper") if info.get("has_stdf_upper", info.get("stdf_upper") is not None) else None,
            "stdf_result_format": info.get("stdf_result_format") or None,
            "stdf_lower_format": info.get("stdf_lower_format") or None,
            "stdf_upper_format": info.get("stdf_upper_format") or None,
            "stdf_result_scale": info.get("stdf_result_scale"),
            "stdf_lower_scale": info.get("stdf_lower_scale"),
            "stdf_upper_scale": info.get("stdf_upper_scale"),
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
    invalid_measurements_count = 0  # Track invalid measurements

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
                # First, always try to populate test catalog regardless of measurement validity
                # This ensures all tests appear in the catalog even if all measurements are invalid
                _populate_test_catalog_from_ptr(data, test_metadata, test_catalog)
                
                # Then process measurement for data collection (with filtering)
                measurement = _extract_measurement(data, test_metadata)
                if measurement is not None:
                    current_measurements.append(measurement)
                else:
                    # Check if this was due to invalid flag
                    parm_flg = data.get("PARM_FLG", 0)
                    test_flg = data.get("TEST_FLG", 0)
                    if isinstance(parm_flg, bytes):
                        parm_flg = parm_flg[0] if parm_flg else 0
                    if isinstance(test_flg, bytes):
                        test_flg = test_flg[0] if test_flg else 0
                    if (parm_flg & RESULT_INVALID) or (test_flg & TEST_FLG_REJECT_MASK):
                        invalid_measurements_count += 1
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
                            "stdf_result_format": measurement.get("stdf_result_format"),
                            "stdf_lower_format": measurement.get("stdf_lower_format"),
                            "stdf_upper_format": measurement.get("stdf_upper_format"),
                            "stdf_result_scale": measurement.get("stdf_result_scale"),
                            "stdf_lower_scale": measurement.get("stdf_lower_scale"),
                            "stdf_upper_scale": measurement.get("stdf_upper_scale"),
                            "timestamp": timestamp if timestamp is not None else float(measurement_index),
                            "measurement_index": measurement_index,
                            "part_status": status,
                        }
                    )
                    # Note: Test catalog is now populated separately in _populate_test_catalog_from_ptr()
                    # This ensures all tests appear in catalog regardless of measurement validity
                current_measurements.clear()
                current_serial = None
                current_site = None

    frame = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    stats = {
        "file": source.file_name,
        "path": str(source.path),
        "measurement_count": int(len(rows)),
        "device_count": int(device_sequence),
        "invalid_measurements_filtered": invalid_measurements_count,
    }
    return frame, test_catalog, stats


def _populate_test_catalog_from_ptr(data: Dict[str, Any], cache: Dict[str, _TestMetadata], test_catalog: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    """Populate test catalog from PTR record regardless of measurement validity."""
    test_num = data.get("TEST_NUM")
    test_name_raw = data.get("TEST_TXT")
    if test_num is None and not test_name_raw:
        return

    # Extract test metadata (similar to _extract_measurement but without flag checks)
    key = str(test_num) if test_num is not None else f"name:{test_name_raw or ''}"
    metadata = cache.get(key, _TestMetadata())

    test_name = _coerce_str(test_name_raw) or metadata.test_name
    if test_name:
        metadata.test_name = test_name

    unit_raw = data.get("UNITS")
    unit_value = _coerce_str(unit_raw) or metadata.unit
    if unit_value:
        metadata.unit = unit_value

    # Extract OPT_FLG for limit detection - try both field name variants
    opt_flg = data.get("OPT_FLG", data.get("OPT_FLAG", 0))
    if isinstance(opt_flg, bytes):
        opt_flg = opt_flg[0] if opt_flg else 0

    res_scale = _select_scale(data.get("RES_SCAL"), metadata.scale)
    metadata.result_scale = res_scale

    res_format = _coerce_str(data.get("C_RESFMT"))
    if res_format:
        metadata.result_format = res_format

    # Process low limit with proper OPT_FLAG semantics
    llm_scale = _select_scale(data.get("LLM_SCAL"), res_scale, metadata.scale)
    metadata.low_scale = llm_scale

    low_format = _coerce_str(data.get("C_LLMFMT"))
    if low_format:
        metadata.low_format = low_format

    raw_low_limit = _apply_scale(_coerce_float(data.get("LO_LIMIT")), llm_scale)
    no_low_limit = bool(opt_flg & 0x40)        # Bit 6: No low limit exists for this test
    use_default_low = bool(opt_flg & 0x10)     # Bit 4: Use default from first PTR

    if no_low_limit:
        # Test truly has no lower limit
        metadata.low_limit = None
        metadata.has_low_limit = False
    elif use_default_low:
        # Use default - don't update metadata (keep existing value)
        pass
    elif raw_low_limit is not None:
        # New limit value provided
        metadata.low_limit = raw_low_limit
        metadata.has_low_limit = True
    # If none of the above, leave metadata unchanged (don't clear on ambiguous None)

    # Process high limit with proper OPT_FLAG semantics
    hlm_scale = _select_scale(data.get("HLM_SCAL"), res_scale, metadata.scale)
    metadata.high_scale = hlm_scale

    high_format = _coerce_str(data.get("C_HLMFMT"))
    if high_format:
        metadata.high_format = high_format

    raw_high_limit = _apply_scale(_coerce_float(data.get("HI_LIMIT")), hlm_scale)
    no_high_limit = bool(opt_flg & 0x80)       # Bit 7: No high limit exists for this test
    use_default_high = bool(opt_flg & 0x20)    # Bit 5: Use default from first PTR

    if no_high_limit:
        # Test truly has no upper limit
        metadata.high_limit = None
        metadata.has_high_limit = False
    elif use_default_high:
        # Use default - don't update metadata (keep existing value)
        pass
    elif raw_high_limit is not None:
        # New limit value provided
        metadata.high_limit = raw_high_limit
        metadata.has_high_limit = True
    # If none of the above, leave metadata unchanged (don't clear on ambiguous None)
    
    metadata.scale = res_scale
    cache[key] = metadata
    
    # Add to test catalog
    catalog_name = metadata.test_name or test_name or ""
    catalog_number = str(test_num) if test_num is not None else ""
    catalog_key = (catalog_name, catalog_number)

    if catalog_key not in test_catalog:
        test_catalog[catalog_key] = {
            "unit": _compose_unit(metadata.unit, metadata.scale),
            "stdf_lower": metadata.low_limit if metadata.has_low_limit else None,
            "stdf_upper": metadata.high_limit if metadata.has_high_limit else None,
            "has_stdf_lower": metadata.has_low_limit,
            "has_stdf_upper": metadata.has_high_limit,
            "stdf_result_format": metadata.result_format,
            "stdf_lower_format": metadata.low_format,
            "stdf_upper_format": metadata.high_format,
            "stdf_result_scale": metadata.result_scale,
            "stdf_lower_scale": metadata.low_scale,
            "stdf_upper_scale": metadata.high_scale,
        }
    else:
        # Update existing entry if we have better information
        existing = test_catalog[catalog_key]
        existing.setdefault("has_stdf_lower", existing.get("stdf_lower") is not None)
        existing.setdefault("has_stdf_upper", existing.get("stdf_upper") is not None)
        if existing.get("unit") in ("", None) and metadata.unit:
            existing["unit"] = _compose_unit(metadata.unit, metadata.scale)
        if metadata.result_format:
            existing["stdf_result_format"] = metadata.result_format
        if metadata.low_format:
            existing["stdf_lower_format"] = metadata.low_format
        if metadata.high_format:
            existing["stdf_upper_format"] = metadata.high_format
        if metadata.result_scale is not None:
            existing["stdf_result_scale"] = metadata.result_scale
        if metadata.low_scale is not None:
            existing["stdf_lower_scale"] = metadata.low_scale
        if metadata.high_scale is not None:
            existing["stdf_upper_scale"] = metadata.high_scale

        # Update limits based on current record's intent
        # Allow explicit clears when no_low_limit or no_high_limit flags are set
        if no_low_limit:
            # Bit 6 set: explicitly clear low limit
            existing["stdf_lower"] = None
            existing["has_stdf_lower"] = False
        elif not use_default_low and metadata.has_low_limit and metadata.low_limit is not None:
            # New limit provided (not using default): update
            existing["stdf_lower"] = metadata.low_limit
            existing["has_stdf_lower"] = True
        # If use_default_low, don't modify existing limit

        if no_high_limit:
            # Bit 7 set: explicitly clear high limit
            existing["stdf_upper"] = None
            existing["has_stdf_upper"] = False
        elif not use_default_high and metadata.has_high_limit and metadata.high_limit is not None:
            # New limit provided (not using default): update
            existing["stdf_upper"] = metadata.high_limit
            existing["has_stdf_upper"] = True
        # If use_default_high, don't modify existing limit


def _extract_measurement(data: Dict[str, Any], cache: Dict[str, _TestMetadata]) -> Optional[MeasurementRecord]:
    test_num = data.get("TEST_NUM")
    test_name_raw = data.get("TEST_TXT")
    if test_num is None and not test_name_raw:
        return None

    # Check if measurement is invalid due to test or parameter flags
    test_flg = data.get("TEST_FLG", 0)
    parm_flg = data.get("PARM_FLG", 0)
    # Try both field name variants for OPT_FLG/OPT_FLAG
    opt_flg = data.get("OPT_FLG", data.get("OPT_FLAG", 0))
    
    # Convert bytes to int if needed
    if isinstance(test_flg, bytes):
        test_flg = test_flg[0] if test_flg else 0
    if isinstance(parm_flg, bytes):
        parm_flg = parm_flg[0] if parm_flg else 0
    if isinstance(opt_flg, bytes):
        opt_flg = opt_flg[0] if opt_flg else 0
    
    # STDF specification: PARM_FLG bit 2 (0x04) indicates RESULT_INVALID
    # Skip measurements where the result is flagged as invalid
    if parm_flg & RESULT_INVALID:  # RESULT_INVALID bit
        return None
    
    # STDF specification: TEST_FLG bits indicate test execution problems
    # Skip measurements with invalid, unreliable, not executed, or aborted tests
    if test_flg & TEST_FLG_REJECT_MASK:
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
    metadata.result_scale = res_scale

    res_format = _coerce_str(data.get("C_RESFMT"))
    if res_format:
        metadata.result_format = res_format

    measurement_value = _apply_scale(_coerce_float(data.get("RESULT")), res_scale)

    # Process low limit with proper OPT_FLAG semantics
    llm_scale = _select_scale(data.get("LLM_SCAL"), res_scale, metadata.scale)
    metadata.low_scale = llm_scale

    low_format = _coerce_str(data.get("C_LLMFMT"))
    if low_format:
        metadata.low_format = low_format

    raw_low_limit = _apply_scale(_coerce_float(data.get("LO_LIMIT")), llm_scale)
    no_low_limit = bool(opt_flg & 0x40)        # Bit 6: No low limit exists for this test
    use_default_low = bool(opt_flg & 0x10)     # Bit 4: Use default from first PTR

    if no_low_limit:
        # Test truly has no lower limit
        metadata.low_limit = None
        metadata.has_low_limit = False
    elif use_default_low:
        # Use default - don't update metadata (keep existing value)
        pass
    elif raw_low_limit is not None:
        # New limit value provided
        metadata.low_limit = raw_low_limit
        metadata.has_low_limit = True
    # If none of the above, leave metadata unchanged (don't clear on ambiguous None)
    low_limit = metadata.low_limit if metadata.has_low_limit else None

    # Process high limit with proper OPT_FLAG semantics
    hlm_scale = _select_scale(data.get("HLM_SCAL"), res_scale, metadata.scale)
    raw_high_limit = _apply_scale(_coerce_float(data.get("HI_LIMIT")), hlm_scale)
    no_high_limit = bool(opt_flg & 0x80)       # Bit 7: No high limit exists for this test
    use_default_high = bool(opt_flg & 0x20)    # Bit 5: Use default from first PTR

    if no_high_limit:
        # Test truly has no upper limit
        metadata.high_limit = None
        metadata.has_high_limit = False
    elif use_default_high:
        # Use default - don't update metadata (keep existing value)
        pass
    elif raw_high_limit is not None:
        # New limit value provided
        metadata.high_limit = raw_high_limit
        metadata.has_high_limit = True
    # If none of the above, leave metadata unchanged (don't clear on ambiguous None)
    high_limit = metadata.high_limit if metadata.has_high_limit else None

    metadata.high_scale = hlm_scale
    high_format = _coerce_str(data.get("C_HLMFMT"))
    if high_format:
        metadata.high_format = high_format

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
        "stdf_result_format": metadata.result_format,
        "stdf_lower_format": metadata.low_format,
        "stdf_upper_format": metadata.high_format,
        "stdf_result_scale": metadata.result_scale,
        "stdf_lower_scale": metadata.low_scale,
        "stdf_upper_scale": metadata.high_scale,
    }


def _apply_scale(value: Optional[float], scale: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    if scale in (None, 0):
        return value
    try:
        return value * (10 ** scale)
    except (TypeError, ValueError, OverflowError):
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
    except (TypeError, ValueError):
        return None
    return text or None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
