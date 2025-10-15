from __future__ import annotations

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cpkanalysis import ingest


def _build_ptr_record(
    *,
    test_num: int = 1,
    test_name: str = "VDD",
    result: float = 1.2,
    lo_limit: float | None = 1.0,
    hi_limit: float | None = 2.0,
    opt_flg: int = 0,
    res_scal: int = 0,
    llm_scal: int = 0,
    hlm_scal: int = 0,
    res_fmt: str | None = None,
    llm_fmt: str | None = None,
    hlm_fmt: str | None = None,
) -> dict:
    return {
        "TEST_NUM": test_num,
        "TEST_TXT": test_name,
        "UNITS": "V",
        "RES_SCAL": res_scal,
        "C_RESFMT": res_fmt,
        "RESULT": result,
        "LLM_SCAL": llm_scal,
        "C_LLMFMT": llm_fmt,
        "LO_LIMIT": lo_limit,
        "HLM_SCAL": hlm_scal,
        "C_HLMFMT": hlm_fmt,
        "HI_LIMIT": hi_limit,
        "OPT_FLG": opt_flg,
        "TEST_FLG": 0,
        "PARM_FLG": 0,
        "TEST_TIM": 0.5,
    }


def test_populate_test_catalog_clears_limits_when_flagged():
    cache: dict[str, ingest._TestMetadata] = {}
    test_catalog: dict[tuple[str, str], dict] = {}

    record_with_limits = _build_ptr_record(lo_limit=1.0, hi_limit=2.0, opt_flg=0)
    ingest._populate_test_catalog_from_ptr(record_with_limits, cache, test_catalog)

    key = ("VDD", "1")
    assert test_catalog[key]["stdf_lower"] == pytest.approx(1.0)
    assert test_catalog[key]["stdf_upper"] == pytest.approx(2.0)
    assert test_catalog[key]["has_stdf_lower"] is True
    assert test_catalog[key]["has_stdf_upper"] is True

    record_without_limits = _build_ptr_record(lo_limit=None, hi_limit=None, opt_flg=0xC0)
    ingest._populate_test_catalog_from_ptr(record_without_limits, cache, test_catalog)

    assert test_catalog[key]["stdf_lower"] is None
    assert test_catalog[key]["stdf_upper"] is None
    assert test_catalog[key]["has_stdf_lower"] is False
    assert test_catalog[key]["has_stdf_upper"] is False


def test_populate_test_catalog_missing_limit_without_flag_clears():
    """Changed behavior: None without explicit flag should preserve existing limits."""
    cache: dict[str, ingest._TestMetadata] = {}
    test_catalog: dict[tuple[str, str], dict] = {}

    ingest._populate_test_catalog_from_ptr(_build_ptr_record(lo_limit=1.0, hi_limit=2.0, opt_flg=0), cache, test_catalog)
    key = ("VDD", "1")
    assert test_catalog[key]["has_stdf_lower"] is True
    assert test_catalog[key]["has_stdf_upper"] is True

    # None without explicit flag (bits 6/7) should preserve existing limits
    ingest._populate_test_catalog_from_ptr(_build_ptr_record(lo_limit=None, hi_limit=None, opt_flg=0), cache, test_catalog)
    assert test_catalog[key]["stdf_lower"] == pytest.approx(1.0)  # Preserved
    assert test_catalog[key]["stdf_upper"] == pytest.approx(2.0)  # Preserved
    assert test_catalog[key]["has_stdf_lower"] is True
    assert test_catalog[key]["has_stdf_upper"] is True


def test_extract_measurement_returns_none_limits_after_flag_clear():
    cache: dict[str, ingest._TestMetadata] = {}

    initial_record = _build_ptr_record(lo_limit=0.5, hi_limit=1.5, opt_flg=0)
    measurement_with_limits = ingest._extract_measurement(initial_record, cache)
    assert measurement_with_limits is not None
    assert measurement_with_limits["low_limit"] == pytest.approx(0.5)

    cleared_record = _build_ptr_record(lo_limit=None, hi_limit=None, opt_flg=0xC0)
    measurement = ingest._extract_measurement(cleared_record, cache)
    assert measurement is not None
    assert measurement["low_limit"] is None
    assert measurement["high_limit"] is None


def test_extract_measurement_missing_limit_without_flag_clears_cache():
    cache: dict[str, ingest._TestMetadata] = {}

    ingest._extract_measurement(_build_ptr_record(lo_limit=0.5, hi_limit=1.5, opt_flg=0), cache)
    follow_up = ingest._extract_measurement(_build_ptr_record(lo_limit=None, hi_limit=None, opt_flg=0), cache)
    assert follow_up is not None
    # Changed behavior: None without explicit flag should preserve existing limits
    assert follow_up["low_limit"] == pytest.approx(0.5)
    assert follow_up["high_limit"] == pytest.approx(1.5)


def test_opt_flag_bit4_preserves_default_low_limit():
    """Test that OPT_FLAG bit 4 (0x10) preserves default low limit."""
    cache: dict[str, ingest._TestMetadata] = {}
    test_catalog: dict[tuple[str, str], dict] = {}

    # First PTR establishes default limits
    ingest._populate_test_catalog_from_ptr(_build_ptr_record(lo_limit=1.0, hi_limit=2.0, opt_flg=0), cache, test_catalog)
    key = ("VDD", "1")
    assert test_catalog[key]["stdf_lower"] == pytest.approx(1.0)

    # Second PTR with bit 4 set should preserve default low limit
    ingest._populate_test_catalog_from_ptr(_build_ptr_record(lo_limit=999.0, hi_limit=2.0, opt_flg=0x10), cache, test_catalog)
    assert test_catalog[key]["stdf_lower"] == pytest.approx(1.0)  # Should still be 1.0, not 999.0


def test_opt_flag_bit5_preserves_default_high_limit():
    """Test that OPT_FLAG bit 5 (0x20) preserves default high limit."""
    cache: dict[str, ingest._TestMetadata] = {}
    test_catalog: dict[tuple[str, str], dict] = {}

    # First PTR establishes default limits
    ingest._populate_test_catalog_from_ptr(_build_ptr_record(lo_limit=1.0, hi_limit=2.0, opt_flg=0), cache, test_catalog)
    key = ("VDD", "1")
    assert test_catalog[key]["stdf_upper"] == pytest.approx(2.0)

    # Second PTR with bit 5 set should preserve default high limit
    ingest._populate_test_catalog_from_ptr(_build_ptr_record(lo_limit=1.0, hi_limit=999.0, opt_flg=0x20), cache, test_catalog)
    assert test_catalog[key]["stdf_upper"] == pytest.approx(2.0)  # Should still be 2.0, not 999.0


def test_extract_measurement_preserves_defaults_with_bit4_bit5():
    """Test that _extract_measurement preserves defaults when bits 4 & 5 are set."""
    cache: dict[str, ingest._TestMetadata] = {}

    # First measurement establishes defaults
    first = ingest._extract_measurement(_build_ptr_record(lo_limit=0.5, hi_limit=1.5, opt_flg=0), cache)
    assert first["low_limit"] == pytest.approx(0.5)
    assert first["high_limit"] == pytest.approx(1.5)

    # Second measurement with bits 4 & 5 should use defaults
    second = ingest._extract_measurement(_build_ptr_record(lo_limit=999.0, hi_limit=888.0, opt_flg=0x30), cache)
    assert second["low_limit"] == pytest.approx(0.5)  # Should use default, not 999.0
    assert second["high_limit"] == pytest.approx(1.5)  # Should use default, not 888.0


def test_format_fields_propagate_to_measurement_and_catalog():
    cache: dict[str, ingest._TestMetadata] = {}
    test_catalog: dict[tuple[str, str], dict] = {}

    record = _build_ptr_record(
        res_scal=-2,
        llm_scal=-2,
        hlm_scal=-2,
        res_fmt="%8.3f",
        llm_fmt="%8.1f",
        hlm_fmt="%8.1f",
    )
    ingest._populate_test_catalog_from_ptr(record, cache, test_catalog)
    measurement = ingest._extract_measurement(record, cache)

    assert measurement is not None
    assert measurement["stdf_result_format"] == "%8.3f"
    assert measurement["stdf_lower_format"] == "%8.1f"
    assert measurement["stdf_upper_format"] == "%8.1f"
    assert measurement["stdf_result_scale"] == -2
    assert measurement["stdf_lower_scale"] == -2
    assert measurement["stdf_upper_scale"] == -2

    key = ("VDD", "1")
    catalog_entry = test_catalog[key]
    assert catalog_entry["stdf_result_format"] == "%8.3f"
    assert catalog_entry["stdf_lower_format"] == "%8.1f"
    assert catalog_entry["stdf_upper_format"] == "%8.1f"
    assert catalog_entry["stdf_result_scale"] == -2
    assert catalog_entry["stdf_lower_scale"] == -2
    assert catalog_entry["stdf_upper_scale"] == -2
