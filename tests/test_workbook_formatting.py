from __future__ import annotations

import pytest

from cpkanalysis.workbook_builder import (
    _excel_number_format,
    _fixed_decimal_format,
    _row_number_format,
    set_fallback_decimals,
)


@pytest.fixture(autouse=True)
def reset_fallback_decimals() -> None:
    set_fallback_decimals(None)
    yield
    set_fallback_decimals(None)


def test_excel_number_format_from_printf_tokens() -> None:
    assert _excel_number_format("%8.3f", None) == "0.000"
    assert _excel_number_format("%10.2E", None) == "0.00E+00"
    assert _excel_number_format("%6.3g", None) == "0.###"
    assert _excel_number_format("%5d", None) == "0"


def test_excel_number_format_uses_scale_when_missing_format() -> None:
    assert _excel_number_format(None, -3) == "0.###"
    assert _excel_number_format(None, 0) == "0"


def test_fallback_decimals_override_applies_to_missing_hints() -> None:
    set_fallback_decimals(2)
    assert _excel_number_format(None, None) == "0.##"
    set_fallback_decimals(5)
    assert _excel_number_format(None, None) == "0.#####"


def test_row_number_format_prefers_specific_field() -> None:
    row = {
        "stdf_lower_format": "%7.1f",
        "stdf_lower_scale": None,
        "stdf_result_scale": -2,
    }
    assert _row_number_format(
        row,
        format_fields=("stdf_lower_format", "stdf_result_format"),
        scale_fields=("stdf_lower_scale", "stdf_result_scale"),
    ) == "0.0"


def test_fixed_decimal_format_matches_current_fallback() -> None:
    set_fallback_decimals(4)
    assert _fixed_decimal_format(4) == "0.0000"
    set_fallback_decimals(0)
    assert _fixed_decimal_format(0) == "0"
