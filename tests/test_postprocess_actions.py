from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis.postprocess import actions, sheet_utils
from cpkanalysis.postprocess.context import PostProcessContext
from cpkanalysis.models import AnalysisInputs


class DummyIO:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def print(self, *args, **kwargs) -> None:  # pragma: no cover
        return None

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def prompt_choice(self, prompt: str, options, show_options: bool = True) -> int:  # pragma: no cover
        return 0

    def prompt(self, prompt: str, default: str | None = None) -> str:
        return default or ""

    def confirm(self, prompt: str, default: bool = True) -> bool:  # pragma: no cover
        return default


def _build_workbook(path: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)

    summary = wb.create_sheet("Summary")
    summary.append(
        ["File", "Test Name", "Test Number", "Unit", "MEAN", "STDEV", "LL_2CPK", "UL_2CPK", "LL_3IQR", "UL_3IQR", "CPK"]
    )
    summary.append(["lot1", "TestA", "1", "V", 1.0, 0.2, 0.7, 1.3, 0.6, 1.4, 1.5])

    limits = wb.create_sheet("Test List and Limits")
    limits.append(
        [
            "Test Name",
            "Test Number",
            "Spec Lower Limit",
            "Spec Upper Limit",
            "User What-If Lower Limit",
            "User What-If Upper Limit",
        ]
    )
    limits.append(["TestA", "1", None, None, None, None])

    template = wb.create_sheet("Template")
    template.append(
        [
            "Test Name",
            "Test Number",
            "Spec Lower",
            "Spec Upper",
            "What-if Lower",
            "What-if Upper",
            "LL_PROP",
            "UL_PROP",
            "CPK_PROP",
            "%YLD LOSS_PROP",
        ]
    )
    template.append(["TestA", "1", None, None, None, None, None, None, None, None])

    measurements = wb.create_sheet("Measurements")
    measurements.append(["Test Name", "Test Number", "Value"])
    measurements.append(["TestA", "1", 0.2])
    measurements.append(["TestA", "1", 1.8])
    measurements.append(["TestA", "1", 1.0])

    wb.save(path)


def _load_context(workbook_path: Path, tmp_path: Path) -> PostProcessContext:
    inputs = AnalysisInputs(
        sources=[],
        output=tmp_path / "analysis.xlsx",
        template_sheet="Template",
        generate_histogram=False,
        generate_cdf=False,
        generate_time_series=False,
    )
    metadata_path = tmp_path / "meta.json"
    metadata_path.write_text("{}", encoding="utf-8")
    return PostProcessContext(inputs, workbook_path, metadata_path, metadata={})


def test_apply_spec_limits_updates_template_and_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    _build_workbook(workbook_path)
    context = _load_context(workbook_path, tmp_path)
    io = DummyIO()

    refreshed: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(
        actions.charts,
        "refresh_tests",
        lambda ctx, tests, **kwargs: refreshed.setdefault("tests", list(tests)),
    )

    params = {"scope": "single", "test_key": "lot1|TestA|1", "target_cpk": 1.0}
    result = actions.apply_spec_limits(context, io, params)

    template_ws = context.template_sheet()
    header_row, headers = sheet_utils.build_header_map(template_ws)
    spec_lower_col = headers[sheet_utils.normalize_header("Spec Lower")]
    spec_upper_col = headers[sheet_utils.normalize_header("Spec Upper")]
    what_lower_col = headers[sheet_utils.normalize_header("What-if Lower")]
    what_upper_col = headers[sheet_utils.normalize_header("What-if Upper")]
    row_idx = header_row + 1

    expected_lower = pytest.approx(1.0 - (3.0 * 1.0 * 0.2))
    expected_upper = pytest.approx(1.0 + (3.0 * 1.0 * 0.2))
    assert template_ws.cell(row=row_idx, column=spec_lower_col).value == expected_lower
    assert template_ws.cell(row=row_idx, column=spec_upper_col).value == expected_upper
    assert template_ws.cell(row=row_idx, column=what_lower_col).value == expected_lower
    assert template_ws.cell(row=row_idx, column=what_upper_col).value == expected_upper

    limits_ws = context.workbook()["Test List and Limits"]
    limits_header_row, limits_headers = sheet_utils.build_header_map(limits_ws)
    spec_ll = limits_headers[sheet_utils.normalize_header("Spec Lower Limit")]
    spec_ul = limits_headers[sheet_utils.normalize_header("Spec Upper Limit")]
    what_ll = limits_headers[sheet_utils.normalize_header("User What-If Lower Limit")]
    what_ul = limits_headers[sheet_utils.normalize_header("User What-If Upper Limit")]
    data_row = limits_header_row + 1

    assert limits_ws.cell(row=data_row, column=spec_ll).value == expected_lower
    assert limits_ws.cell(row=data_row, column=spec_ul).value == expected_upper
    assert limits_ws.cell(row=data_row, column=what_ll).value == expected_lower
    assert limits_ws.cell(row=data_row, column=what_ul).value == expected_upper

    assert refreshed["tests"]
    assert result["summary"] == "Applied Spec/What-If limits for 1 test(s)."
    assert result["audit"]["parameters"]["target_cpk"] == 1.0


def test_apply_spec_limits_missing_columns_cancels(tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    _build_workbook(workbook_path)
    wb = load_workbook(workbook_path)
    template = wb["Template"]
    template.delete_cols(3, 6)  # remove spec/what-if and proposal columns
    template.cell(row=1, column=3, value="Other")
    wb.save(workbook_path)
    wb.close()

    context = _load_context(workbook_path, tmp_path)
    io = DummyIO()

    with pytest.raises(actions.ActionCancelled):
        actions.apply_spec_limits(context, io, {"scope": "all", "target_cpk": 1.0})


def test_calculate_proposed_limits_populates_proposal_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    _build_workbook(workbook_path)
    context = _load_context(workbook_path, tmp_path)
    io = DummyIO()

    monkeypatch.setattr(
        actions.charts,
        "refresh_tests",
        lambda ctx, tests, include_spec=True, include_proposed=True: None,
    )

    result = actions.calculate_proposed_limits(context, io, {"scope": "all", "target_cpk": 1.0})

    template_ws = context.template_sheet()
    header_row, headers = sheet_utils.build_header_map(template_ws)
    ll_prop = headers[sheet_utils.normalize_header("LL_PROP")]
    ul_prop = headers[sheet_utils.normalize_header("UL_PROP")]
    cpk_prop = headers[sheet_utils.normalize_header("CPK_PROP")]
    yld_prop = headers[sheet_utils.normalize_header("%YLD LOSS_PROP")]
    row_idx = header_row + 1

    width = 3.0 * 1.0 * 0.2
    expected_ll = pytest.approx(1.0 - width)
    expected_ul = pytest.approx(1.0 + width)

    assert template_ws.cell(row=row_idx, column=ll_prop).value == expected_ll
    assert template_ws.cell(row=row_idx, column=ul_prop).value == expected_ul
    assert template_ws.cell(row=row_idx, column=cpk_prop).value == pytest.approx(1.0)
    assert template_ws.cell(row=row_idx, column=yld_prop).value == pytest.approx(2 / 3, rel=1e-3)

    assert result["summary"] == "Updated proposed limits for 1 test(s). Recomputed metrics for 1 test(s)."

    audit_params = result["audit"]["parameters"]
    assert audit_params["target_cpk"] == pytest.approx(1.0)
    assert len(audit_params["per_test"]) == 1
    per_test = audit_params["per_test"][0]
    assert per_test["lower_origin"] == "computed"
    assert per_test["upper_origin"] == "computed"
    assert per_test["cpk_updated"] is True
    assert per_test["yield_updated"] is True

    proposal_state = context.metadata["post_processing_state"]["proposed_limits"]
    entry = proposal_state["lot1|TestA|1"]
    assert entry["ll"] == pytest.approx(expected_ll)
    assert entry["ul"] == pytest.approx(expected_ul)


def test_calculate_proposed_limits_respects_user_proposals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    _build_workbook(workbook_path)
    context = _load_context(workbook_path, tmp_path)
    io = DummyIO()

    monkeypatch.setattr(
        actions.charts,
        "refresh_tests",
        lambda ctx, tests, include_spec=True, include_proposed=True: None,
    )

    descriptor_key = "lot1|TestA|1"
    template_ws = context.template_sheet()
    header_row, headers = sheet_utils.build_header_map(template_ws)
    ll_prop = headers[sheet_utils.normalize_header("LL_PROP")]
    ul_prop = headers[sheet_utils.normalize_header("UL_PROP")]
    cpk_prop = headers[sheet_utils.normalize_header("CPK_PROP")]
    yld_prop = headers[sheet_utils.normalize_header("%YLD LOSS_PROP")]
    data_row = header_row + 1

    user_lower = 0.5
    user_upper = 1.6
    template_ws.cell(row=data_row, column=ll_prop, value=user_lower)
    template_ws.cell(row=data_row, column=ul_prop, value=user_upper)
    template_ws.cell(row=data_row, column=cpk_prop, value=None)
    template_ws.cell(row=data_row, column=yld_prop, value=None)

    context.metadata["post_processing_state"] = {
        "proposed_limits": {descriptor_key: {"ll": 0.7, "ul": 1.3, "timestamp": "prev"}},
    }

    result = actions.calculate_proposed_limits(
        context,
        io,
        {"scope": "single", "test_key": descriptor_key, "target_cpk": 1.2},
    )

    assert template_ws.cell(row=data_row, column=ll_prop).value == pytest.approx(user_lower)
    assert template_ws.cell(row=data_row, column=ul_prop).value == pytest.approx(user_upper)

    cpk_value = template_ws.cell(row=data_row, column=cpk_prop).value
    yld_value = template_ws.cell(row=data_row, column=yld_prop).value
    assert cpk_value == pytest.approx(0.833333, rel=1e-3)
    assert yld_value == pytest.approx(2 / 3, rel=1e-3)

    per_test = result["audit"]["parameters"]["per_test"][0]
    assert per_test["lower_origin"] == "user"
    assert per_test["upper_origin"] == "user"
    assert per_test["cpk_updated"] is True
    assert per_test["yield_updated"] is True

    state = context.metadata["post_processing_state"]["proposed_limits"][descriptor_key]
    assert state["ll"] == pytest.approx(user_lower)
    assert state["ul"] == pytest.approx(user_upper)


def test_calculate_proposed_limits_computes_blank_side(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    _build_workbook(workbook_path)
    context = _load_context(workbook_path, tmp_path)
    io = DummyIO()

    monkeypatch.setattr(
        actions.charts,
        "refresh_tests",
        lambda ctx, tests, include_spec=True, include_proposed=True: None,
    )

    descriptor_key = "lot1|TestA|1"
    context.metadata["post_processing_state"] = {
        "proposed_limits": {descriptor_key: {"ll": 0.4, "ul": 1.5, "timestamp": "prev"}},
    }

    template_ws = context.template_sheet()
    header_row, headers = sheet_utils.build_header_map(template_ws)
    ll_prop = headers[sheet_utils.normalize_header("LL_PROP")]
    ul_prop = headers[sheet_utils.normalize_header("UL_PROP")]
    cpk_prop = headers[sheet_utils.normalize_header("CPK_PROP")]
    yld_prop = headers[sheet_utils.normalize_header("%YLD LOSS_PROP")]
    data_row = header_row + 1

    template_ws.cell(row=data_row, column=ll_prop, value=0.4)
    template_ws.cell(row=data_row, column=ul_prop, value=None)
    template_ws.cell(row=data_row, column=cpk_prop, value=None)
    template_ws.cell(row=data_row, column=yld_prop, value=None)

    result = actions.calculate_proposed_limits(
        context,
        io,
        {"scope": "single", "test_key": descriptor_key, "target_cpk": 1.0},
    )

    assert template_ws.cell(row=data_row, column=ll_prop).value == pytest.approx(0.4)
    assert template_ws.cell(row=data_row, column=ul_prop).value == pytest.approx(1.6)

    per_test = result["audit"]["parameters"]["per_test"][0]
    assert per_test["lower_origin"] == "unchanged"
    assert per_test["upper_origin"] == "computed"


def test_calculate_proposed_limits_skips_when_no_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    _build_workbook(workbook_path)
    context = _load_context(workbook_path, tmp_path)
    io = DummyIO()

    monkeypatch.setattr(
        actions.charts,
        "refresh_tests",
        lambda ctx, tests, include_spec=True, include_proposed=True: None,
    )

    descriptor_key = "lot1|TestA|1"
    context.metadata["post_processing_state"] = {
        "proposed_limits": {descriptor_key: {"ll": 0.7, "ul": 1.3, "timestamp": "prev"}},
    }

    template_ws = context.template_sheet()
    header_row, headers = sheet_utils.build_header_map(template_ws)
    ll_prop = headers[sheet_utils.normalize_header("LL_PROP")]
    ul_prop = headers[sheet_utils.normalize_header("UL_PROP")]
    cpk_prop = headers[sheet_utils.normalize_header("CPK_PROP")]
    yld_prop = headers[sheet_utils.normalize_header("%YLD LOSS_PROP")]
    data_row = header_row + 1
    template_ws.cell(row=data_row, column=ll_prop, value=0.7)
    template_ws.cell(row=data_row, column=ul_prop, value=1.3)
    template_ws.cell(row=data_row, column=cpk_prop, value=0.9)
    template_ws.cell(row=data_row, column=yld_prop, value=0.05)

    with pytest.raises(actions.ActionCancelled, match="No tests updated."):
        actions.calculate_proposed_limits(
            context,
            io,
            {"scope": "single", "test_key": descriptor_key, "target_cpk": 1.0},
        )


def test_calculate_proposed_limits_recomputes_metrics_when_blank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    _build_workbook(workbook_path)
    context = _load_context(workbook_path, tmp_path)
    io = DummyIO()

    monkeypatch.setattr(
        actions.charts,
        "refresh_tests",
        lambda ctx, tests, include_spec=True, include_proposed=True: None,
    )

    descriptor_key = "lot1|TestA|1"
    context.metadata["post_processing_state"] = {
        "proposed_limits": {descriptor_key: {"ll": 0.7, "ul": 1.3, "timestamp": "prev"}},
    }

    template_ws = context.template_sheet()
    header_row, headers = sheet_utils.build_header_map(template_ws)
    ll_prop = headers[sheet_utils.normalize_header("LL_PROP")]
    ul_prop = headers[sheet_utils.normalize_header("UL_PROP")]
    cpk_prop = headers[sheet_utils.normalize_header("CPK_PROP")]
    yld_prop = headers[sheet_utils.normalize_header("%YLD LOSS_PROP")]
    data_row = header_row + 1
    template_ws.cell(row=data_row, column=ll_prop, value=0.7)
    template_ws.cell(row=data_row, column=ul_prop, value=1.3)
    template_ws.cell(row=data_row, column=cpk_prop, value=None)
    template_ws.cell(row=data_row, column=yld_prop, value=None)

    result = actions.calculate_proposed_limits(
        context,
        io,
        {"scope": "single", "test_key": descriptor_key, "target_cpk": 1.0},
    )

    assert template_ws.cell(row=data_row, column=cpk_prop).value == pytest.approx(0.5)
    assert template_ws.cell(row=data_row, column=yld_prop).value == pytest.approx(2 / 3, rel=1e-3)

    per_test = result["audit"]["parameters"]["per_test"][0]
    assert per_test["lower_origin"] == "unchanged"
    assert per_test["upper_origin"] == "unchanged"
    assert per_test["cpk_updated"] is True
    assert per_test["yield_updated"] is True
