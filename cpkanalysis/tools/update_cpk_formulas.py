from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from ..postprocess import sheet_utils

HELPER_SHEET = "cpk_helpers"


def _set(cell, formula: str) -> None:
    cell.value = formula


def _quote_sheet(name: str) -> str:
    escaped = name.replace("'", "''")
    return f"'{escaped}'!"


def _require_column(header_map: dict[str, int], *aliases: str) -> int:
    for header in aliases:
        key = sheet_utils.normalize_header(header)
        column = header_map.get(key)
        if column is not None:
            return column
    alias_list = ", ".join(aliases)
    raise ValueError(f"Template sheet missing required column (aliases: {alias_list}).")


def _ensure_column(ws: Worksheet, header_row: int, header_map: dict[str, int], *aliases: str) -> int:
    for header in aliases:
        key = sheet_utils.normalize_header(header)
        column = header_map.get(key)
        if column is not None:
            return column
    header = aliases[0]
    column = ws.max_column + 1
    ws.cell(row=header_row, column=column, value=header)
    header_map[sheet_utils.normalize_header(header)] = column
    return column


def _cell(column_letter: str, row: int) -> str:
    return f"${column_letter}{row}"


def _sheet_cell(sheet_ref: str, column_letter: str, row: int) -> str:
    return f"{sheet_ref}${column_letter}{row}"


def apply_formulas(workbook: Workbook, template_sheet: Union[str, Worksheet]) -> None:
    if isinstance(template_sheet, Worksheet):
        ws = template_sheet
    else:
        if template_sheet not in workbook.sheetnames:
            raise ValueError(f"Sheet {template_sheet!r} not found in workbook.")
        ws = workbook[template_sheet]

    sheet_ref = _quote_sheet(ws.title)
    header_row, header_map = sheet_utils.build_header_map(ws)
    data_start_row = header_row + 1

    mean_col = _require_column(header_map, "MEAN")
    stdev_col = _require_column(header_map, "STDEV")
    median_col = _require_column(header_map, "MEDIAN")
    iqr_col = _require_column(header_map, "IQR")
    ll_spec_col = _require_column(header_map, "LL_SPEC", "Spec Lower", "Spec Lower Limit", "Spec LL")
    ul_spec_col = _require_column(header_map, "UL_SPEC", "Spec Upper", "Spec Upper Limit", "Spec UL")
    ll_ate_col = _require_column(header_map, "LL_ATE", "LL ATE", "Lower ATE", "LL")
    ul_ate_col = _require_column(header_map, "UL_ATE", "UL ATE", "Upper ATE", "UL")
    cpl_col = _require_column(header_map, "CPL")
    cpu_col = _require_column(header_map, "CPU")
    cpk_col = _require_column(header_map, "CPK")
    ll_2cpk_col = _require_column(header_map, "LL_2CPK", "LL 2CPK")
    ul_2cpk_col = _require_column(header_map, "UL_2CPK", "UL 2CPK")
    cpk_2_col = _require_column(header_map, "CPK_2.0", "CPK 2.0")
    ll_3iqr_col = _require_column(header_map, "LL_3IQR", "LL 3IQR")
    ul_3iqr_col = _require_column(header_map, "UL_3IQR", "UL 3IQR")
    cpk_3iqr_col = _require_column(header_map, "CPK_3IQR", "CPK 3IQR")
    ll_prop_col = _require_column(header_map, "LL_PROP", "Proposed LL", "LL Proposed")
    ul_prop_col = _require_column(header_map, "UL_PROP", "Proposed UL", "UL Proposed")
    cpk_prop_col = _require_column(header_map, "CPK_PROP", "CPK Proposed")

    cpk_spec_col = _ensure_column(ws, header_row, header_map, "CPK_SPEC")
    spec_prop_lower_col = _ensure_column(ws, header_row, header_map, "Proposed Spec Lower")
    spec_prop_upper_col = _ensure_column(ws, header_row, header_map, "Proposed Spec Upper")
    spec_prop_cpk_col = _ensure_column(ws, header_row, header_map, "CPK Proposed Spec")

    letters = {
        "mean": get_column_letter(mean_col),
        "stdev": get_column_letter(stdev_col),
        "median": get_column_letter(median_col),
        "iqr": get_column_letter(iqr_col),
        "ll_spec": get_column_letter(ll_spec_col),
        "ul_spec": get_column_letter(ul_spec_col),
        "ll_ate": get_column_letter(ll_ate_col),
        "ul_ate": get_column_letter(ul_ate_col),
        "cpl": get_column_letter(cpl_col),
        "cpu": get_column_letter(cpu_col),
        "cpk": get_column_letter(cpk_col),
        "ll_2cpk": get_column_letter(ll_2cpk_col),
        "ul_2cpk": get_column_letter(ul_2cpk_col),
        "cpk_2": get_column_letter(cpk_2_col),
        "ll_3iqr": get_column_letter(ll_3iqr_col),
        "ul_3iqr": get_column_letter(ul_3iqr_col),
        "cpk_3iqr": get_column_letter(cpk_3iqr_col),
        "ll_prop": get_column_letter(ll_prop_col),
        "ul_prop": get_column_letter(ul_prop_col),
        "cpk_prop": get_column_letter(cpk_prop_col),
        "cpk_spec": get_column_letter(cpk_spec_col),
        "spec_prop_lower": get_column_letter(spec_prop_lower_col),
        "spec_prop_upper": get_column_letter(spec_prop_upper_col),
        "spec_prop_cpk": get_column_letter(spec_prop_cpk_col),
    }

    if HELPER_SHEET in workbook.sheetnames:
        helper_index = workbook.sheetnames.index(HELPER_SHEET)
        del workbook[HELPER_SHEET]
        helper_insert_index = helper_index
    else:
        helper_insert_index = workbook.sheetnames.index(ws.title) + 1
    helper_ws = workbook.create_sheet(HELPER_SHEET, helper_insert_index)
    helper_ws.sheet_state = "hidden"

    helper_ws.cell(row=header_row, column=1, value="CPL_SPEC")
    helper_ws.cell(row=header_row, column=2, value="CPU_SPEC")
    helper_ws.cell(row=header_row, column=3, value="CPK_SPEC")

    max_row = ws.max_row
    for row in range(data_start_row, max_row + 1):
        mean = _cell(letters["mean"], row)
        stdev = _cell(letters["stdev"], row)
        median = _cell(letters["median"], row)
        iqr = _cell(letters["iqr"], row)
        ll_spec = _cell(letters["ll_spec"], row)
        ul_spec = _cell(letters["ul_spec"], row)
        ll_ate = _cell(letters["ll_ate"], row)
        ul_ate = _cell(letters["ul_ate"], row)
        cpl_ref = _cell(letters["cpl"], row)
        cpu_ref = _cell(letters["cpu"], row)
        ll_2 = _cell(letters["ll_2cpk"], row)
        ul_2 = _cell(letters["ul_2cpk"], row)
        ll_3 = _cell(letters["ll_3iqr"], row)
        ul_3 = _cell(letters["ul_3iqr"], row)
        ll_prop = _cell(letters["ll_prop"], row)
        ul_prop = _cell(letters["ul_prop"], row)

        _set(
            ws.cell(row=row, column=cpl_col),
            f"=IF(OR({stdev}<=0,AND({ll_ate}=\"\",{ll_spec}=\"\")),\"\",({mean}-IF({ll_ate}<>\"\",{ll_ate},{ll_spec}))/(3*{stdev}))",
        )
        _set(
            ws.cell(row=row, column=cpu_col),
            f"=IF(OR({stdev}<=0,AND({ul_ate}=\"\",{ul_spec}=\"\")),\"\",(IF({ul_ate}<>\"\",{ul_ate},{ul_spec})-{mean})/(3*{stdev}))",
        )
        _set(
            ws.cell(row=row, column=cpk_col),
            f"=IF(OR({cpl_ref}=\"\",{cpu_ref}=\"\"),\"\",MIN({cpl_ref},{cpu_ref}))",
        )
        _set(
            ws.cell(row=row, column=ll_2cpk_col),
            f"=IF({stdev}>0,{mean}-6*{stdev},\"\")",
        )
        _set(
            ws.cell(row=row, column=ul_2cpk_col),
            f"=IF({stdev}>0,{mean}+6*{stdev},\"\")",
        )
        _set(
            ws.cell(row=row, column=cpk_2_col),
            f"=IF(AND({stdev}>0,{ll_2}<>\"\",{ul_2}<>\"\"),MIN(({mean}-{ll_2})/(3*{stdev}),({ul_2}-{mean})/(3*{stdev})),\"\")",
        )
        _set(
            ws.cell(row=row, column=ll_3iqr_col),
            f"=IF(AND({median}<>\"\",{iqr}<>\"\"),{median}-3*{iqr},\"\")",
        )
        _set(
            ws.cell(row=row, column=ul_3iqr_col),
            f"=IF(AND({median}<>\"\",{iqr}<>\"\"),{median}+3*{iqr},\"\")",
        )
        _set(
            ws.cell(row=row, column=cpk_3iqr_col),
            f"=IF(AND({stdev}>0,{ll_3}<>\"\",{ul_3}<>\"\"),MIN(({mean}-{ll_3})/(3*{stdev}),({ul_3}-{mean})/(3*{stdev})),\"\")",
        )
        _set(
            ws.cell(row=row, column=cpk_prop_col),
            f"=IF(AND({stdev}>0,{ll_prop}<>\"\",{ul_prop}<>\"\"),MIN(({mean}-{ll_prop})/(3*{stdev}),({ul_prop}-{mean})/(3*{stdev})),\"\")",
        )
        spec_prop_lower = _cell(letters["spec_prop_lower"], row)
        spec_prop_upper = _cell(letters["spec_prop_upper"], row)
        _set(
            ws.cell(row=row, column=spec_prop_cpk_col),
            f"=IF(AND({stdev}>0,{spec_prop_lower}<>\"\",{spec_prop_upper}<>\"\"),"
            f"MIN(({mean}-{spec_prop_lower})/(3*{stdev}),({spec_prop_upper}-{mean})/(3*{stdev})),\"\")",
        )

        sheet_mean = _sheet_cell(sheet_ref, letters["mean"], row)
        sheet_stdev = _sheet_cell(sheet_ref, letters["stdev"], row)
        sheet_ll_spec = _sheet_cell(sheet_ref, letters["ll_spec"], row)
        sheet_ul_spec = _sheet_cell(sheet_ref, letters["ul_spec"], row)

        _set(
            helper_ws.cell(row=row, column=1),
            f"=IF(AND({sheet_stdev}>0,{sheet_ll_spec}<>\"\"),({sheet_mean}-{sheet_ll_spec})/(3*{sheet_stdev}),\"\")",
        )
        _set(
            helper_ws.cell(row=row, column=2),
            f"=IF(AND({sheet_stdev}>0,{sheet_ul_spec}<>\"\"),({sheet_ul_spec}-{sheet_mean})/(3*{sheet_stdev}),\"\")",
        )
        _set(
            helper_ws.cell(row=row, column=3),
            f"=IF(OR($A{row}=\"\",$B{row}=\"\"),\"\",MIN($A{row},$B{row}))",
        )

        _set(
            ws.cell(row=row, column=cpk_spec_col),
            f"=IF('{HELPER_SHEET}'!$C{row}=\"\",\"\",'{HELPER_SHEET}'!$C{row})",
        )


def update_workbook(path: Path, template_sheet: str) -> None:
    wb = openpyxl.load_workbook(path)
    apply_formulas(wb, template_sheet)
    wb.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject CPK formulas into the template workbook.")
    parser.add_argument(
        "workbook",
        type=Path,
        help="Path to the template workbook.",
    )
    parser.add_argument(
        "--sheet",
        required=True,
        help="Template sheet name as resolved by the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_workbook(args.workbook.resolve(), template_sheet=args.sheet)


if __name__ == "__main__":
    main()
