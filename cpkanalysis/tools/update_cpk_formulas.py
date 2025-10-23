from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl


HEADER_ROW = 10
DATA_START_ROW = HEADER_ROW + 1
HELPER_SHEET = "cpk_helpers"


def _set(cell, formula: str) -> None:
    cell.value = formula


def _quote_sheet(name: str) -> str:
    escaped = name.replace("'", "''")
    return f"'{escaped}'!"


def update_workbook(path: Path, template_sheet: str) -> None:
    wb = openpyxl.load_workbook(path)
    if template_sheet not in wb.sheetnames:
        raise ValueError(f"Sheet {template_sheet!r} not found in {path}")
    ws = wb[template_sheet]
    sheet_ref = _quote_sheet(ws.title)

    if HELPER_SHEET in wb.sheetnames:
        helper_index = wb.sheetnames.index(HELPER_SHEET)
        del wb[HELPER_SHEET]
        helper_insert_index = helper_index
    else:
        helper_insert_index = wb.sheetnames.index(ws.title) + 1
    helper_ws = wb.create_sheet(HELPER_SHEET, helper_insert_index)
    helper_ws.sheet_state = "hidden"

    helper_ws.cell(row=HEADER_ROW, column=1, value="CPL_SPEC")
    helper_ws.cell(row=HEADER_ROW, column=2, value="CPU_SPEC")
    helper_ws.cell(row=HEADER_ROW, column=3, value="CPK_SPEC")

    cpk_spec_header = ws.cell(row=HEADER_ROW, column=31)
    if (cpk_spec_header.value or "").strip() == "":
        cpk_spec_header.value = "CPK_SPEC"

    max_row = ws.max_row
    for row in range(DATA_START_ROW, max_row + 1):
        _set(
            ws.cell(row=row, column=14),
            f"=IF(OR($L{row}<=0,AND($G{row}=\"\",$E{row}=\"\")),\"\",($J{row}-IF($G{row}<>\"\",$G{row},$E{row}))/(3*$L{row}))",
        )
        _set(
            ws.cell(row=row, column=15),
            f"=IF(OR($L{row}<=0,AND($H{row}=\"\",$F{row}=\"\")),\"\",(IF($H{row}<>\"\",$H{row},$F{row})-$J{row})/(3*$L{row}))",
        )
        _set(
            ws.cell(row=row, column=16),
            f"=IF(OR($N{row}=\"\",$O{row}=\"\"),\"\",MIN($N{row},$O{row}))",
        )
        _set(ws.cell(row=row, column=18), f"=IF($L{row}>0,$J{row}-6*$L{row},\"\")")
        _set(ws.cell(row=row, column=19), f"=IF($L{row}>0,$J{row}+6*$L{row},\"\")")
        _set(
            ws.cell(row=row, column=20),
            f"=IF(AND($L{row}>0,$R{row}<>\"\",$S{row}<>\"\"),MIN(($J{row}-$R{row})/(3*$L{row}),($S{row}-$J{row})/(3*$L{row})),\"\")",
        )
        _set(
            ws.cell(row=row, column=22),
            f"=IF(AND($K{row}<>\"\",$M{row}<>\"\"),$K{row}-3*$M{row},\"\")",
        )
        _set(
            ws.cell(row=row, column=23),
            f"=IF(AND($K{row}<>\"\",$M{row}<>\"\"),$K{row}+3*$M{row},\"\")",
        )
        _set(
            ws.cell(row=row, column=24),
            f"=IF(AND($L{row}>0,$V{row}<>\"\",$W{row}<>\"\"),MIN(($J{row}-$V{row})/(3*$L{row}),($W{row}-$J{row})/(3*$L{row})),\"\")",
        )
        _set(
            ws.cell(row=row, column=29),
            f"=IF(AND($L{row}>0,$AA{row}<>\"\",$AB{row}<>\"\"),MIN(($J{row}-$AA{row})/(3*$L{row}),($AB{row}-$J{row})/(3*$L{row})),\"\")",
        )

        _set(
            helper_ws.cell(row=row, column=1),
            f"=IF(AND({sheet_ref}$L{row}>0,{sheet_ref}$E{row}<>\"\"),({sheet_ref}$J{row}-{sheet_ref}$E{row})/(3*{sheet_ref}$L{row}),\"\")",
        )
        _set(
            helper_ws.cell(row=row, column=2),
            f"=IF(AND({sheet_ref}$L{row}>0,{sheet_ref}$F{row}<>\"\"),({sheet_ref}$F{row}-{sheet_ref}$J{row})/(3*{sheet_ref}$L{row}),\"\")",
        )
        _set(
            helper_ws.cell(row=row, column=3),
            f"=IF(OR($A{row}=\"\",$B{row}=\"\"),\"\",MIN($A{row},$B{row}))",
        )

        _set(
            ws.cell(row=row, column=31),
            f"=IF('{HELPER_SHEET}'!$C{row}=\"\",\"\",'{HELPER_SHEET}'!$C{row})",
        )

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
