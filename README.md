# mergeXlsx

Python utility for merging worksheets from multiple Excel workbooks into a single file while preserving formatting.

## Requirements
- Python 3.11+
- openpyxl
- matplotlib (for addCharts.py)

## Usage
Run the script from the directory that holds the source workbooks.

```bash
python mergeXlsx.py -o F104547_Merge.xlsx
```

### What the script does
- Copies every visible worksheet (unless `--include-hidden` is set) into a single workbook while keeping basic formatting.
- Builds a **Summary** sheet that lists every merged worksheet with:
  - Hyperlinks directly to the sheet and back to the source file on disk
  - Lot/Event/Int metadata obtained from the original filename (`Original_Lot_Event_Int.xlsx`)
  - Counts of passing and failing units derived from the sheet's "Unit SN" table (retests collapse to a single final outcome per serial number).
- Adds one or more **Measurements** sheets that consolidate each unit's detailed results with columns `Lot`, `Event`, `Int`, `Source Worksheet`, `SN`, `PASS/FAIL`, `Test Number`, `Test Name`, `Test Unit`, `Low Limit`, `High Limit`, and `Measurement`. Serial numbers that appear in both FAIL UNITS and PASS UNITS are treated as passes and only their passing measurements are exported; serial numbers that never pass contribute only their last recorded measurements. Fail-table readings that store numeric values with an `F` prefix (for example `F-0.808`) are normalized so the numeric portion is exported. Each sheet stays under Excel's 1,048,576-row limit.
- Skips any `.xlsx` file that matches the requested output filename or starts with Excel's `~$` temp prefix.

### Adding Histogram Charts
Run `python addCharts.py` after the merge completes to append a **Charts** worksheet. The script groups rows in the Measurements tables by Test Name, renders a histogram that respects the recorded low/high limits, and embeds one chart per test sized approximately 10" x 5" with vertical limit markers. Each execution widens the X axis slightly so the limit lines stay visible even when they coincide with the min/max data values. Use `--output <file>` to save the charts to a separate workbook, or `--charts-per-row` to adjust the layout.

### Helpful options
- `--values-only` to skip formatting and speed up the merge.
- `--directory <path>` to merge all `.xlsx` files found in another folder.
- Provide explicit file paths after the flags to merge just those workbooks.

## Maintainer Notes
- Keep `load_workbook(..., read_only=False)` so formatting survives copy; switching to read-only drops styles and merged ranges.
- `_normalize_serial_value` and `create_measurements_sheet` coerce Serial/SN values to numbers and set the table column format to `0`; update both if the Measurements layout changes.
- After code changes run `python -m py_compile mergeXlsx.py` and spot-check `python mergeXlsx.py` against sample workbooks; the latter takes ~3 minutes on the current data set.
- The Measurements exporter chunks data to stay under Excel's 1,048,576-row ceiling; adjust `data_capacity` if you ever add headers or padding rows.
- Preserve the hyperlink logic in `create_summary_sheet`; it ties the Summary view back to both merged sheets and source files.
