# mergeXlsx

Python utility for merging worksheets from multiple Excel workbooks into a single file while preserving formatting.

## Requirements
- Python 3.11+
- openpyxl
- matplotlib (for addHistoCharts.py / addBoxCharts.py)

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
Run `python addHistoCharts.py` after the merge completes to append histogram worksheets immediately after the **Measurements** sheets. The script creates one sheet per Event (named `Histogram_<EVENT>`), grouping rows in the Measurements tables by Test Name and Lot so each chart shows a single Test/Lot pair. Charts are laid out with Test Names on rows and Lots on columns; the X axis is padded slightly so low/high limit markers stay visible even when they coincide with the data extremes.

When the Measurements data includes an `INT`/Interval column, each chart overlays color-coded histograms for every interval value and adds a legend showing the subgroup labels. `INT Prescreen` is always displayed first, numeric INT labels follow from smallest to largest so time-based stress steps read left to right, and any remaining text labels (including `INT Missing`) are appended afterward. Rows without an interval entry are collected into an `INT Missing` series, and the combined legend is positioned outside the plot area on the right so the bins remain unobstructed.

A bold annotation in the upper-right corner reports the mean shift between the smallest and largest INT groups (mean of max INT minus mean of min INT). Units are appended when available.

Each run also creates a **ShiftSummary** sheet that lists the Event, Test Name/Test Number/Lot combination, the smallest and largest INT labels, their means, the resulting mean shift, and the low/high limits derived from the largest INT dataset for that lot.

ShiftSummary columns:
- `Event` – copied from the Measurements table (fallback `Unknown Event` when no Event column or empty cell exists).
- `Test Name` / `Test Number` / `Unit` – carried forward from the Measurements data.
- `Lot` – the lot name hosting the histogram.
- `Min INT` / `Max INT` – labels for the smallest and largest INT groups contributing data.
- `Min INT Mean` / `Max INT Mean` – mean of the measurements inside those INT groups.
- `Mean Shift` – `Max INT Mean` minus `Min INT Mean` (unit-aware when provided).
- `Low Limit (Max INT)` / `High Limit (Max INT)` – limits associated with the largest INT data set (falls back to the lot-level limits when the INT-specific limits are missing).

Event-specific histogram sheets are named `Histogram_<EVENT>` with invalid Excel characters replaced by underscores and truncated to Excel's 31-character limit.

Use `--output <file>` to save the charts to a separate workbook, or `--max-lots <N>` to limit the number of Lot columns rendered.

### Adding Boxplot Charts
Run `python addBoxCharts.py` to produce the same Event/Test/Lot layout but with INT-grouped boxplots instead of histograms. Sheets are named `Boxplot_<EVENT>` and share the ShiftSummary sheet described above. Each box displays the INT subgroups for a lot, uses colour coding consistent with the histogram view, and carries the same mean-shift annotation and low/high limit markers (rendered as horizontal lines). The INT ordering mirrors the histogram view (`INT Prescreen`, then ascending numeric INT labels, followed by any remaining text labels), and the legend is rendered outside the plot area on the right so the callout and boxes stay clear.

The command-line arguments mirror `addHistoCharts.py`; you can reuse `--output` and `--max-lots` in the same way.
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
