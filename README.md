# mergeXlsx

Python utility for merging worksheets from multiple Excel workbooks into a single file while preserving formatting.

## Requirements
- Python 3.11+
- openpyxl

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
- Adds one or more **Measurements** sheets that consolidate every passing unit's detailed results with columns `Lot`, `Event`, `Int`, `SN`, `Test Number`, `Test Name`, `Test Unit`, `Low Limit`, `High Limit`, and `Measurement`. Failing serial numbers are excluded and each sheet stays under Excel's 1,048,576-row limit.
- Skips any `.xlsx` file that matches the requested output filename or starts with Excel's `~$` temp prefix.

### Helpful options
- `--values-only` to skip formatting and speed up the merge.
- `--directory <path>` to merge all `.xlsx` files found in another folder.
- Provide explicit file paths after the flags to merge just those workbooks.

