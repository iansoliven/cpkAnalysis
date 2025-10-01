# Shift Analysis Toolkit

This repository packages a command-line workflow for building shift-analysis workbooks from production test data. The workflow reuses the historical Excel merge scripts, adds STDF ingestion through the bundled `istdf` reader, and layers in statistics plus chart generation.

## Installation

1. Use Python 3.11 or newer.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. (Optional) Initialise the `Submodules/istdf` submodule if you have not already cloned it.

Sample STDF and XLSX files live under `Sample/` for smoke testing; they are ignored by Git.

## Command Line Usage

Invoke the orchestrator with `python -m shiftanalysis.cli <subcommand> [options]`.

Subcommands:

- `readdir [DIRECTORY]` – scan the working directory (or the provided path) for files named `XXX_LOT_EVENT_INTERVAL.(xlsx|stdf)`, display the detected metadata, and persist it to JSON.
- `convert` – read the metadata JSON and emit a `DataWorkBook.xlsx` containing the summary table plus a flattened `Measurements` table.
- `calc-shift` – append a `ShiftSummary` sheet describing min/max interval statistics, mean/median values, and standard-deviation deltas.
- `plot` – generate per-event boxplot and histogram sheets using the measurements table (requires `matplotlib`/`numpy`). Use `--BoxChartOnly` or `--HistoChartOnly` to limit the output.
- `auto [DIRECTORY]` – run the full sequence (`readdir` → `convert` → `calc-shift` → `plot`).

The orchestrator stores file metadata in `shift_sources.json` by default and writes the consolidated workbook to `DataWorkBook.xlsx`. Override these paths with `--metadata` and `--output`.

### Typical Run

```bash
python -m shiftanalysis.cli auto Sample --assume-yes --values-only
```

This command scans `Sample/`, skips the confirmation prompt, generates the workbook without copying Excel formatting, calculates shift statistics, and renders both chart families.

### Stage Details

| Stage          | Output                                                                           |
|----------------|-----------------------------------------------------------------------------------|
| ReadDirectory  | Metadata JSON (`path`, `lot`, `event`, `interval`, `file_type`)                   |
| ConvertToData  | `DataWorkBook` with `Summary` and `Measurements` sheets                           |
| CalculateShift | `ShiftSummary` sheet appended to the workbook                                    |
| GeneratePlot   | `Boxplot_<EVENT>` / `Histogram_<EVENT>` sheets plus updated `ShiftSummary` charts |

- Excel sources reuse the legacy unit analysis to count pass/fail devices.
- STDF sources stream through `istdf.STDFReader`, keep only the last attempt per serial number, and honour per-record limits.
- The `Measurements` sheet includes a `Serial Number` column and retains per-test metadata (name, unit, limits) for every device row.
- Interval ordering follows: `Prescreen` (case-insensitive), numeric values ascending, then remaining strings alphabetically.

## GUI Scaffolding

`shiftanalysis.gui.ShiftAnalysisGUI` contains placeholder methods (`launch`, `shutdown`) marked `TBD`. Implementations can hook into the CLI stages without disrupting the core workflow.

## Development Notes

- `shiftanalysis/models.py` defines the shared dataclasses exchanged between stages.
- `ShiftAnalysisPlan.txt` tracks the original requirements and completion status.
- The legacy `mergeXlsx.py` script now lives inside `shiftanalysis/stages/convert_to_data.py`; invoke the CLI instead of calling the script directly.
- Run small-sample validation with:

  ```bash
  python -m shiftanalysis.cli auto temp_validation --assume-yes --values-only --BoxChartOnly
  ```

- Linting/tests are not yet wired up; consider adding type-checking or unit tests before large refactors.

## Third-Party Components

Refer to `THIRD_PARTY_LICENSES.txt` for full notices. The project currently bundles:

- `openpyxl`
- `numpy`
- `matplotlib`
- `istdf` (vendored under `Submodules/istdf/`)

Ensure downstream distributions comply with the respective licenses.
