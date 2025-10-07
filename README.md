# CPK Analysis Workflow

This repository provides a high-throughput analysis pipeline for transforming large volumes of STDF measurements into a consolidated Excel workbook suitable for CPK reporting, limit management, and rapid chart review. The workflow ingests multiple STDF files, applies optional outlier filtering, computes Cp/Cpk statistics with configurable limit precedence, renders Plotly WebGL charts, and fills an external CPK template while emitting reproducibility metadata.

## Key Capabilities

- **STDF Ingestion** &mdash; Parses Standard Test Data Format files via the `iSTDF` submodule, preserving per-device context, STDF limits, and timestamps or measurement indices. Implements comprehensive STDF flag filtering to ensure only valid measurements contribute to statistical calculations.
- **Columnar Storage** &mdash; Streams measurements into Parquet using pandas + pyarrow for fast spill-to-disk without losing vectorized performance.
- **Data Quality Assurance** &mdash; Filters invalid measurements based on PARM_FLG and TEST_FLG specifications while preserving complete test visibility and catalog integrity.
- **Outlier Filtering** &mdash; Optional IQR or standard-deviation guards with configurable multipliers; undo by re-running with `--outlier-method none`.
- **Comprehensive Statistics** &mdash; Generates per-file/per-test metrics (COUNT, MEAN, MEDIAN, STDEV, IQR, CpL, CpU, Cpk, yield loss variants, 2.0 and 3xIQR targets).
- **Enhanced Template Integration** &mdash; Intelligent header matching that searches multiple rows to find template headers; properly populates test names, test numbers, STDF limits, and units data into template sheets.
- **Workbook Authoring** &mdash; Produces Summary, Measurements, and Test List & Limits sheets; embeds Matplotlib-rendered histogram/CDF/time-series charts; fills the required CPK template with hyperlinks into the histogram sheets.
- **Metadata Logging** &mdash; Captures processing parameters, limit sources, and per-source counts in a JSON sidecar for audit trails.

## Requirements

- **Python** 3.11 or newer (3.13 recommended)
- **Operating Systems**: Windows, macOS, or Linux
- **Python Packages** (installed by `pip install -r requirements.txt`):
  - `openpyxl` &mdash; Excel workbook authoring
  - `pandas`, `numpy`, `pyarrow` &mdash; columnar data processing
- `matplotlib` &mdash; Static chart rendering in PNG form

## Installation

```bash
git clone <repository-url>
cd cpkAnalysis
git submodule update --init --recursive
pip install -r requirements.txt
```

> **Note:** The STDF reader lives under `Submodules/istdf` and is pulled automatically by the `git submodule` command above.

## Command-Line Usage

Run the full pipeline against one or more STDF files:

```bash
python -m cpkanalysis.cli run D:/data/lot1.stdf D:/data/lot2.stdf --output CPK_Workbook.xlsx
```

Key options:

| Option | Description |
| --- | --- |
| `--metadata path.json` | Load STDF file list from a metadata JSON (see `scan`). |
| `--template path` | Excel template file or directory containing the CPK template; if a directory is supplied, the first `.xlsx` file is used. |
| `--outlier-method {none,iqr,stdev}` | Selects the outlier filter (default `none`). |
| `--outlier-k value` | Multiplier `k` for IQR or standard deviation filtering (default `1.5`). |
| `--no-histogram`, `--no-cdf`, `--no-time-series` | Skip generating the corresponding chart families. |

To quickly build a metadata manifest of STDF files in a directory:

```bash
python -m cpkanalysis.cli scan ./Sample --metadata stdf_sources.json
python -m cpkanalysis.cli run --metadata stdf_sources.json
```
Supplying `--template` (and optionally `--template-sheet`) causes the pipeline to refresh the matching template sheet automatically after chart generation.

### Move Template

Copy the latest CPK Report contents into the corresponding columns on the template sheet:

```bash
python -m cpkanalysis.cli move-template --workbook temp/CPK_Workbook.xlsx --sheet J95323
```

If `--sheet` is omitted, the first sheet other than CPK Report is used. The template integration features intelligent header detection that searches multiple rows to locate template headers, ensuring compatibility with various template formats. Headers are matched by name (TEST NAME, TEST NUM, UNITS, etc.) and data including hyperlinks and number formats are preserved during the copy process.\n\n## Minimal Console GUI

A lightweight text-driven harness mirrors the planned GUI flow and walks through file selection and option entry:

```bash
python -m cpkanalysis.gui
```

## Output Workbook Structure

| Sheet | Contents |
| --- | --- |
| **Summary** | Per-file/per-test statistics (COUNT, MEAN, MEDIAN, STDEV, IQR, CpL, CpU, Cpk, yield loss variants, 2.0 and 3xIQR projections). |
| **Measurements** (`Measurements`, `Measurements_2`, ...) | Flattened measurement table (`File`, `DeviceID`, `Test Name`, `Test Number`, `Value`, `Units`, `Timestamp/Index`) with Excel tables and frozen headers. |
| **Test List and Limits** | One row per test with STDF limits, Spec overrides, and User What-If limits; active limits respect the priority What-If > Spec > STDF. |
| **Histogram_* / CDF_* / TimeSeries_*`** | Matplotlib charts rendered to PNG for each file, arranged by test with consistent axes. |
| **CPK Report** | Template-driven report populated with computed statistics, test names, test numbers, STDF limits, units data, and hyperlinks that jump directly to the histogram chart for each test. |

The workbook is saved to the path supplied via `--output`, and a companion JSON metadata file is written alongside it (same stem, `.json` extension).

## Data & Chart Pipeline

1. **Ingestion:** `cpkanalysis.ingest` streams STDF records with `iSTDF.STDFReader`, collating measurement values, STDF limits, device identifiers, and execution timestamps. Measurements are stored both in-memory (pandas DataFrame) and on disk (`temp/session_*/raw_measurements.parquet`) for reproducibility.
2. **Outlier Filtering:** `cpkanalysis.outliers` optionally trims extreme values per file/test using configurable IQR or sigma bounds.
3. **Statistics:** `cpkanalysis.stats` computes all requested metrics, including robust 3xIQR variants, yielding source tracking (What-If vs Spec vs STDF) for metadata logging.
4. **Workbook Authoring:** `cpkanalysis.workbook_builder` lays out the Summary, Measurements, and Limits sheets, renders Matplotlib charts (histogram, CDF, time-series), and maps chart anchors back to the CPK template.
5. **Template Filling:** The existing `cpk_template` workbook is loaded and updated in place so Proposal/Lot Qual columns remain blank but PLOTS hyperlinks resolve into the histogram sheets.
6. **Metadata:** Pipeline settings, per-source counts, and limit provenance are captured in `<output>.json` for downstream automation.

Temporary artifacts are isolated under `temp/session_*` and removed automatically after a successful run.

## STDF Flag Filtering Technical Details

The system implements STDF specification-compliant flag filtering at the measurement ingestion level:

### Flag Processing Architecture
1. **Test Catalog Population**: All PTR records populate the test catalog regardless of flag status (ensures complete test visibility)
2. **Measurement Filtering**: Individual measurements are validated against flag criteria before inclusion in statistical calculations
3. **Dual-Layer Approach**: Maintains complete test coverage while ensuring data quality

### Flag Validation Logic
```python
# PARM_FLG validation
if parm_flg & 0x04:  # Bit 2: Result invalid
    return None      # Reject measurement

# TEST_FLG validation  
if test_flg & 0x7E:  # Bits 1,2,3,4,5,6: Multiple validity issues
    return None      # Reject measurement
```

### Measurement Exclusion Criteria
- **Equipment Issues**: Sensor calibration errors, test hardware malfunctions
- **Execution Problems**: Test timeouts, aborted tests, tests not executed
- **Data Quality**: Unreliable results, measurements without P/F indication
- **Invalid Results**: Flagged parameter and test result invalidity

### Benefits
- **Data Integrity**: Only reliable measurements contribute to CPK statistics
- **Quality Monitoring**: Track invalid measurement rates for equipment health
- **Complete Visibility**: All tests appear in reports regardless of measurement validity
- **Robust Statistics**: Accurate CPK calculations uncontaminated by invalid data

## Recent Improvements

### STDF Flag Filtering Enhancement (Major Update)
The system now implements comprehensive STDF flag filtering to ensure only valid, reliable measurements are included in CPK calculations while preserving complete test visibility:

#### Enhanced Data Quality
- **PARM_FLG Filtering**: Rejects measurements with bit 2 (0x04) set - indicating invalid parameter results
- **TEST_FLG Filtering**: Rejects measurements with bits 1,2,3,4,5,6 (0x7E mask) set:
  - Bit 1 (0x02): Result is not valid
  - Bit 2 (0x04): Test result is unreliable  
  - Bit 3 (0x08): Timeout occurred
  - Bit 4 (0x10): Test not executed
  - Bit 5 (0x20): Test aborted
  - Bit 6 (0x40): Test completed without P/F indication

#### Test Catalog Preservation Fix
**Critical Bug Fixed**: Tests with all invalid measurements no longer disappear from analysis
- **Problem**: Test catalog was only populated from valid measurements, causing tests with 100% invalid flags to vanish entirely
- **Solution**: Decoupled test catalog population from measurement filtering - all tests now appear in catalog regardless of measurement validity
- **Result**: Complete test visibility maintained, no silent data loss, but only valid measurements used for CPK calculations

#### Impact on Analysis
- **Improved Accuracy**: CPK calculations based only on reliable measurements
- **Complete Coverage**: All tests visible in reports even if they have no valid measurements
- **Equipment Monitoring**: Invalid measurement tracking enables quality monitoring and equipment health assessment
- **Data Integrity**: Proper filtering prevents contamination from sensor errors, timeouts, and execution failures

### Template Integration Enhancements
- **Fixed Template Header Alignment**: Resolved issues where test names and test numbers weren't populating in template sheets by aligning CPK Report column headers with template expectations (TEST NAME, TEST NUM).
- **Enhanced Header Detection**: Template processing now searches multiple rows instead of just the first row to locate headers, improving compatibility with various template formats.
- **STDF Limits Integration**: Added STDF limit values and units data to the CPK Report sheet for complete template population.
- **Statistical Data Enhancement**: Extended CPK Report with comprehensive statistical columns including all Cpk variants, yield loss calculations, and limit proposals.

### Bug Fixes
- **GUI Configuration**: Fixed missing template sheet parameter in the GUI interface to ensure proper template sheet selection.
- **Pipeline Imports**: Resolved import errors in the analysis pipeline for seamless execution.
- **Measurement Validation**: Enhanced STDF specification compliance with proper flag handling for measurement validity.

## Developing & Extending

- Keep `requirements.txt` synchronized when introducing new dependencies.
- Large edits are best made through the modular pipeline (`ingest`, `outliers`, `stats`, `workbook_builder`) to preserve testability.
- When modifying the workbook layout, ensure the column names on `Summary`, `Measurements`, and `Test List and Limits` stay consistent with the acceptance criteria.
- Plot exports rely on Matplotlib; adjust `_figure_to_png` in `mpl_charts.py` if you need different formats.

---

For questions, enhancement ideas, or integration guidance, open an issue or start a discussion referencing the relevant module (`cpkanalysis.ingest`, `cpkanalysis.stats`, etc.).


