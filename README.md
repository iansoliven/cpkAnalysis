# CPK Analysis Workflow

> **High-throughput STDF analysis pipeline for semiconductor test data → Excel CPK reports with statistical analysis and visualization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-90%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-75%25-brightgreen.svg)](test_coverage_report/index.html)

Transform large volumes of STDF (Standard Test Data Format) files into comprehensive CPK reports with:
- ✅ **STDF V4 compliant** ingestion with proper flag filtering
- 📊 **Statistical analysis** (CPK, yield loss, outlier detection)
- 📈 **Automated charting** (histograms, CDF, time-series)
- 🔄 **Interactive post-processing** for limit adjustments
- 🔌 **Extensible plugin system** for custom workflows

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Output Structure](#-output-structure)
- [Architecture](#-architecture)
- [STDF Ingestion](#-stdf-ingestion--flag-filtering)
- [Post-Processing](#-post-processing)
- [Plugin System](#-plugin-system)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Recent Improvements](#-recent-improvements)
- [Contributing](#-contributing)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (3.13 recommended)
- Windows, macOS, or Linux
- Git with submodule support

### Install

```bash
git clone https://github.com/iansoliven/cpkAnalysis.git
cd cpkAnalysis
git submodule update --init --recursive
pip install -r requirements.txt
```

### Run Your First Analysis

```bash
# Analyze STDF files and generate CPK report
python -m cpkanalysis.cli run Sample/lot2.stdf Sample/lot3.stdf \
  --output CPK_Workbook.xlsx \
  --template cpkTemplate/cpk_report_output_template.xlsx

# Launch interactive post-processing
python -m cpkanalysis.cli run Sample/lot2.stdf --postprocess
```

---

## ✨ Key Features

### 🔧 **STDF Ingestion**
- Parses Standard Test Data Format (STDF V4) files via `iSTDF` submodule
- **Comprehensive flag filtering**: PARM_FLG, TEST_FLG, OPT_FLAG validation
- **Dual-layer processing**: Complete test catalog + validated measurements
- **Smart limit handling**: Proper OPT_FLAG bits 4/5/6/7 semantics
- **Site data preservation**: SITE_NUM extraction for multi-site test systems
- Preserves device context, timestamps, and measurement indices

### 📊 **Statistical Analysis**
- **CPK metrics**: CPL, CPU, CPK with configurable limits
- **Per-site aggregation**: Optional site-by-site statistics for multi-site test heads
- **Yield loss calculations**: Percentage out-of-spec (overall + per-site)
- **Robust statistics**: 3xIQR alternatives using median
- **Limit precedence**: What-If > Spec > STDF priority
- **Outlier filtering**: Optional IQR or σ-based methods (site-aware grouping)

### 📈 **Chart Generation**
- **Histograms** with Freedman-Diaconis binning (overall + per-site)
- **Cumulative Distribution Functions** (CDF)
- **Time-series plots** with device sequence
- **Side-by-side layout**: Per-site plots appended horizontally
- Adaptive axis bounds with limit markers
- Multiple limit visualization (STDF, Spec, What-If, Proposed)

### 🔄 **Post-Processing**
- **Interactive menu** for limit updates without re-ingestion
- **Chart regeneration** with new markers
- **Target CPK calculations**: Compute proposed limits for desired capability
- **GRR-based proposed limits**: Calculate spec and FT limits with guardband protection from metrology (Gauge R&R) data
- **Audit logging**: Full traceability of changes
- Available via CLI and GUI

### 🔌 **Plugin System**
- Event-driven architecture with lifecycle hooks
- Multiple discovery mechanisms (entry points, TOML manifests)
- TOML profile management for persistent settings
- CLI overrides for per-run customization

### 📝 **Metadata & Traceability**
- JSON sidecar with processing parameters
- Source file statistics and limit provenance
- Post-processing audit trail
- Reproducibility metadata for compliance

---

## 📦 Installation

### 1. Clone Repository with Submodules

```bash
git clone https://github.com/iansoliven/cpkAnalysis.git
cd cpkAnalysis
git submodule update --init --recursive
```

> **Note**: The `Submodules/istdf` directory contains the STDF reader library

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `openpyxl` — Excel I/O
- `pandas`, `numpy` — Data manipulation
- `pyarrow` — Parquet storage
- `matplotlib` — Chart rendering

### 3. Verify Installation

```bash
python -m cpkanalysis.cli --help
pytest tests/ -v
```

---

## 💡 Usage Examples

### Basic Analysis

```bash
# Single STDF file
python -m cpkanalysis.cli run Sample/lot2.stdf --output results.xlsx

# Multiple files
python -m cpkanalysis.cli run Sample/*.stdf --output results.xlsx
```

### With Template Integration

```bash
python -m cpkanalysis.cli run Sample/lot2.stdf \
  --output CPK_Workbook.xlsx \
  --template cpkTemplate/template.xlsx \
  --template-sheet "J95323"
```

### Outlier Filtering

```bash
# IQR method (1.5x multiplier)
python -m cpkanalysis.cli run Sample/*.stdf \
  --outlier-method iqr \
  --outlier-k 1.5

# Standard deviation method (3σ)
python -m cpkanalysis.cli run Sample/*.stdf \
  --outlier-method stdev \
  --outlier-k 3.0
```

### Scan Directory for STDFs

```bash
# Create manifest
python -m cpkanalysis.cli scan ./TestData --metadata manifest.json

# Use manifest
python -m cpkanalysis.cli run --metadata manifest.json --output results.xlsx
```

### Chart Control

```bash
# Skip time-series charts for faster processing
python -m cpkanalysis.cli run Sample/*.stdf --no-time-series

# Generate only histograms
python -m cpkanalysis.cli run Sample/*.stdf --no-cdf --no-time-series
```

### Yield & Pareto Analysis

```bash
# Create Yield and Pareto tables and charts
python -m cpkanalysis.cli run Sample/*.stdf --generate-yield-pareto
```

Generates a `Yield and Pareto` sheet with file-level yield summaries, per-test Pareto tables, and paired charts for each STDF file.

### Per-Site Aggregation *(optional)*

Modern semiconductor testers use **multi-site test heads** to maximize throughput by testing multiple devices simultaneously. CPK Analysis supports optional **per-site aggregation** to analyze each test site independently.

#### Detection & Prompting

```bash
# Automatic detection - prompts if SITE_NUM found
python -m cpkanalysis.cli run Sample/*.stdf

# Example output:
# Detecting site support in STDF files...
# ✓ SITE_NUM data detected (sites: 1, 2, 3, 4)
# Enable per-site statistics and charts? [y/N]: y
```

#### Explicit Control

```bash
# Force enable per-site mode (skip prompt)
python -m cpkanalysis.cli run Sample/*.stdf --site-breakdown

# Force disable even if SITE_NUM exists
python -m cpkanalysis.cli run Sample/*.stdf --no-site-breakdown
```

#### What You Get

**When enabled**, the workbook includes:

1. **Per-Site CPK Statistics**
   - Independent CPL, CPU, CPK calculations per site
   - Site column in CPK Report sheet
   - Clickable hyperlinks to site-specific plots

2. **Per-Site Yield & Pareto**
   - Yield percentages calculated per (file, site) combination
   - Pareto analysis showing top failures per site
   - Horizontal layout: Overall | Site 1 | Site 2 | Site 3...

3. **Per-Site Charts**
   - Histogram plots for each site (side-by-side with overall)
   - CDF and time-series plots per site
   - Consistent axis bounds for easy comparison

4. **Site-Aware Outlier Filtering**
   - Outliers calculated per site (not globally)
   - Value might be outlier in site 1 but normal in site 2

#### Layout Design

```
Excel Workbook Layout (Horizontal Blocks):

┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  Overall        │  Site 1         │  Site 2         │  Site 3         │
│  (All Sites)    │                 │                 │                 │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│  VDD Test       │  VDD - Site 1   │  VDD - Site 2   │  VDD - Site 3   │
│  [Histogram]    │  [Histogram]    │  [Histogram]    │  [Histogram]    │
│  [CDF]          │  [CDF]          │  [CDF]          │  [CDF]          │
│  [TimeSeries]   │  [TimeSeries]   │  [TimeSeries]   │  [TimeSeries]   │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
  Columns 1-18      Columns 19-36     Columns 37-54     Columns 55-72
```

#### Use Cases

**Multi-Site Test Head Comparison**
- Identify systematic bias in specific test sites
- Example: Site 3 shows CPK=0.8 while others show CPK=1.3 → calibration issue

**Production Monitoring**
- Track site-specific yield trends over time
- Spot equipment degradation in individual sites

**Graceful Degradation**
- If SITE_NUM data missing, analysis proceeds normally
- Clear warning if site breakdown requested but unavailable
- Zero impact when disabled (default behavior unchanged)

#### Metadata Tracking

```json
{
  "site_breakdown": {
    "requested": true,   // User asked for it
    "available": true,   // SITE_NUM found in data
    "generated": true    // Output includes per-site stats
  },
  "site_summary": { "rows": 42 },
  "site_yield_summary": { "rows": 6 },
  "site_pareto_summary": { "rows": 18 }
}
```

**See also:** [docs/STDF_INGESTION.md](docs/STDF_INGESTION.md#site-data-handling) for technical details

### Output Formatting

```bash
# Limit fallback decimals to two places when STDF format hints are absent
python -m cpkanalysis.cli run Sample/*.stdf --display-decimals 2
```

When available, STDF `C_RESFMT`, `C_LLMFMT`, and `C_HLMFMT` strings drive Excel number formats. The `--display-decimals` option (also available in the console GUI) only applies when no format hint is provided and defaults to four decimal places.

### Interactive GUI

```bash
# Launch console GUI with prompts
python -m cpkanalysis.gui
```

The GUI walks through:
1. STDF file selection
2. Template configuration
3. Outlier settings
4. Chart preferences (histogram, CDF, time-series, Yield & Pareto)
5. Format preferences
6. Plugin selection
7. Output path

### Post-Processing Existing Workbook

```bash
# Open post-processing menu
python -m cpkanalysis.cli post-process --workbook CPK_Workbook.xlsx
```

**Menu options:**
- Update STDF limits
- Apply Spec/What-If limits
- Calculate proposed limits for target CPK
- View audit log
- Reload workbook

---

## 📊 Output Structure

### Excel Workbook Sheets

| Sheet | Contents |
|-------|----------|
| **Summary** | Per-file/per-test statistics: COUNT, MEAN, MEDIAN, STDEV, IQR, CPL, CPU, CPK, yield loss variants, 2.0 CPK projections, 3xIQR alternatives |
| **Measurements** | Flattened measurement table with File, DeviceID, Test Name, Test Number, Value, Units, Timestamp/Index (split into multiple sheets if >1M rows) |
| **Test List and Limits** | Test catalog with STDF limits, Spec overrides, User What-If limits (priority: What-If > Spec > STDF) |
| **Yield and Pareto** | File-level yield summary plus per-test Pareto tables and charts; per-site sections are appended to the right when enabled |
| **Histogram_\*** | Matplotlib-rendered histograms per file, arranged by test with limit markers (per-site plots occupy additional blocks to the right) |
| **CDF_\*** | Cumulative distribution function plots with limit markers |
| **TimeSeries_\*** | Time-series plots showing measurement progression with limit bands |
| **CPK Report** | Template-driven report populated with statistics, test info, and clickable hyperlinks to charts (includes site column when per-site data is generated) |
| **\_PlotAxisRanges** | Hidden metadata sheet for axis bounds (enables consistent chart regeneration) |

### Companion Metadata JSON

**File**: `<output>.json` (e.g., `CPK_Workbook.json`)

```json
{
  "analysis_inputs": {
    "sources": ["lot2.stdf", "lot3.stdf"],
    "outlier_method": "none",
    "generate_histogram": true,
    "template_path": "cpkTemplate/template.xlsx",
    "enable_site_breakdown": true
  },
  "site_breakdown": {
    "requested": true,
    "available": true,
    "generated": true
  },
  "site_summary": {
    "rows": 42
  },
  "site_yield_summary": {
    "rows": 6
  },
  "site_pareto_summary": {
    "rows": 4
  },
  "per_file_stats": [
    {
      "file": "lot2.stdf",
      "measurement_count": 150000,
      "device_count": 1500,
      "invalid_measurements_filtered": 234
    }
  ],
  "limit_sources": {
    "VDD_CORE": {
      "lower": "stdf",
      "upper": "what_if"
    }
  },
  "post_processing": {
    "runs": [
      {
        "action": "update_stdf_limits",
        "scope": "all",
        "target_cpk": 2.0,
        "timestamp": "2025-10-11T04:38:12"
      }
    ]
  }
}
```

### Temporary Files

- `temp/session_*/raw_measurements.parquet` — Intermediate storage (auto-deleted)
- `post_processing_profile.toml` — Plugin configuration (persistent)

---

## 🏗️ Architecture

### Pipeline Stages

```
┌─────────────┐
│   STDF      │
│   Files     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  1. INGESTION       │  ← Parse PTR/PRR records, filter invalid flags
│  cpkanalysis.ingest │  ← Build test catalog + measurement DataFrame
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  2. OUTLIER FILTER  │  ← Optional IQR or σ-based trimming
│  cpkanalysis.outliers│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  3. STATISTICS      │  ← Compute CPK, yield loss, limit sources
│  cpkanalysis.stats  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  4. CHARTS          │  ← Render histograms, CDF, time-series (PNG)
│  cpkanalysis.mpl_charts│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  5. WORKBOOK        │  ← Build Excel sheets, embed charts
│  cpkanalysis.workbook_builder│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  6. TEMPLATE FILL   │  ← Populate external CPK template
│  cpkanalysis.move_to_template│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  7. METADATA        │  ← Write JSON sidecar with audit trail
└─────────────────────┘
```

### Event-Driven Plugin System

```python
# Pipeline emits events at each stage
class PipelineEvent:
    IngestReadyEvent      # After STDF parsing
    OutlierReadyEvent     # After outlier filtering
    SummaryReadyEvent     # After statistics computation
    WorkbookReadyEvent    # After Excel generation
    TemplateReadyEvent    # After template population
    MetadataReadyEvent    # After metadata write

# Plugins subscribe to events
@plugin.on_event("SummaryReadyEvent")
def log_summary(event):
    print(f"Computed {len(event.summary)} test summaries")
```

---

## 🔧 STDF Ingestion & Flag Filtering

### Specification Compliance

The system implements **STDF V4** specification-compliant ingestion with comprehensive flag filtering and proper OPT_FLAG bit handling.

### Flag Processing Architecture

1. **Test Catalog Population** — All PTR records populate catalog (ensures complete test visibility)
2. **Measurement Filtering** — Individual measurements validated against flags
3. **Dual-Layer Approach** — Complete coverage + data quality

### Flag Validation

```python
# PARM_FLG: Bit 2 (0x04) = Result invalid
if parm_flg & 0x04:
    return None  # Reject measurement

# TEST_FLG: Bits 1-6 (0x7E) = Various validity issues
# Rejects: Invalid, Unreliable, Timeout, Not Executed, Aborted, No P/F
if test_flg & 0x7E:
    return None  # Reject measurement
```

### OPT_FLAG Bit Handling

| Bit | Hex  | Meaning | Implementation |
|-----|------|---------|----------------|
| 4   | 0x10 | LO_LIMIT invalid → use default from first PTR | ✅ Preserves cached default |
| 5   | 0x20 | HI_LIMIT invalid → use default from first PTR | ✅ Preserves cached default |
| 6   | 0x40 | No low limit exists for this test | ✅ Clears limit |
| 7   | 0x80 | No high limit exists for this test | ✅ Clears limit |

**Key Insight**: Bits 4/5 (use defaults) ≠ Bits 6/7 (no limits)

### Benefits

- ✅ **Data Integrity**: Only reliable measurements contribute to CPK
- ✅ **Complete Visibility**: All tests appear even with 0 valid measurements
- ✅ **Limit Propagation**: Default limits correctly carry forward (~80% of STDFs use this)
- ✅ **Equipment Monitoring**: Track invalid measurement rates

### 📚 Deep Dive

For comprehensive technical details including:
- Complete OPT_FLAG bit definitions
- Real-world ATE usage patterns
- Metadata caching architecture
- Edge case handling
- Before/after fix comparisons

**See: [docs/STDF_INGESTION.md](docs/STDF_INGESTION.md)** | **[HTML Version](help/stdf_ingestion.html)**

---

## 🔄 Post-Processing

Modify workbook limits and regenerate charts **without re-running ingestion**.

### Launch Post-Processing

```bash
# After initial run
python -m cpkanalysis.cli run lot1.stdf --postprocess

# Existing workbook
python -m cpkanalysis.cli post-process --workbook CPK_Workbook.xlsx

# From GUI
python -m cpkanalysis.gui  # Then type 'post' at the prompt
```

### Available Actions

#### 1. Update STDF Limits
- Recompute ATE lower/upper limits
- Optional target CPK parameter
- Updates template + Test List & Limits catalog
- Regenerates all affected charts

#### 2. Apply Spec / What-If Limits
- Propagate manual Spec/What-If edits
- Or compute from target CPK
- Adds additional limit markers to charts
- Original STDF markers preserved

#### 3. Calculate Proposed Limits
- Computes LL_PROP, UL_PROP for target CPK
- Adds CPK_PROP and %YLD LOSS_PROP columns
- Updates charts with "Proposed" markers
- Extends axis bounds as needed

#### 4. Calculate Proposed Limits (GRR-Based)
- **GRR-aware limit calculation** for production readiness
- Ingests Gauge R&R data from `Total_GRR.xlsx` workbook
- Applies **guardband protection** (50% or 100% GRR) to spec walls
- Computes both **spec limits** and **FT (Final Test) limits**
- Automatically widens specs when guardband causes CPK violations
- Adds four new columns: Proposed Spec Lower/Upper, CPK Proposed Spec, Guardband Selection
- Generates overlay charts showing original and proposed limits
- Ensures measurement capability (GRR) won't compromise yield

**GRR Workflow:**
1. Prompt for GRR data directory containing `Total_GRR.xlsx`
2. Enter minimum and maximum target FT CPK values
3. Select scope (all tests or single test)
4. Algorithm applies guardband ladder (100% → 50% → spec widening)
5. Writes proposed spec and FT limits to new columns
6. Regenerates charts with proposed limit overlays

#### 5. Utility Functions
- **Re-run last action** — Repeat with same parameters
- **Reload workbook** — Discard unsaved changes
- **View audit log** — Display all post-processing runs
- **Exit** — Save and close menu

### Scope Selection

Each action prompts for scope:
- **All tests** — Applies to entire workbook
- **Single test** — Searchable list by name/number

### Target CPK

When prompted for target CPK:
- **Blank** — Reuse existing statistics/limits
- **Positive value** — Compute symmetrical limits: `mean ± (target_cpk × 3σ)`

### Audit Trail

All changes logged to metadata JSON:

```json
{
  "post_processing": {
    "runs": [
      {
        "action": "update_stdf_limits",
        "scope": "single",
        "tests": [["VDD_CORE", "100"]],
        "target_cpk": 2.0,
        "warnings": [],
        "timestamp": "2025-10-11T04:38:12.123456"
      },
      {
        "action": "calculate_proposed_limits_grr",
        "scope": "all",
        "grr_directory": "GRR_Data/",
        "min_ft_cpk": 1.33,
        "max_ft_cpk": 2.0,
        "tests_processed": 45,
        "tests_widened": 3,
        "warnings": [],
        "timestamp": "2025-10-22T14:23:45.678901"
      }
    ]
  }
}
```

### GRR-Based Proposed Limits

**New in 2025-10-22**: Post-processing now supports **Gauge R&R (GRR) driven limit proposals** that account for measurement system capability when setting production limits.

#### Why GRR Matters

In semiconductor testing, measurement repeatability and reproducibility (Gauge R&R) affects how tightly you can set limits:
- **Tight limits + poor GRR** = False rejects (good parts fail due to measurement noise)
- **GRR guardband** = Safety margin between spec and test limits
- **Common practice**: Reserve 50% or 100% of GRR as buffer

#### What This Feature Does

1. **Reads GRR Data**: Ingests `Total_GRR.xlsx` with per-test GRR measurements
2. **Applies Guardband Ladder**:
   - Try 100% GRR guardband first
   - If CPK too low, try 50% GRR
   - If still too low, widen spec to meet minimum CPK target
3. **Computes Dual Limits**:
   - **Spec Limits**: Customer/design requirements (possibly widened)
   - **FT Limits**: Tighter limits for production test (spec minus guardband)
4. **Outputs**:
   - Four new template columns (Proposed Spec Lower/Upper, CPK Proposed Spec, Guardband Selection)
   - Overlay charts showing original vs proposed limits
   - Audit trail with guardband decisions

#### Example Scenario

```
Test: VDD_CORE
Original Spec: 0.95V - 1.05V
GRR: 0.02V (worst case)
Mean: 1.00V, Stdev: 0.015V
Min FT CPK Target: 1.33

Calculation:
1. Try 100% GRR guardband (0.02V):
   FT Limits: 0.97V - 1.03V
   FT CPK = min((1.03-1.00), (1.00-0.97)) / (3×0.015) = 0.67 ❌ (< 1.33)

2. Try 50% GRR guardband (0.01V):
   FT Limits: 0.96V - 1.04V
   FT CPK = min((1.04-1.00), (1.00-0.96)) / (3×0.015) = 0.89 ❌ (< 1.33)

3. Widen spec to meet CPK target:
   Required FT range: 1.00V ± (1.33 × 3 × 0.015) = 0.9401V - 1.0599V
   Add 50% GRR guardband: Spec must be 0.9301V - 1.0699V

Result:
   Proposed Spec: 0.93V - 1.07V (widened)
   Proposed FT: 0.94V - 1.06V (with 0.01V guardband)
   FT CPK: 1.33 ✓
   Guardband: "50% GRR"
```

#### GRR File Format

The `Total_GRR.xlsx` file should contain a sheet named `"GRR data"` (or use the first sheet) with these columns:

| Column | Variations Accepted | Description |
|--------|---------------------|-------------|
| Test Number | `Test #`, `Test Num`, `Test` | Numeric test ID (leading zeros stripped) |
| Test Name | `Test Name`, `Description` | Test name for matching |
| Unit | `Unit`, `Units` | Measurement units (normalized: V→VOLTS) |
| Spec Min | `Min\nSpec`, `Spec Min`, `Lower Spec` | Lower specification limit |
| Spec Max | `Max\nSpec`, `Spec Max`, `Upper Spec` | Upper specification limit |
| GRR | `Worst R&R` | GRR magnitude (worst case across conditions) |

**Header normalization** handles whitespace, newlines, and case variations automatically.

#### Access via CLI

```bash
# Launch post-processing menu
python -m cpkanalysis.cli post-process --workbook CPK_Workbook.xlsx

# Select option 4: "Calculate Proposed Limits (GRR-Based)"
# Follow prompts for GRR directory, CPK targets, and scope
```

---

## 🔌 Plugin System

Extend the pipeline with custom event handlers.

### Discovery Mechanisms

1. **Python Entry Points** (`pyproject.toml` or `setup.py`)
   ```toml
   [project.entry-points."cpkanalysis.pipeline_plugins"]
   my_plugin = "mypackage.plugin:MyPlugin"
   ```

2. **TOML Manifests** (`cpk_plugins/*.toml` in workspace)
   ```toml
   [plugin]
   id = "custom.data_exporter"
   name = "Data Exporter"
   enabled = true
   priority = 50

   [plugin.params]
   output_format = "csv"
   ```

3. **Built-in Plugins**
   - `builtin.summary_logger` — Logs summary row counts

### CLI Overrides

```bash
# Enable plugin for this run
python -m cpkanalysis.cli run lot1.stdf --plugin enable:custom.data_exporter

# Change priority
python -m cpkanalysis.cli run lot1.stdf --plugin priority:custom.data_exporter:10

# Override parameter
python -m cpkanalysis.cli run lot1.stdf --plugin param:custom.data_exporter:output_format=json

# Disable plugin
python -m cpkanalysis.cli run lot1.stdf --plugin disable:builtin.summary_logger
```

### Persistent Profiles

Settings stored in `post_processing_profile.toml`:

```toml
[plugins.enabled]
"builtin.summary_logger" = true
"custom.data_exporter" = false

[plugins.priorities]
"builtin.summary_logger" = 50

[plugins.params."builtin.summary_logger"]
message = "Summary complete: {rows} rows"
```

### Plugin Development

```python
from cpkanalysis.plugins import Plugin, on_event

class MyPlugin(Plugin):
    @on_event("SummaryReadyEvent")
    def process_summary(self, event):
        summary = event.summary
        # Custom processing here
        print(f"Processed {len(summary)} tests")

    @on_event("WorkbookReadyEvent")
    def enhance_workbook(self, event):
        workbook = event.workbook
        # Add custom sheets or modifications
```

---

## 📚 Documentation

### Help System

- **[Getting Started](help/getting_started.html)** — Installation and first run
- **[STDF Ingestion](help/stdf_ingestion.html)** — Technical deep dive (NEW!)
- **[CLI Reference](help/cli_reference.html)** — Command-line options
- **[GUI Guide](help/gui_reference.html)** — Console GUI walkthrough
- **[Post-Processing](help/post_processing.html)** — Interactive menu details
- **[Testing Guidance](help/testing_guidance.html)** — Automated test suite
- **[Manual Verification](help/manual_verification.html)** — QA checklist

### Technical References

- **[STDF_INGESTION.md](docs/STDF_INGESTION.md)** — 700+ line technical reference
- **[Architecture Diagrams](docs/)** — Pipeline flow and data models
- **[API Documentation](#)** — Module-level API reference (coming soon)

### Quick Links

- 📖 **[Full README](README.md)** — This file
- 🏠 **[Help Index](help/index.html)** — Central documentation hub
- 🐛 **[Issue Tracker](https://github.com/iansoliven/cpkAnalysis/issues)** — Bug reports
- 💬 **[Discussions](https://github.com/iansoliven/cpkAnalysis/discussions)** — Q&A

---

## 🧪 Testing

### Comprehensive Test Suite

- **90 tests** across 17 test modules
- **75% code coverage** (3,151 of 4,175 statements)
- ✅ All tests passing
- Automated CI/CD ready

### Quick Start

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=cpkanalysis --cov-report=html:test_coverage_report
# View test_coverage_report/index.html

# Run specific area
pytest tests/test_ingest_*.py -v      # Ingestion tests
pytest tests/test_stats_*.py -v       # Statistics tests
pytest tests/test_postprocess*.py -v  # Post-processing tests
```

### Coverage Highlights

| Module | Coverage | Status |
|--------|----------|--------|
| `mpl_charts.py` | 92% | ✅ Excellent |
| `outliers.py` | 90% | ✅ Excellent |
| `pipeline.py` | 88% | ✅ Excellent |
| `stats.py` | 87% | ✅ Excellent |
| `ingest.py` | 84% | ✅ Very Good |
| `plugins.py` | 83% | ✅ Very Good |

### For Developers

**See [help/testing_guidance.html](help/testing_guidance.html) for:**
- Complete test file listing (all 17 modules)
- Coverage statistics by module
- How to write new tests
- CI/CD integration examples
- Contributing guidelines
- Debugging test failures

### Contributing

All code contributions **must include tests**. See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Pre-commit checklist
- Code quality standards
- Pull request guidelines

---

## 🎉 Recent Improvements

### ✨ NEW: GRR-Based Proposed Limits (2025-10-22)

**Production-ready limit proposals with Gauge R&R guardband protection**

#### Features Added
- **GRR-aware calculations**: Ingests metrology data from `Total_GRR.xlsx` workbook
- **Guardband ladder logic**: Automatically tries 100% GRR → 50% GRR → spec widening
- **Dual limit computation**: Calculates both spec limits and FT (Final Test) limits
- **Intelligent spec adjustment**: Widens specs when guardband causes CPK violations below target
- **New workbook columns**: Proposed Spec Lower/Upper, CPK Proposed Spec, Guardband Selection
- **Overlay charts**: Generates new chart sheets showing original vs proposed limits side-by-side
- **Unit normalization**: Handles common unit aliases (V→VOLTS, mA→MILLIAMPS) and mismatches
- **Audit logging**: Tracks guardband decisions, spec widening events, and CPK targets

#### Benefits
✅ **Measurement-aware limits**: Prevents false rejects by accounting for test equipment capability
✅ **Automated guardband selection**: No manual calculation of GRR offsets
✅ **Spec optimization**: Identifies when specs need widening to meet CPK targets
✅ **Visual validation**: Side-by-side charts make proposal review easy
✅ **Production readiness**: Follows industry best practices for limit setting
✅ **Traceability**: Full audit trail of guardband and limit decisions

#### Example Workflow
```bash
# Run post-processing menu
python -m cpkanalysis.cli post-process --workbook CPK_Workbook.xlsx

# Select: 4. Calculate Proposed Limits (GRR-Based)
# Prompts:
#   GRR directory: ./GRR_Data/
#   Min FT CPK: 1.33
#   Max FT CPK: 2.0
#   Scope: All tests

# Result:
#   45 tests processed
#   3 specs widened (VDD_CORE, ILOAD, FREQ_OSC)
#   New columns populated with proposed limits
#   Overlay charts created in "Histogram_lot2 (Proposed)" sheet
```

**Documentation**: See [GRR-Based Proposed Limits](#grr-based-proposed-limits) section above

**Implementation**: [cpkanalysis/postprocess/proposed_limits_grr.py](cpkanalysis/postprocess/proposed_limits_grr.py)

---

### ✨ NEW: Per-Site Aggregation (2025-10-19)

**Optional multi-site test head analysis for production monitoring**

#### Features Added
- **Site data preservation**: SITE_NUM extracted from PIR records during ingestion
- **Automatic detection**: Probes STDF files (first 500 records) to check for site data
- **User control**: CLI flags `--site-breakdown` / `--no-site-breakdown` or interactive prompt
- **Site-aware statistics**: Independent CPK/yield/Pareto calculations per test site
- **Site-aware outlier filtering**: Outliers calculated per (file, site, test) group
- **Horizontal layout**: Per-site plots and tables appended right of overall data
- **Metadata tracking**: JSON captures requested vs. available vs. generated state

#### Benefits
✅ **Multi-site test head comparison**: Identify systematic bias in specific sites
✅ **Equipment monitoring**: Track site-specific yield trends and degradation
✅ **Backward compatible**: Disabled by default, zero impact when off
✅ **Graceful degradation**: Clear warnings when unavailable
✅ **Non-destructive**: Overall stats preserved, site data appended

#### Example Output
```
CPK Report:
  Overall: VDD CPK=1.2 (all sites combined)
  Site 1:  VDD CPK=1.3 ✓
  Site 2:  VDD CPK=1.4 ✓
  Site 3:  VDD CPK=0.8 ⚠ ← Equipment calibration issue detected!
  Site 4:  VDD CPK=1.3 ✓
```

**Documentation**: See [docs/STDF_INGESTION.md#site-data-handling](docs/STDF_INGESTION.md#site-data-handling)

**Tests**: 19 new tests covering detection, statistics, layout, and integration

---

### ⚠️ Critical Fix: STDF OPT_FLAG Bits 4 & 5 (2025-10-11)

**Fixed critical data loss affecting ~80% of production STDF files**

#### Problem
Code only handled OPT_FLAG bits 6 & 7 ("no limit exists"), **completely ignoring** bits 4 & 5 ("use default from first PTR"). When ATE systems used the default limit optimization (industry standard), limits were incorrectly overwritten or cleared.

#### Impact
```
Device 1:  opt_flg=0x00, LO_LIMIT=1.0   → 1.0 ✅
Device 2:  opt_flg=0x10, LO_LIMIT=999   → OLD: 999 ❌  NEW: 1.0 ✅ (uses default)
Device 3:  opt_flg=0x10, LO_LIMIT=None  → OLD: None ❌  NEW: 1.0 ✅ (uses default)
Device 4:  opt_flg=0x00, LO_LIMIT=None  → OLD: None ❌  NEW: 1.0 ✅ (preserves)
Device 5:  opt_flg=0x40, LO_LIMIT=None  → OLD: 1.0 ❌  NEW: None ✅ (explicit clear)
```

#### Solution
- Properly distinguish "use default" (bits 4/5) from "no limit" (bits 6/7)
- Implement metadata caching for default limit propagation
- Add priority order: bit 6/7 > bit 4/5 > explicit value > preserve existing

#### Verification
- ✅ 7 new tests added covering all OPT_FLAG scenarios
- ✅ All 14 tests pass (100% success rate)
- ✅ Comprehensive documentation in [STDF_INGESTION.md](docs/STDF_INGESTION.md)

---

### 🔍 STDF Flag Filtering Enhancement

**Comprehensive flag filtering** for measurement validation:

- **PARM_FLG**: Reject bit 2 (0x04) — Invalid parameter results
- **TEST_FLG**: Reject bits 1-6 (0x7E) — Invalid, unreliable, timeout, not executed, aborted, no P/F
- **Dual-layer processing**: Complete test catalog + validated measurements only

**Benefits:**
- Improved CPK accuracy (no contamination from invalid data)
- Equipment health monitoring (track invalid measurement rates)
- Complete test visibility (tests appear even with 0 valid measurements)

---

### 📝 Template Integration

- **Intelligent header detection** — Searches multiple rows for template headers
- **Column alignment** — Proper TEST NAME, TEST NUM mapping
- **STDF limit integration** — Populates ATE limit columns
- **Hyperlink preservation** — Maintains clickable chart links

---

### 🐛 Bug Fixes

- **GUI template sheet parameter** — Fixed missing sheet selection
- **Pipeline imports** — Resolved module import errors
- **Measurement validation** — Enhanced STDF specification compliance
- **Test catalog preservation** — Fixed disappearing tests with 100% invalid flags

---

## 🤝 Contributing

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/iansoliven/cpkAnalysis.git
cd cpkAnalysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black mypy ruff

# Run tests
pytest
```

### Code Style

- **Formatting**: `black cpkanalysis/`
- **Linting**: `ruff check cpkanalysis/`
- **Type checking**: `mypy cpkanalysis/`

### Pull Request Guidelines

1. **Add tests** for new functionality
2. **Update documentation** (README, help files, docstrings)
3. **Run test suite** (`pytest`) — must pass
4. **Follow existing patterns** (dataclasses, type hints, event-driven)
5. **Reference issue number** in PR description

### Reporting Issues

When reporting bugs, include:
- Python version
- Operating system
- STDF file characteristics (if applicable)
- Full error traceback
- Steps to reproduce

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **iSTDF** — STDF reader library (submodule)
- **STDF V4 Specification** — Semiconductor test data standard
- **Contributors** — See [CONTRIBUTORS.md](CONTRIBUTORS.md)

---

## 📞 Support

- 📖 **Documentation**: [help/index.html](help/index.html)
- 🐛 **Issues**: [GitHub Issues](https://github.com/iansoliven/cpkAnalysis/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/iansoliven/cpkAnalysis/discussions)

---

**Made with ❤️ for semiconductor test engineers**

*Last updated: 2025-10-19*


