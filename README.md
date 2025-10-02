# Shift Analysis Application

A comprehensive Python application for analyzing measurement data shifts across time intervals from Excel workbooks and STDF test files. This tool provides automated processing of semiconductor test data to identify performance degradation patterns over stress testing intervals.

## Overview

The Shift Analysis Application processes test measurement files following a structured naming convention, extracts measurement data, calculates statistical shifts between time intervals, and generates visualizations. It supports both Excel workbooks (`.xlsx`) and Standard Test Data Format (`.stdf`) files commonly used in semiconductor testing.

### Key Features

- **Automated File Discovery**: Scans directories for files matching the required naming pattern
- **Multi-Format Support**: Processes both Excel (.xlsx) and STDF (.stdf) files
- **Data Consolidation**: Combines measurements from multiple sources into a unified DataWorkbook
- **Statistical Analysis**: Calculates mean shifts, standard deviation changes, and interval comparisons
- **Visualization**: Generates histogram and boxplot charts for trend analysis
- **CLI Interface**: Complete command-line workflow with individual stage control
- **GUI Scaffolding**: Prepared for future graphical interface integration

## System Requirements

- **Python**: 3.11+ (with type hints and modern features)
- **Operating System**: Windows, macOS, Linux
- **Memory**: Sufficient for processing large Excel files (typically 1GB+ recommended)
- **Storage**: Space for input files plus generated DataWorkbooks and charts

### Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `openpyxl>=3.1` - Excel file manipulation and workbook generation
- `matplotlib>=3.10` - Chart generation and plotting
- `numpy>=2.3` - Numerical computations and data processing

## Installation & Setup

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd shiftanalysis
   ```

2. **Initialize Submodules**:
   ```bash
   git submodule init
   git submodule update
   ```
   
   This initializes the `Submodules/istdf` dependency required for STDF file processing.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**:
   ```bash
   python -m shiftanalysis.cli --help
   ```

## Usage Guide

### Full Automated Pipeline (Recommended)

Run the complete shift analysis workflow in one command:

```bash
# Run full pipeline on current directory
python -m shiftanalysis.cli auto

# Run on specific directory with options
python -m shiftanalysis.cli auto /path/to/test/files --assume-yes --values-only

# Common automation scenarios
python -m shiftanalysis.cli auto Sample --assume-yes  # Non-interactive mode
python -m shiftanalysis.cli auto --output MyAnalysis.xlsx  # Custom output name
```

**Pipeline Stages Executed:**
1. **ReadDirectory** - Discover and confirm source files
2. **ConvertToData** - Extract and consolidate measurement data
3. **Calculate Shift** - Compute statistical shifts between intervals
4. **Generate Charts** - Create histogram and boxplot visualizations

### Individual Stage Commands

For granular control or debugging, run individual pipeline stages:

#### 1. File Discovery & Confirmation
```bash
# Scan current directory
python -m shiftanalysis.cli readdir

# Scan specific directory
python -m shiftanalysis.cli readdir /path/to/files

# Non-interactive mode (skip confirmation)
python -m shiftanalysis.cli readdir --assume-yes

# Custom metadata file
python -m shiftanalysis.cli readdir --metadata my_sources.json
```

#### 2. Data Conversion
```bash
# Convert using saved metadata
python -m shiftanalysis.cli convert

# Fast processing (skip Excel formatting)
python -m shiftanalysis.cli convert --values-only

# Custom output workbook
python -m shiftanalysis.cli convert --output Analysis_Results.xlsx
```

#### 3. Shift Calculation
```bash
# Calculate shifts using existing DataWorkbook
python -m shiftanalysis.cli calc-shift

# Specify workbook location
python -m shiftanalysis.cli calc-shift --output MyWorkbook.xlsx
```

#### 4. Chart Generation
```bash
# Generate both histogram and boxplot charts
python -m shiftanalysis.cli plot

# Generate only boxplots
python -m shiftanalysis.cli plot --BoxChartOnly

# Generate only histograms
python -m shiftanalysis.cli plot --HistoChartOnly
```

### Command-Line Options

**Global Options** (available for all commands):
- `--output PATH` - Output workbook filename (default: `DataWorkbook.xlsx`)
- `--metadata PATH` - Metadata JSON file (default: `shift_sources.json`)

**ReadDirectory Options**:
- `--assume-yes` - Skip user confirmation prompt (automation mode)

**Convert Options**:
- `--values-only` - Skip Excel formatting for faster processing

**Plot Options**:
- `--BoxChartOnly` - Generate only boxplot charts
- `--HistoChartOnly` - Generate only histogram charts

### File Naming Convention

Input files must follow the strict pattern: `PREFIX_LOT_EVENT_INTERVAL.extension`

**Pattern**: `XXX_LOT_EVENT_INTERVAL.{xlsx|stdf}`

**Examples**:
```
A104549HTA_A_HAST_Prescreen.xlsx    # Lot A, HAST event, Prescreen interval
B104549HTB_B_HAST_0.xlsx           # Lot B, HAST event, 0-hour interval  
C104549UHC_C_UHAST_96.stdf         # Lot C, UHAST event, 96-hour interval
D104549HTD_D_HAST_192.xlsx         # Lot D, HAST event, 192-hour interval
```

**Component Definitions**:
- `PREFIX` - Project/wafer identifier (any alphanumeric text)
- `LOT` - Lot designation (typically A, B, C, etc.)
- `EVENT` - Test event type (HAST, UHAST, HTOL, etc.)
- `INTERVAL` - Time point (Prescreen, 0, 24, 96, 192, etc.)
- `EXTENSION` - File type (xlsx for Excel, stdf for STDF)

### Typical Workflow Example

**Scenario**: Analyze HAST stress test data from multiple lots over time intervals

1. **Prepare Files**: Ensure all test files follow naming convention
   ```
   Sample/
   ├── A104549HTA_A_HAST_Prescreen.xlsx
   ├── A104549HTB_A_HAST_0.xlsx
   ├── A104549HTC_A_HAST_96.xlsx
   ├── B104549HTA_B_HAST_Prescreen.xlsx
   └── ...
   ```

2. **Run Analysis**:
   ```bash
   python -m shiftanalysis.cli auto Sample --assume-yes
   ```

3. **Review Results**:
   - `DataWorkbook.xlsx` - Consolidated data and analysis
   - `shift_sources.json` - File metadata for reproducibility
   - Charts embedded in workbook sheets

4. **Generated Content**:
   - **Summary Sheet** - Overview of all processed files with pass/fail counts
   - **Measurements Sheets** - Detailed measurement data with test results
   - **Shift Calculations** - Statistical analysis of interval-to-interval changes  
   - **Histogram Charts** - Distribution visualizations by test and lot
   - **Boxplot Charts** - Statistical summary visualizations

## Application Architecture

### Core Components

```
shiftanalysis/
├── __init__.py              # Package initialization
├── cli.py                   # Main command-line interface
├── models.py                # Data structures and types
├── gui.py                   # GUI scaffolding (TBD)
├── stages/                  # Pipeline stage implementations
│   ├── __init__.py
│   ├── read_directory.py    # File discovery and metadata extraction
│   ├── convert_to_data.py   # Data conversion and consolidation
│   ├── calculate_shift.py   # Statistical shift analysis
│   └── generate_plot.py     # Chart generation orchestration
└── charts/                  # Visualization modules
    ├── __init__.py
    ├── boxplots.py          # Boxplot chart generation
    └── histograms.py        # Histogram chart generation
```

### Data Models

**SourceFile**: Represents a discovered input file
```python
@dataclass
class SourceFile:
    path: Path                    # Absolute file path
    lot: str                      # Extracted lot identifier
    event: str                    # Test event type
    interval: str                 # Time interval
    file_type: Literal["xlsx", "stdf"]  # File format
```

**SummaryRow**: Summary statistics per source file
```python
@dataclass  
class SummaryRow:
    lot: str                      # Lot identifier
    event: str                    # Test event
    interval: str                 # Time interval
    file_name: str                # Original filename
    source_path: Path             # Source file location
    pass_count: int               # Number of passing units
    fail_count: int               # Number of failing units
```

**MeasurementRow**: Individual test measurement
```python
@dataclass
class MeasurementRow:
    lot: str                      # Lot identifier
    event: str                    # Test event
    interval: str                 # Time interval
    source: str                   # Source file/sheet
    status: str                   # PASS/FAIL status
    test_number: str              # Test identifier
    test_name: str                # Human-readable test name
    test_unit: str                # Measurement unit
    low_limit: float | None       # Lower specification limit
    high_limit: float | None      # Upper specification limit
    measurement: float | str | None  # Actual measurement value
    serial_number: str            # Unit serial number
```

### Pipeline Stages

#### Stage 1: ReadDirectory
- **Purpose**: Discover source files and extract metadata
- **Input**: Directory path, file naming pattern
- **Output**: List of SourceFile objects, metadata JSON
- **Key Logic**: 
  - Regex pattern matching for filename parsing
  - User confirmation of discovered files
  - Metadata persistence for downstream stages

#### Stage 2: ConvertToData
- **Purpose**: Extract measurement data from source files
- **Input**: SourceFile list, output workbook path
- **Output**: Summary sheet, Measurements sheets
- **Key Logic**:
  - Excel workbook processing with pass/fail analysis
  - STDF file parsing via istdf library integration
  - Data normalization and consolidation
  - Excel table formatting and structure

#### Stage 3: CalculateShift
- **Purpose**: Compute statistical shifts between intervals
- **Input**: MeasurementRow data, workbook path
- **Output**: Shift calculation sheet in workbook
- **Key Logic**:
  - Interval ordering (Prescreen → numeric → alphabetic)
  - Mean, median, and standard deviation calculations
  - Min/max interval identification and comparison

#### Stage 4: GeneratePlot
- **Purpose**: Create histogram and boxplot visualizations
- **Input**: MeasurementRow data, workbook path, chart options
- **Output**: Chart sheets embedded in workbook
- **Key Logic**:
  - Chart layout by test name (rows) and lot (columns)
  - Color-coded interval grouping
  - Statistical annotations and limit markers

## Third-Party Components

This application incorporates several open-source libraries. See `THIRD_PARTY_LICENSES.txt` for complete license information.

### Core Dependencies

**openpyxl** (MIT License)
- **Purpose**: Excel file reading, writing, and manipulation
- **Usage**: Workbook processing, cell formatting, table creation
- **Copyright**: openpyxl contributors
- **Website**: https://openpyxl.readthedocs.io/

**matplotlib** (BSD-style License)  
- **Purpose**: Chart generation and plotting
- **Usage**: Histogram and boxplot creation, statistical visualization
- **Copyright**: Matplotlib Development Team
- **Website**: https://matplotlib.org/

**numpy** (BSD 3-Clause License)
- **Purpose**: Numerical computations and data processing
- **Usage**: Statistical calculations, array operations
- **Copyright**: NumPy Developers  
- **Website**: https://numpy.org/

### Bundled Components

**istdf** (License TBD)
- **Purpose**: STDF file parsing and data extraction
- **Location**: `Submodules/istdf/`
- **Usage**: Optional dependency for STDF file support
- **Note**: Verify upstream license before redistribution

### License Compliance

All third-party licenses are compatible with this project's usage. The application properly attributes dependencies and includes required license texts in `THIRD_PARTY_LICENSES.txt`. Users redistributing this software should review license requirements for their specific use case.

## GUI Scaffolding

The application includes foundational scaffolding for future graphical user interface development.

### Current GUI Structure

```python
# shiftanalysis/gui.py
class ShiftAnalysisGUI:
    """GUI scaffolding marked for future implementation"""
    
    def __init__(self, state: ApplicationState | None = None):
        self.state = state or ApplicationState()
    
    def launch(self) -> None:
        """TBD: Initialize and display GUI widgets"""
        raise NotImplementedError("TBD: GUI launch workflow")
        
    def shutdown(self) -> None:
        """TBD: Cleanup before application exit"""  
        raise NotImplementedError("TBD: GUI shutdown workflow")
```

### Planned GUI Features

**File Management**:
- Directory browser for source file selection
- Drag-and-drop file import
- File validation and naming convention checking

**Pipeline Control**:
- Visual pipeline progress indication  
- Individual stage execution controls
- Parameter configuration dialogs

**Data Visualization**:
- Interactive chart previews
- Chart customization options
- Export controls for visualizations

**Results Management**:
- Workbook preview and navigation
- Statistical summary displays
- Report generation options

### GUI Framework Considerations

**Recommended Frameworks**:
- **tkinter**: Built-in Python GUI toolkit (cross-platform)
- **PyQt/PySide**: Professional-grade framework with advanced widgets
- **wxPython**: Native look-and-feel across platforms
- **Kivy**: Modern touch-friendly interface

**Integration Points**:
- CLI module reuse for backend processing
- Model classes for data binding
- Chart modules for embedded visualization
- Configuration management for user preferences

## Development Notes

### Code Organization Principles

- **Modular Design**: Clear separation between CLI, data models, and processing stages
- **Type Safety**: Comprehensive type hints for maintainability
- **Error Handling**: Graceful failure modes with informative error messages
- **Testability**: Pure functions and dependency injection for unit testing

### Performance Considerations

- **Memory Management**: Streaming processing for large datasets
- **Excel Processing**: Optional formatting skip for performance (`--values-only`)
- **STDF Handling**: Efficient parsing with record filtering
- **Chart Generation**: Batch processing and image optimization

### Extending the Application

**Adding New File Formats**:
1. Extend `FileType` enum in `models.py`
2. Add parsing logic in `convert_to_data.py`
3. Update filename pattern in `read_directory.py`

**Adding New Chart Types**:
1. Create chart module in `charts/` directory
2. Implement chart generation function
3. Integrate with `generate_plot.py` orchestrator

**Adding New Statistical Calculations**:
1. Extend shift calculation logic in `calculate_shift.py`
2. Add new columns to output sheet structure
3. Update documentation and examples

### Testing & Validation

**Unit Testing** (Future Development):
- Individual stage function testing
- Data model validation
- File parsing accuracy verification

**Integration Testing**:
- End-to-end pipeline validation
- Sample file processing verification
- Chart generation accuracy

**Performance Testing**:
- Large dataset processing benchmarks
- Memory usage profiling
- Processing time optimization

### Debugging & Troubleshooting

**Common Issues**:

1. **File Not Found**: Verify file paths and naming convention compliance
2. **Permission Errors**: Check file access rights and directory permissions  
3. **Memory Issues**: Use `--values-only` flag for large Excel files
4. **Chart Generation Failures**: Verify matplotlib installation and display backend

**Debug Mode**:
```bash
# Enable verbose output (future enhancement)
python -m shiftanalysis.cli auto --verbose

# Check individual stages
python -m shiftanalysis.cli readdir --debug
```

**Log Files** (Future Enhancement):
- Processing logs for debugging
- Error tracking and reporting
- Performance metrics collection

## License

This project is licensed under the terms specified in the `LICENSE` file. Third-party components retain their original licenses as documented in `THIRD_PARTY_LICENSES.txt`.
