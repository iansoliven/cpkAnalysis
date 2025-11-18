# STDF Parsing Tool Generation Guide

**A comprehensive guide for building robust STDF parsers based on real-world production edge cases**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Architecture](#core-architecture)
3. [Data Quality & Validation](#data-quality--validation)
4. [Metadata Extraction](#metadata-extraction)
5. [Error Recovery & Resilience](#error-recovery--resilience)
6. [Edge Cases & Numeric Handling](#edge-cases--numeric-handling)
7. [Multi-File Processing](#multi-file-processing)
8. [Performance Optimization](#performance-optimization)
9. [Testing Strategy](#testing-strategy)
10. [Common Pitfalls](#common-pitfalls)

---

## Introduction

### Purpose

This guide consolidates lessons learned from developing a production STDF parser that processes millions of semiconductor test records. It addresses edge cases encountered in real-world ATE files from multiple vendors (Advantest, Teradyne, etc.) and provides proven solutions.

### Scope

- **STDF V4 Format**: Focus on parametric test records (PTR), functional test records (FTR), and metadata records
- **Production-Grade Requirements**: Error recovery, data integrity, performance, multi-vendor compatibility
- **Real-World Edge Cases**: Based on actual failures and fixes from production deployments

### Key Lessons

Over 20+ STDF-related bug fixes and enhancements have revealed critical insights:

1. **OPT_FLAG semantics** affect ~80% of production files (default limit preservation)
2. **Generator exhaustion** silently truncates data extraction
3. **Limit overwriting** from multiple PTR records destroys valid metadata
4. **Format strings** eliminate floating-point display artifacts in Excel/reporting
5. **NaN/Inf values** require special preservation through the pipeline
6. **Flag validation** (PARM_FLG, TEST_FLG) is essential for data quality
7. **Field name variations** exist across ATE vendors (OPT_FLG vs OPT_FLAG)

---

## Core Architecture

### Record-Level Processing

#### Best Practice: Manual Iteration for Error Recovery

**Problem**: Python generators become permanently dead after first exception.

**Impact**: A single corrupted record at position 3,710 stopped extraction at 47 devices instead of continuing to 1,205 devices.

**Solution**: Implement manual record iteration that survives exceptions:

```python
def robust_record_iterator(file_object):
    """
    Manually drive STDF record iteration with per-record error recovery.
    
    Unlike generators, this approach allows continuing after exceptions
    by re-entering the iteration loop with a fresh parser state.
    """
    parser = istdf.make_parser(file_object)
    
    while True:
        try:
            record = parser.read_record()
            if record is None:
                break
            yield record
        except StopIteration:
            break
        except Exception as e:
            # Log error but CONTINUE iteration
            logger.warning(f"Corrupted record: {e}")
            continue  # Parser seeks to next record automatically
```

**Why It Works**:
- Parser maintains internal file position state
- Exceptions don't kill iteration state (no generator exhaustion)
- Each `read_record()` call seeks to next valid record header
- Enables 100% data extraction even from files with corrupted records

**Testing Evidence**: Bhavik file extraction improved from 47 → 1,205 devices (25.6x increase).

---

### Metadata Caching Pattern

#### Rationale

STDF files often contain 1,000+ PTR records per device. Test metadata (limits, units, formats) appears in early PTRs, then subsequent PTRs rely on:
- OPT_FLAG bits 4/5 to "use default limit"
- Cached metadata for efficient lookup

#### Implementation

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class _TestMetadata:
    """Cache test metadata across devices."""
    test_name: str
    test_number: str
    unit: str = ""
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None
    has_low_limit: bool = False
    has_high_limit: bool = False
    result_format: Optional[str] = None
    lower_format: Optional[str] = None
    upper_format: Optional[str] = None
    result_scale: Optional[int] = None
    lower_scale: Optional[int] = None
    upper_scale: Optional[int] = None


# Global cache (per file parse)
test_metadata: Dict[str, _TestMetadata] = {}

def get_or_create_metadata(test_name: str, test_number: str) -> _TestMetadata:
    """Get cached metadata or create new entry."""
    key = f"{test_name}|{test_number}"
    if key not in test_metadata:
        test_metadata[key] = _TestMetadata(
            test_name=test_name,
            test_number=test_number,
        )
    return test_metadata[key]
```

**Benefits**:
- O(1) metadata lookup
- Automatic default propagation across devices
- Memory-efficient (one entry per unique test, not per measurement)

---

## Data Quality & Validation

### Flag-Based Filtering

#### PARM_FLG Validation

**Purpose**: Indicate measurement validity

**Critical Bits**:
```python
RESULT_INVALID = 0x01  # Bit 0: Result is unreliable or invalid
RESULT_UNRELIABLE = 0x02  # Bit 1: Result is questionable
NO_RESULT_VALUE = 0x10  # Bit 4: No result recorded
```

**Implementation**:
```python
def is_measurement_valid(parm_flg: int) -> bool:
    """
    Check if measurement should be included in analysis.
    
    REJECT if:
    - Bit 0 set (result invalid)
    - Bit 4 set (no result value)
    
    ACCEPT bit 1 (unreliable) as it often indicates marginal pass,
    which is valuable for statistical analysis.
    """
    if parm_flg & 0x01:  # RESULT_INVALID
        return False
    if parm_flg & 0x10:  # NO_RESULT_VALUE
        return False
    return True
```

**Rationale**: Bit 1 (unreliable) often represents "barely passed" measurements that are statistically important for Cpk analysis.

---

#### TEST_FLG Validation

**Purpose**: Indicate test execution status

**Critical Bits**:
```python
TEST_ABORTED = 0x01      # Bit 0: Test execution aborted
TEST_NO_EXECUTE = 0x02   # Bit 1: Test did not execute
TEST_TIMEOUT = 0x04      # Bit 2: Test timed out
TEST_NOT_EXECUTED = 0x08 # Bit 3: Test was not executed

TEST_FLG_REJECT_MASK = 0x0F  # Reject if any of above bits set
```

**Implementation**:
```python
def is_test_executed(test_flg: int) -> bool:
    """
    Check if test actually executed successfully.
    
    Reject measurements from tests that:
    - Aborted mid-execution
    - Never started
    - Timed out
    - Skipped for any reason
    """
    return (test_flg & 0x0F) == 0
```

**Impact**: ETS file showed 0% invalid measurements after proper TEST_FLG filtering. Without this, aborted tests pollute datasets.

---

### Combined Flag Validation

**Best Practice**: Apply both PARM_FLG and TEST_FLG checks

```python
def extract_measurement(ptr_record):
    """Extract measurement with comprehensive validation."""
    parm_flg = ptr_record.get("PARM_FLG", 0)
    test_flg = ptr_record.get("TEST_FLG", 0)
    
    # Early rejection based on flags
    if not is_measurement_valid(parm_flg):
        logger.debug(f"Rejected: PARM_FLG=0x{parm_flg:02X}")
        return None
        
    if not is_test_executed(test_flg):
        logger.debug(f"Rejected: TEST_FLG=0x{test_flg:02X}")
        return None
    
    # Proceed with measurement extraction
    # ...
```

**Logging**: Track rejection counts for debugging:
```python
invalid_measurements_count = 0

if measurement is None:
    invalid_measurements_count += 1
    
# At end of file:
logger.info(f"Invalid measurements filtered: {invalid_measurements_count}")
```

---

## Metadata Extraction

### OPT_FLAG Semantics (Critical)

**Impact**: ~80% of production STDF files rely on OPT_FLAG bits 4/5 for default limit preservation.

#### Bit Definitions

```
Bit 0 (0x01): Reserved
Bit 1 (0x02): Reserved
Bit 2 (0x04): Result scaled by RES_SCAL
Bit 3 (0x08): Reserved
Bit 4 (0x10): No low limit, use default     ← CRITICAL
Bit 5 (0x20): No high limit, use default    ← CRITICAL
Bit 6 (0x40): No low limit exists           ← CRITICAL
Bit 7 (0x80): No high limit exists          ← CRITICAL
```

#### Semantic Priority Rules

**Bits 6/7 override bits 4/5**:

```
If bit 6 set (0x40): "No low limit exists" → lo_limit = None, ignore bit 4
If bit 4 set (0x10): "Use default low limit" → preserve cached limit
Otherwise:            Use lo_limit field value
```

**Example Decision Tree**:
```
OPT_FLAG = 0x50 (bits 4 and 6 both set)

Bit 6 = 1 → "No low limit exists"
Bit 4 = 1 → "Use default low limit"

Resolution: Bit 6 takes priority → lo_limit = None
```

---

#### Implementation Pattern

```python
def process_limits_from_ptr(ptr_record, metadata):
    """
    Extract limits from PTR record using OPT_FLAG semantics.
    
    Priority order (high to low):
    1. Bit 6/7: Explicit "no limit exists"
    2. Bit 4/5: Use default (keep cached value)
    3. Field value: Update to new explicit limit
    """
    # Handle both field name variants (ATE vendor compatibility)
    opt_flg = ptr_record.get("OPT_FLG", ptr_record.get("OPT_FLAG", 0))
    
    # Handle byte object from binary readers
    if isinstance(opt_flg, bytes):
        opt_flg = int.from_bytes(opt_flg, byteorder='little')
    
    raw_low_limit = ptr_record.get("LO_LIMIT")
    raw_high_limit = ptr_record.get("HI_LIMIT")
    
    # LOW LIMIT PROCESSING
    no_low_limit = bool(opt_flg & 0x40)      # Bit 6
    use_default_low = bool(opt_flg & 0x10)   # Bit 4
    
    if no_low_limit:
        # Explicit "no limit exists"
        metadata.low_limit = None
        metadata.has_low_limit = False
    elif use_default_low:
        # Use default: keep existing cached value
        pass
    elif raw_low_limit is not None:
        # Explicit new limit
        metadata.low_limit = raw_low_limit
        metadata.has_low_limit = True
    else:
        # No limit in this record, no flag set → clear
        metadata.low_limit = None
        metadata.has_low_limit = False
    
    # HIGH LIMIT PROCESSING (mirror logic)
    no_high_limit = bool(opt_flg & 0x80)     # Bit 7
    use_default_high = bool(opt_flg & 0x20)  # Bit 5
    
    if no_high_limit:
        metadata.high_limit = None
        metadata.has_high_limit = False
    elif use_default_high:
        pass
    elif raw_high_limit is not None:
        metadata.high_limit = raw_high_limit
        metadata.has_high_limit = True
    else:
        metadata.high_limit = None
        metadata.has_high_limit = False
```

---

#### Testing OPT_FLAG Handling

**Test 1: Bit 4 preserves default low limit**

```python
def test_opt_flag_bit4_preserves_default_low_limit():
    """OPT_FLAG bit 4 should preserve default, not use garbage value."""
    cache = {}
    catalog = {}
    
    # Establish default limit: 1.0
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=1.0, opt_flg=0x00),
        cache, catalog
    )
    
    # Send garbage value (999.0) with bit 4 set
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=999.0, opt_flg=0x10),
        cache, catalog
    )
    
    # Verify: Default preserved, garbage ignored
    assert catalog[key]["stdf_lower"] == pytest.approx(1.0)
    assert cache[key].low_limit == pytest.approx(1.0)
```

**Test 2: Bit 6 clears limit**

```python
def test_opt_flag_bit6_clears_low_limit():
    """OPT_FLAG bit 6 should explicitly clear limit."""
    cache = {}
    catalog = {}
    
    # Establish limit: 1.0
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=1.0, opt_flg=0x00),
        cache, catalog
    )
    
    # Clear with bit 6
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=None, opt_flg=0x40),
        cache, catalog
    )
    
    # Verify: Limit cleared
    assert catalog[key]["stdf_lower"] is None
    assert cache[key].has_low_limit is False
```

**Test 3: Bits 4 and 6 both set (bit 6 wins)**

```python
def test_opt_flag_bit6_overrides_bit4():
    """When both bits set, bit 6 (no limit) takes priority."""
    cache = {}
    catalog = {}
    
    # Establish default: 1.0
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=1.0, opt_flg=0x00),
        cache, catalog
    )
    
    # Set both bit 4 and bit 6 (0x50)
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=999.0, opt_flg=0x50),
        cache, catalog
    )
    
    # Verify: Bit 6 wins → limit is None
    assert catalog[key]["stdf_lower"] is None
    assert cache[key].has_low_limit is False
```

**Real-World Impact**: After fixing OPT_FLAG handling, ETS file extraction improved from 0 → 265 tests with valid limits.

---

### Format String Extraction

**Problem**: Floating-point artifacts in Excel/reports

```
Expected:  0.04
Displayed: 0.03999999898951501
```

**Root Cause**: IEEE 754 floating-point representation + default Excel formatting

**Solution**: Extract STDF format strings and convert to Excel format codes

---

#### STDF Format String Fields

```python
# PTR record contains three format strings
C_RESFMT: str   # Result value format (e.g., "%8.3f")
C_LLMFMT: str   # Low limit format (e.g., "%8.3f")
C_HLMFMT: str   # High limit format (e.g., "%8.3f")
```

**Example PTR Record**:
```
TEST_NUM: 100
TEST_TXT: "VDD_CORE"
RESULT:   1.234567
LO_LIMIT: 1.0
HI_LIMIT: 2.0
C_RESFMT: "%8.3f"    ← Format string
C_LLMFMT: "%8.3f"
C_HLMFMT: "%8.3f"
```

---

#### Extraction Logic

```python
def extract_format_strings(ptr_record, metadata):
    """
    Extract format strings from PTR record.
    
    These format strings represent the ATE's intended display precision
    and eliminate floating-point artifacts in downstream tools.
    """
    c_resfmt = ptr_record.get("C_RESFMT")
    c_llmfmt = ptr_record.get("C_LLMFMT")
    c_hlmfmt = ptr_record.get("C_HLMFMT")
    
    # Only update if provided and non-empty
    if c_resfmt:
        metadata.result_format = c_resfmt.strip()
    if c_llmfmt:
        metadata.lower_format = c_llmfmt.strip()
    if c_hlmfmt:
        metadata.upper_format = c_hlmfmt.strip()
```

---

#### STDF → Excel Format Conversion

**STDF Format Syntax** (C printf-style):
```
"%8.3f"   → 8 characters wide, 3 decimal places, float
"%12.6e"  → 12 characters wide, 6 decimals, scientific notation
"%5.0f"   → 5 characters wide, 0 decimals (integer display)
```

**Excel Format Codes**:
```
"0.000"       → 3 decimal places
"0.000000"    → 6 decimal places
"0"           → Integer (no decimals)
"0.00E+00"    → Scientific notation
```

**Conversion Implementation**:
```python
import re

def stdf_format_to_excel(stdf_format: str) -> str:
    """
    Convert STDF printf-style format to Excel number format.
    
    Examples:
        "%8.3f"  → "0.000"
        "%12.6f" → "0.000000"
        "%5.0f"  → "0"
        "%12.3e" → "0.000E+00"
    """
    if not stdf_format:
        return "0.0000"  # Default: 4 decimals
    
    # Match printf format: %[width].[precision][type]
    match = re.match(r'%(\d+)?\.(\d+)([eEfFgG])', stdf_format)
    if not match:
        return "0.0000"
    
    width, precision, fmt_type = match.groups()
    precision = int(precision)
    
    # Scientific notation
    if fmt_type in ('e', 'E'):
        if precision == 0:
            return "0E+00"
        return "0." + ("0" * precision) + "E+00"
    
    # Fixed-point notation
    if precision == 0:
        return "0"
    return "0." + ("0" * precision)


# Example usage in workbook builder
def apply_cell_format(worksheet, row, col, value, format_string):
    """Apply Excel format to cell."""
    excel_format = stdf_format_to_excel(format_string)
    
    cell = worksheet.cell(row=row, column=col)
    cell.value = value
    cell.number_format = excel_format
```

**Result**:
```
Before:  0.03999999898951501  (default Excel formatting)
After:   0.040                (C_RESFMT="%6.3f" → "0.000")
```

---

#### Format Priority System

**Priority Order** (highest to lowest):

1. **Format String**: If `C_RESFMT` exists, use it
2. **Scale Factor**: If `RES_SCAL` exists, calculate decimals (e.g., -3 → µ → 6 decimals)
3. **User Preference**: User-configured default (if specified)
4. **Fallback**: 4 decimal places

**Implementation**:
```python
def determine_number_format(metadata, user_default=None):
    """Determine Excel number format using priority system."""
    
    # Priority 1: Format string
    if metadata.result_format:
        return stdf_format_to_excel(metadata.result_format)
    
    # Priority 2: Scale factor
    if metadata.result_scale is not None:
        # Scale factor indicates unit multiplier
        # -3 → milli → 3 decimals
        # -6 → micro → 6 decimals
        # -9 → nano → 9 decimals
        decimals = abs(metadata.result_scale)
        return f"0.{'0' * decimals}"
    
    # Priority 3: User preference
    if user_default:
        return user_default
    
    # Priority 4: Fallback
    return "0.0000"
```

**Real-World Impact**: Eliminated floating-point display issues in 100% of Excel reports. Test engineers no longer question data precision.

---

### Field Name Variations

**Problem**: Different ATE vendors use different field names for the same data

**Example**: OPT_FLAG field

```
ETS Files (Advantest):  "OPT_FLG"
T2K Files (Teradyne):   "OPT_FLAG"
```

**Impact**: T2K files showed `0.0` limits instead of blank cells for tests with OPT_FLAG=0xCE (bits 6&7 set).

**Solution**: Check both field names

```python
def get_opt_flag(ptr_record):
    """
    Get OPT_FLAG value supporting multiple field name variants.
    
    Different ATE vendors use different field names:
    - Advantest: OPT_FLG
    - Teradyne: OPT_FLAG
    """
    opt_flg = ptr_record.get("OPT_FLG", ptr_record.get("OPT_FLAG", 0))
    
    # Handle byte objects from binary STDF readers
    if isinstance(opt_flg, bytes):
        opt_flg = int.from_bytes(opt_flg, byteorder='little')
    
    return opt_flg
```

**Best Practice**: For any critical field, check known variants:
- OPT_FLG / OPT_FLAG
- TEST_TXT / TEST_NAM
- UNIT / UNITS

**Testing**: Include STDF files from multiple ATE vendors in test suite.

---

## Error Recovery & Resilience

### Generator Exhaustion (Critical)

**Problem**: Generators die on first exception

**Impact**: Single corrupted record stops extraction (47/1,205 devices = 3.9%)

**Solution**: Manual iteration with `robust_record_iterator()`

**Result**: 100% device extraction from corrupted files

**Code Pattern**:
```python
def parse_stdf_file(file_path):
    """Parse STDF file with error recovery."""
    with open(file_path, 'rb') as f:
        record_count = 0
        error_count = 0
        
        for record in robust_record_iterator(f):
            record_count += 1
            try:
                process_record(record)
            except Exception as e:
                error_count += 1
                logger.warning(f"Record {record_count} error: {e}")
                continue
        
        logger.info(f"Processed {record_count} records, {error_count} errors")
```

**Validation**: Compare device count to commercial tools (SEDana, WinSTDF)

---

### Corrupted Record Handling

**Common Corruption Types**:

1. **Malformed FTR Records**: Repeat count field corrupted
   ```
   Expected: VECT_NAM repeated RTN_ICNT times
   Reality:  RTN_ICNT = 796 (should be 1-8)
   ```

2. **Truncated Records**: File ends mid-record
   ```
   Expected: 1024 bytes
   Reality:  512 bytes (power loss during write)
   ```

3. **Invalid Record Types**: Unknown REC_TYP value
   ```
   Expected: REC_TYP in [0, 1, 2, 5, 10, 15, 20, 50]
   Reality:  REC_TYP = 255 (corruption or new STDF version)
   ```

**Handling Strategy**:

```python
def robust_record_iterator(file_object):
    """Iterate STDF records with per-record error recovery."""
    parser = istdf.make_parser(file_object)
    
    while True:
        try:
            record = parser.read_record()
            if record is None:
                break
            yield record
            
        except istdf.ParseError as e:
            # STDF parser detected malformed record
            logger.warning(f"ParseError: {e}")
            # Parser automatically seeks to next record header
            continue
            
        except struct.error as e:
            # Binary unpacking failed (truncated data)
            logger.warning(f"struct.error: {e}")
            continue
            
        except EOFError:
            # Unexpected end of file
            logger.warning("Unexpected EOF, file may be truncated")
            break
            
        except Exception as e:
            # Catch-all for unknown errors
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            continue
```

**Logging**: Track error statistics

```python
error_stats = defaultdict(int)

for record in robust_record_iterator(f):
    try:
        process_record(record)
    except Exception as e:
        error_stats[type(e).__name__] += 1

# Report at end
for error_type, count in sorted(error_stats.items()):
    logger.info(f"{error_type}: {count} occurrences")
```

---

### Partial Data Extraction

**Goal**: Extract maximum data even when file is corrupted

**Pattern**: Continue extraction, mark incomplete

```python
@dataclass
class ParseResult:
    """Result of STDF file parsing."""
    measurements: pd.DataFrame
    test_catalog: pd.DataFrame
    is_complete: bool  # False if errors encountered
    error_count: int
    record_count: int
    device_count: int


def parse_stdf_file(file_path):
    """Parse STDF with partial extraction capability."""
    errors = []
    measurements = []
    
    for record in robust_record_iterator(f):
        try:
            if isinstance(record, istdf.PTR):
                measurement = extract_measurement(record)
                if measurement:
                    measurements.append(measurement)
        except Exception as e:
            errors.append(str(e))
    
    return ParseResult(
        measurements=pd.DataFrame(measurements),
        test_catalog=build_catalog(),
        is_complete=(len(errors) == 0),
        error_count=len(errors),
        record_count=total_records,
        device_count=len(set(m['serial'] for m in measurements)),
    )
```

**User Communication**:
```
⚠ Warning: STDF file partially corrupted
  Extracted: 1,205 / 1,205 devices (100%)
  Errors: 796 corrupted FTR records (skipped)
  Analysis: Continuing with available data
```

---

## Edge Cases & Numeric Handling

### NaN and Inf Values

**Problem**: Outlier filtering silently removed NaN and Inf values

**Root Cause**: Comparison operations with non-finite values return `False`

```python
import numpy as np

value = np.nan
lower = 1.0
upper = 2.0

# This evaluates to False (removes NaN)
mask = (value >= lower) & (value <= upper)  # False
```

**Impact**: Valid edge cases (sensor saturation, divide-by-zero) removed from analysis

---

#### Solution: Explicit Non-Finite Preservation

```python
def filter_outliers_iqr(values, k=1.5):
    """
    IQR outlier filtering with NaN/Inf preservation.
    
    Rationale:
    - NaN and Inf have passed STDF validation (TEST_FLG/PARM_FLG checks)
    - They represent valid edge cases (saturation, divide-by-zero)
    - Outlier filtering should only remove statistical outliers
    """
    # Separate finite and non-finite values
    is_finite = np.isfinite(values)
    finite_values = values[is_finite]
    
    # Calculate IQR on finite values only
    q1 = np.percentile(finite_values, 25)
    q3 = np.percentile(finite_values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Include values within bounds OR non-finite
    is_within_bounds = (values >= lower_bound) & (values <= upper_bound)
    mask = is_within_bounds | ~is_finite
    
    return values[mask]
```

**Testing**:
```python
def test_outlier_filtering_preserves_nan():
    """NaN values should be preserved."""
    data = pd.DataFrame({
        'file': ['f1'] * 10,
        'test': ['t1'] * 10,
        'value': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
    })
    
    result = filter_outliers(data, method='iqr', k=1.5)
    
    # Verify NaN preserved
    assert result['value'].isna().sum() == 1


def test_outlier_filtering_preserves_inf():
    """Inf values should be preserved."""
    data = pd.DataFrame({
        'file': ['f1'] * 10,
        'test': ['t1'] * 10,
        'value': [1, 2, 3, np.inf, 5, 6, 7, -np.inf, 9, 10],
    })
    
    result = filter_outliers(data, method='iqr', k=1.5)
    
    # Verify Inf preserved
    assert np.isinf(result['value']).sum() == 2
```

**Real-World Impact**: ETS file analysis showed 0% data loss from non-finite values after fix.

---

### None vs Missing Values

**Problem**: Distinguishing between "no limit exists" and "limit not yet seen"

**Example**:
```python
# Case 1: No limit exists (OPT_FLAG bit 6 set)
metadata.low_limit = None
metadata.has_low_limit = False

# Case 2: Limit not yet seen (early in file)
metadata.low_limit = None  
metadata.has_low_limit = False  # But might be updated later
```

**Solution**: Use separate flag to track "has limit" state

```python
@dataclass
class _TestMetadata:
    low_limit: Optional[float] = None
    has_low_limit: bool = False  # Explicit tracking
    
    
# When processing PTR record
if opt_flg & 0x40:  # Bit 6: no low limit exists
    metadata.low_limit = None
    metadata.has_low_limit = False  # EXPLICIT
    
elif opt_flg & 0x10:  # Bit 4: use default
    pass  # Keep existing value
    
elif raw_low_limit is not None:
    metadata.low_limit = raw_low_limit
    metadata.has_low_limit = True  # EXPLICIT
```

**Benefit**: Prevents `None` from overwriting valid limits during multi-file merging

---

### Zero Variance Edge Case

**Problem**: Standard deviation outlier filtering fails on constant values

```python
values = [5.0, 5.0, 5.0, 5.0, 5.0]
std = np.std(values)  # 0.0
threshold = mean + k * std  # mean + 0 = mean
```

**Solution**: Handle zero variance explicitly

```python
def filter_outliers_std(values, k=3):
    """Standard deviation filtering with zero variance handling."""
    mean = np.nanmean(values)
    std = np.nanstd(values)
    
    # Zero variance: all values identical, no outliers
    if std == 0:
        return values
    
    lower_bound = mean - k * std
    upper_bound = mean + k * std
    
    is_finite = np.isfinite(values)
    is_within_bounds = (values >= lower_bound) & (values <= upper_bound)
    mask = is_within_bounds | ~is_finite
    
    return values[mask]
```

**Testing**:
```python
def test_zero_variance_no_removal():
    """Zero variance groups should have no outliers removed."""
    data = pd.DataFrame({
        'file': ['f1'] * 5,
        'test': ['t1'] * 5,
        'value': [5.0, 5.0, 5.0, 5.0, 5.0],
    })
    
    result = filter_outliers(data, method='std', k=3)
    
    # All values preserved
    assert len(result) == 5
```

---

## Multi-File Processing

### Catalog Merging

**Problem**: Multiple STDF files for same test suite need unified catalog

**Challenge**: Later files may have different limits or "no limit" flags

---

#### Merging Logic

```python
def merge_catalogs(existing, new):
    """
    Merge test catalogs from multiple files.
    
    Rules:
    1. Preserve non-None limits (don't overwrite with None)
    2. Honor explicit "no limit" flags (bits 6/7)
    3. Propagate units across files
    4. Preserve format strings
    """
    for key, new_info in new.items():
        if key not in existing:
            # New test, add to catalog
            existing[key] = new_info.copy()
            continue
        
        # Existing test, merge carefully
        existing_info = existing[key]
        
        # LOW LIMIT MERGING
        new_has_low = new_info.get("has_stdf_lower", False)
        new_low_value = new_info.get("stdf_lower")
        
        if new_has_low and new_low_value is not None:
            # Explicit new limit, update
            existing_info["stdf_lower"] = new_low_value
            existing_info["has_stdf_lower"] = True
            
        elif new_has_low is False and new_low_value is None:
            # Explicit "no limit" (bit 6 set)
            existing_info["stdf_lower"] = None
            existing_info["has_stdf_lower"] = False
            
        # else: keep existing value
        
        # HIGH LIMIT MERGING (mirror logic)
        # ...
        
        # UNIT PROPAGATION
        if new_info.get("unit") and not existing_info.get("unit"):
            existing_info["unit"] = new_info["unit"]
        
        # FORMAT STRING PROPAGATION
        if new_info.get("stdf_result_format"):
            existing_info["stdf_result_format"] = new_info["stdf_result_format"]
```

**Real-World Example**: ETS file analysis

```
File 1: Test "VDD" has lo_limit=1.0, hi_limit=2.0
File 2: Test "VDD" has lo_limit=None (OPT_FLAG=0x40), hi_limit=2.0

Before fix: Final catalog showed lo_limit=None (File 2 overwrote File 1)
After fix:  Final catalog shows lo_limit=1.0 (preserved from File 1)
```

---

### Per-File Statistics

**Purpose**: Track parsing success per file for debugging

```python
@dataclass
class FileStats:
    """Statistics for a single STDF file."""
    file_path: str
    device_count: int
    measurement_count: int
    invalid_measurement_count: int
    corrupted_record_count: int
    parse_time_seconds: float
    is_complete: bool


def parse_stdf_file(file_path):
    """Parse with per-file statistics tracking."""
    start_time = time.time()
    
    device_count = 0
    measurement_count = 0
    invalid_count = 0
    corrupted_count = 0
    
    # Parse file...
    for record in robust_record_iterator(f):
        try:
            if isinstance(record, istdf.PTR):
                measurement = extract_measurement(record)
                if measurement:
                    measurement_count += 1
                else:
                    invalid_count += 1
        except Exception:
            corrupted_count += 1
    
    return FileStats(
        file_path=file_path,
        device_count=device_count,
        measurement_count=measurement_count,
        invalid_measurement_count=invalid_count,
        corrupted_record_count=corrupted_count,
        parse_time_seconds=time.time() - start_time,
        is_complete=(corrupted_count == 0),
    )
```

**Output Example**:
```
File: ETS124728.stdf
  Devices:      200
  Measurements: 52,400
  Invalid:      0 (0.0%)
  Corrupted:    0
  Parse time:   1.23s
  Status:       ✓ Complete

File: Bhavik.stdf
  Devices:      1,205
  Measurements: 316,310
  Invalid:      0 (0.0%)
  Corrupted:    796 (FTR records)
  Parse time:   4.56s
  Status:       ⚠ Partial (continued past errors)
```

---

### Site Description Tracking

**Purpose**: Track test site configurations across files

```python
@dataclass
class SiteDescription:
    """Description of a test site from SDR record."""
    site_number: int
    handler_type: str
    handler_id: str
    probe_card: str
    load_board: str
    dib_board: str
    cable_id: str
    contactor_id: str
    laser_id: str
    extr_type: str


def extract_site_description(sdr_record):
    """Extract site description from SDR record."""
    return SiteDescription(
        site_number=sdr_record.get("SITE_NUM", 0),
        handler_type=sdr_record.get("HAND_TYP", ""),
        handler_id=sdr_record.get("HAND_ID", ""),
        probe_card=sdr_record.get("CARD_TYP", ""),
        load_board=sdr_record.get("LOAD_TYP", ""),
        dib_board=sdr_record.get("DIB_TYP", ""),
        cable_id=sdr_record.get("CABL_ID", ""),
        contactor_id=sdr_record.get("CONT_ID", ""),
        laser_id=sdr_record.get("LASR_ID", ""),
        extr_type=sdr_record.get("EXTR_TYP", ""),
    )


# Deduplication across files
all_site_descriptions = []
unique_sites = tuple(dict.fromkeys(all_site_descriptions))
```

**Use Case**: Identify multi-site test configurations, correlate failures to hardware

---

## Performance Optimization

### Memory Efficiency

**Problem**: Large STDF files (1M+ measurements) cause memory issues

**Strategies**:

#### 1. Columnar Storage

```python
# Bad: List of dictionaries (high memory overhead)
measurements = [
    {"serial": "A1", "test": "VDD", "value": 1.2},
    {"serial": "A1", "test": "GND", "value": 0.0},
    # ... 1 million rows
]

# Good: Columnar dict (pandas-friendly)
column_data = {
    "serial": [],
    "test": [],
    "value": [],
}

for record in robust_record_iterator(f):
    measurement = extract_measurement(record)
    if measurement:
        column_data["serial"].append(measurement["serial"])
        column_data["test"].append(measurement["test"])
        column_data["value"].append(measurement["value"])

# Convert to DataFrame once at end
df = pd.DataFrame(column_data)
```

**Memory Savings**: ~60% reduction for large files

---

#### 2. Parquet Storage for Raw Data

```python
import pyarrow.parquet as pq
import pyarrow as pa

# Convert to Arrow table and write as Parquet
table = pa.Table.from_pandas(df)
pq.write_table(table, "raw_data.parquet", compression="snappy")

# Benefits:
# - 10x compression vs CSV
# - Column-oriented (fast filtering)
# - Schema preservation (types, metadata)
# - Interoperable (R, Julia, Spark)
```

---

#### 3. Incremental DataFrame Construction

```python
# For extremely large files, build incrementally
batch_size = 10000
batches = []

for i, record in enumerate(robust_record_iterator(f)):
    measurement = extract_measurement(record)
    if measurement:
        column_data["serial"].append(measurement["serial"])
        # ...
    
    # Every 10k records, convert to DataFrame and clear
    if i % batch_size == 0:
        batches.append(pd.DataFrame(column_data))
        column_data = {col: [] for col in EXPECTED_COLUMNS}

# Combine all batches
df = pd.concat(batches, ignore_index=True)
```

---

### Processing Speed

**Optimization Checklist**:

1. **Avoid repeated dict lookups**:
   ```python
   # Bad
   value = record.get("RESULT")
   flags = record.get("TEST_FLG")
   # ... use 'value' and 'flags' 10 times
   
   # Good
   result = record.get("RESULT")
   test_flg = record.get("TEST_FLG")
   parm_flg = record.get("PARM_FLG")
   # Cache locally, single lookup
   ```

2. **Pre-compile regex**:
   ```python
   # Module level
   FORMAT_REGEX = re.compile(r'%(\d+)?\.(\d+)([eEfFgG])')
   
   # In function
   match = FORMAT_REGEX.match(stdf_format)
   ```

3. **Use sets for membership testing**:
   ```python
   # Bad
   valid_tests = ["VDD", "GND", "ILOAD", ...]
   if test_name in valid_tests:  # O(n) lookup
   
   # Good
   VALID_TESTS = {"VDD", "GND", "ILOAD", ...}
   if test_name in VALID_TESTS:  # O(1) lookup
   ```

4. **Batch pandas operations**:
   ```python
   # Bad: Row-by-row updates
   for i in range(len(df)):
       df.loc[i, 'scaled'] = df.loc[i, 'value'] * 1000
   
   # Good: Vectorized operation
   df['scaled'] = df['value'] * 1000
   ```

---

## Testing Strategy

### Test Pyramid

```
        /\
       /UI\          Small: Integration tests (multi-file)
      /----\
     /Unit \         Medium: Module tests (single file)
    /------\
   /  Edge  \        Large: Edge case tests (corrupted data)
  /----------\
```

---

### Unit Tests (70%)

**Focus**: Individual functions, isolated behavior

```python
def test_opt_flag_bit4_preserves_default_low_limit():
    """Test OPT_FLAG bit 4 semantics."""
    cache = {}
    catalog = {}
    
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=1.0, opt_flg=0x00),
        cache, catalog
    )
    
    populate_test_catalog_from_ptr(
        build_ptr_record(lo_limit=999.0, opt_flg=0x10),
        cache, catalog
    )
    
    assert catalog[key]["stdf_lower"] == pytest.approx(1.0)
```

**Coverage Goals**:
- All OPT_FLAG bit combinations (16 tests: 2^4 bits)
- All flag validation paths (PARM_FLG, TEST_FLG)
- Format string conversion (20+ examples)
- NaN/Inf preservation (8 tests)
- Zero variance edge case

---

### Integration Tests (20%)

**Focus**: Full file parsing, multi-file merging

```python
def test_parse_multi_site_stdf():
    """Test parsing STDF with multiple sites."""
    result = parse_stdf_files([
        "test_data/site1.stdf",
        "test_data/site2.stdf",
    ])
    
    assert len(result.site_descriptions) == 2
    assert result.frame["site"].nunique() == 2
    assert result.is_complete is True


def test_catalog_merging_preserves_limits():
    """Test multi-file catalog merging."""
    result = parse_stdf_files([
        "test_data/file1_with_limits.stdf",
        "test_data/file2_no_limits.stdf",
    ])
    
    # Limits from file1 should be preserved
    vdd_test = result.test_catalog[
        result.test_catalog["test_name"] == "VDD"
    ]
    assert vdd_test["stdf_lower"].iloc[0] == pytest.approx(1.0)
```

---

### Edge Case Tests (10%)

**Focus**: Corrupted data, boundary conditions

```python
def test_corrupted_ftr_records_recoverable():
    """Test parsing file with 796 corrupted FTR records."""
    result = parse_stdf_file("test_data/bhavik_corrupted.stdf")
    
    # Should extract all 1,205 devices despite corruption
    assert result.device_count == 1205
    assert result.corrupted_record_count == 796
    assert result.is_complete is False  # Has errors


def test_truncated_file_partial_extraction():
    """Test parsing truncated STDF file."""
    result = parse_stdf_file("test_data/truncated.stdf")
    
    # Should extract partial data
    assert result.device_count > 0
    assert result.is_complete is False
```

---

### Validation Tests

**Purpose**: Compare to commercial tools (SEDana, WinSTDF)

```python
def test_device_count_matches_sedana():
    """Validate device count against SEDana."""
    # Parse with our tool
    result = parse_stdf_file("test_data/reference.stdf")
    
    # Compare to SEDana export (CSV)
    sedana_df = pd.read_csv("test_data/reference_sedana.csv")
    sedana_device_count = sedana_df["Serial"].nunique()
    
    assert result.device_count == sedana_device_count


def test_limit_extraction_matches_manual_inspection():
    """Validate limits against manual STDF inspection."""
    result = parse_stdf_file("test_data/known_limits.stdf")
    
    # Known from manual inspection of STDF
    expected_limits = {
        ("VDD_CORE", "100"): (0.95, 1.05),
        ("GND", "101"): (0.0, 0.05),
        ("ILOAD", "102"): (None, 50.0),  # No low limit
    }
    
    for (test_name, test_num), (exp_low, exp_high) in expected_limits.items():
        row = result.test_catalog[
            (result.test_catalog["test_name"] == test_name) &
            (result.test_catalog["test_number"] == test_num)
        ]
        
        if exp_low is None:
            assert pd.isna(row["stdf_lower"].iloc[0])
        else:
            assert row["stdf_lower"].iloc[0] == pytest.approx(exp_low)
        
        if exp_high is None:
            assert pd.isna(row["stdf_upper"].iloc[0])
        else:
            assert row["stdf_upper"].iloc[0] == pytest.approx(exp_high)
```

---

## Common Pitfalls

### Pitfall 1: Using Generators for Error Recovery

**Problem**: Generators die on first exception

```python
# BAD: Generator exhaustion
def parse_records(file_object):
    parser = istdf.make_parser(file_object)
    for record in parser:  # Generator
        yield record
    # After ANY exception, generator is dead

# GOOD: Manual iteration
def parse_records(file_object):
    parser = istdf.make_parser(file_object)
    while True:
        try:
            record = parser.read_record()
            if record is None:
                break
            yield record
        except Exception as e:
            logger.warning(f"Error: {e}")
            continue  # Keep going
```

**Impact**: 25.6x data extraction improvement (47 → 1,205 devices)

---

### Pitfall 2: Ignoring OPT_FLAG Bits 4/5

**Problem**: Default limits not preserved

```python
# BAD: Always use field value
low_limit = ptr_record.get("LO_LIMIT")

# GOOD: Check OPT_FLAG
opt_flg = get_opt_flag(ptr_record)
if opt_flg & 0x10:  # Bit 4: use default
    pass  # Keep cached value
else:
    low_limit = ptr_record.get("LO_LIMIT")
```

**Impact**: ~80% of production files affected, 0 → 265 limits captured

---

### Pitfall 3: Overwriting Valid Limits with None

**Problem**: Later records clear earlier valid limits

```python
# BAD: Always update
existing["stdf_lower"] = new["stdf_lower"]  # Might be None

# GOOD: Defensive merging
if new["has_stdf_lower"] and new["stdf_lower"] is not None:
    existing["stdf_lower"] = new["stdf_lower"]
```

**Impact**: Multi-file analysis loses limit data

---

### Pitfall 4: Removing NaN/Inf During Outlier Filtering

**Problem**: Comparison with non-finite values returns False

```python
# BAD: Removes NaN/Inf
mask = (values >= lower) & (values <= upper)

# GOOD: Preserve non-finite
is_finite = np.isfinite(values)
is_within_bounds = (values >= lower) & (values <= upper)
mask = is_within_bounds | ~is_finite
```

**Impact**: Valid edge cases lost from analysis

---

### Pitfall 5: Ignoring Field Name Variants

**Problem**: ATE vendor differences

```python
# BAD: Only checks one name
opt_flg = ptr_record.get("OPT_FLG", 0)

# GOOD: Check variants
opt_flg = ptr_record.get("OPT_FLG", ptr_record.get("OPT_FLAG", 0))
```

**Impact**: T2K files show incorrect limits

---

### Pitfall 6: Using Default Floating-Point Formatting

**Problem**: Excel shows artifacts (0.03999999898951501)

```python
# BAD: No format control
cell.value = 0.04

# GOOD: Extract and apply format string
format_str = ptr_record.get("C_RESFMT", "%6.3f")
excel_format = stdf_format_to_excel(format_str)
cell.number_format = excel_format
cell.value = 0.04  # Displays as "0.040"
```

**Impact**: Test engineers question data precision

---

### Pitfall 7: Not Tracking Parse Statistics

**Problem**: Silent failures unnoticed

```python
# BAD: No tracking
df = parse_stdf_file(file_path)

# GOOD: Track and report
result = parse_stdf_file(file_path)
logger.info(f"Devices: {result.device_count}")
logger.info(f"Errors: {result.error_count}")
if not result.is_complete:
    logger.warning("⚠ File had errors, check logs")
```

**Impact**: Incomplete data used for critical decisions

---

### Pitfall 8: Batch Operations on None Values

**Problem**: pandas operations fail on None

```python
# BAD: Direct operation
df["scaled"] = df["value"] * 1000  # Fails if value=None

# GOOD: Handle None explicitly
df["scaled"] = df["value"].fillna(0) * 1000
# Or
df["scaled"] = df["value"].apply(
    lambda x: x * 1000 if x is not None else None
)
```

---

## Conclusion

### Key Takeaways

1. **Error Recovery is Critical**: Use manual iteration, not generators
2. **OPT_FLAG Semantics Matter**: ~80% of files rely on bits 4/5/6/7
3. **Defensive Merging**: Don't overwrite valid data with None
4. **Format Strings**: Extract and apply for professional output
5. **Preserve Edge Cases**: NaN/Inf are valid, not errors
6. **Flag Validation**: PARM_FLG and TEST_FLG ensure data quality
7. **Multi-Vendor Support**: Check field name variants
8. **Track Statistics**: Report parse success, errors, completeness

### Production Readiness Checklist

- [ ] Manual iteration for error recovery
- [ ] OPT_FLAG bits 4/5/6/7 handling
- [ ] PARM_FLG and TEST_FLG validation
- [ ] Format string extraction and conversion
- [ ] NaN/Inf preservation in outlier filtering
- [ ] Field name variant support (OPT_FLG/OPT_FLAG)
- [ ] Defensive catalog merging (preserve non-None)
- [ ] Per-file statistics tracking
- [ ] Corrupted record logging
- [ ] Validation against commercial tools
- [ ] Comprehensive test suite (unit + integration + edge cases)
- [ ] Memory-efficient columnar storage
- [ ] Parquet export for raw data

### Further Reading

- **STDF V4 Specification**: Official format documentation from your ATE vendor or search for "STDF V4 specification PDF"

---

## Appendix: Real-World Case Studies

### Case Study 1: Bhavik File (Generator Exhaustion)

**Symptoms**:
- Parser stopped at record #3,710
- Extracted 47 devices instead of 1,205
- SEDana extracted all 1,205 devices

**Root Cause**: 796 corrupted FTR records with malformed repeat counts

**Investigation**:
```python
# Generator approach (FAILED)
for record in parser:  # Dies on first exception
    process(record)

# Result: 47 devices (3.9% extraction rate)
```

**Solution**: Manual iteration
```python
# Manual iteration (SUCCESS)
while True:
    try:
        record = parser.read_record()
        if record is None:
            break
        yield record
    except Exception:
        continue

# Result: 1,205 devices (100% extraction rate)
```

**Impact**: 25.6x data extraction improvement

---

### Case Study 2: ETS File (Limit Overwriting)

**Symptoms**:
- Catalog showed 0 tests with valid limits
- Manual inspection showed limits in STDF
- Excel report showed blank limit columns

**Root Cause**: Multiple PTR records per test, later records with None overwrote valid limits

**Investigation**:
```python
# PTR Record 1 for Test 100
{
    "TEST_NUM": 100,
    "TEST_TXT": "VDD_CORE",
    "LO_LIMIT": 0.95,
    "HI_LIMIT": 1.05,
    "OPT_FLG": 0x00,
}

# PTR Record 2 for Test 100 (different device)
{
    "TEST_NUM": 100,
    "TEST_TXT": "VDD_CORE",
    "LO_LIMIT": None,
    "HI_LIMIT": None,
    "OPT_FLG": 0x30,  # Bits 4&5: use defaults
}

# BAD: Record 2 overwrote Record 1 limits with None
catalog[("VDD_CORE", "100")]["stdf_lower"] = None  # LOST!
```

**Solution**: Defensive merging
```python
# Only update if has_limit=True AND value is not None
if new["has_stdf_lower"] and new["stdf_lower"] is not None:
    existing["stdf_lower"] = new["stdf_lower"]
```

**Impact**: 0 → 265 tests with valid limits captured

---

### Case Study 3: T2K File (Field Name Mismatch)

**Symptoms**:
- Excel showed `0.0` instead of blank for tests with no limits
- OPT_FLAG=0xCE (bits 6&7 set) not recognized

**Root Cause**: T2K uses `OPT_FLAG` field name, parser only checked `OPT_FLG`

**Investigation**:
```python
# Parser code (FAILED)
opt_flg = ptr_record.get("OPT_FLG", 0)  # Returns 0 for T2K files

# Reality: T2K uses "OPT_FLAG"
ptr_record = {
    "TEST_NUM": 100,
    "OPT_FLAG": 0xCE,  # Not "OPT_FLG"
    "LO_LIMIT": None,
}
```

**Solution**: Check both field names
```python
opt_flg = ptr_record.get("OPT_FLG", ptr_record.get("OPT_FLAG", 0))
```

**Impact**: T2K files now show correct blank cells for tests with no limits

---

### Case Study 4: Excel Display Artifacts

**Symptoms**:
- Excel showed `0.03999999898951501` instead of `0.04`
- Test engineers questioned data precision
- Values were "correct" in STDF but "wrong" in Excel

**Root Cause**: IEEE 754 floating-point representation + default Excel formatting

**Investigation**:
```python
# STDF value (binary)
result = 0.04  # Stored as 0.03999999898951501 in IEEE 754

# Excel default format
cell.value = result
# Displays: 0.03999999898951501 (general format, full precision)
```

**Solution**: Extract and apply format strings
```python
# PTR record contains format string
c_resfmt = ptr_record.get("C_RESFMT")  # "%6.3f"

# Convert to Excel format
excel_format = stdf_format_to_excel(c_resfmt)  # "0.000"

# Apply to cell
cell.number_format = excel_format
cell.value = 0.04
# Displays: 0.040 (professional, matches ATE display)
```

**Impact**: 100% elimination of floating-point display artifacts

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-28  
**Based On**: 20+ production bug fixes and enhancements from real-world STDF parser development
