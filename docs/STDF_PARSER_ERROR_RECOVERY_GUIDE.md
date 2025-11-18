# STDF Parser Error Recovery: A Technical Guide

## Overview

This document provides guidance for implementing robust error recovery in STDF (Standard Test Data Format) parsers, based on real-world experience parsing imperfect STDF files from production test equipment.

## The Problem: Generator Exhaustion

### Symptom

Your STDF parser stops reading after encountering the first malformed record, even though the file contains valid records both before and after the corruption point.

**Example:**
```
File has 1,071,498 records total
├─ Records 1-3,708: Valid ✓
├─ Record 3,709: Corrupted FTR record ✗ (parser fails here)
├─ Records 3,710-1,071,498: Valid ✓ (never processed!)
└─ Result: Only extract data from first 3,708 records
```

### Root Cause

Most STDF parsers use a **generator pattern** for memory efficiency:

```python
def records(self):
    """Generator that yields STDF records."""
    while True:
        header = stream.read(4)
        if not header:
            break
        
        rec_len = int.from_bytes(header[0:2], byte_order)
        payload = stream.read(rec_len)
        
        record = self._decode_record(payload)  # ← Exception here!
        yield record
```

**The Problem:** When `_decode_record()` raises an exception:
1. The exception propagates out through the `yield` statement
2. Python marks the generator as **exhausted/dead**
3. Subsequent `next()` calls raise `StopIteration`
4. You cannot resume iteration - the generator is permanently broken

### Why Simple Try-Except Fails

This naive approach doesn't work:

```python
# ❌ BROKEN: Generator still dies after first exception
reader = STDFReader(file_path)
for record in reader.records():
    try:
        process(record)  # Too late - exception already escaped generator
    except Exception:
        continue  # Generator is already dead, loop exits
```

The exception occurs **inside** the generator (during `_decode_record()`), not during your iteration. By the time you catch it, the generator is already dead.

## Common Scenarios

### 1. Malformed Repeat Count Fields

**Issue:** Test equipment writes incorrect repeat counts in array fields.

**Example - FTR Record:**
```
Record header says: Length = 186 bytes
Fixed fields consume: 44 bytes
Remaining payload: 142 bytes

But field says: PGM_ICNT = 39
This requires: 39 × (U2 + N1) = 117 bytes for arrays alone
Plus: Variable-length fields (FAIL_PIN, VECT_NAM, etc.)

Total needed: >142 bytes → Parser fails trying to read past end
```

**Why This Happens:**
- Firmware bugs in test equipment
- Non-standard STDF extensions
- Spec misinterpretation by equipment vendor
- Data corruption during file transfer

### 2. Truncated Variable-Length Fields

**Issue:** `Cn` or `Dn` fields claim more bytes than remain in record.

```python
# Field header says: "String length = 16,384 bytes"
# Record has: Only 107 bytes remaining
# Parser fails: "Unexpected end of stream reading 16384 bytes"
```

### 3. Partial Record Corruption

**Issue:** Record body is corrupted but subsequent records are valid.

```
Byte 0-3:     Valid header (length = 186)
Byte 4-50:    Valid fields
Byte 51-60:   Corrupted data (invalid UTF-8, wrong enum values, etc.)
Byte 61-186:  Valid fields (never parsed because parser stopped)
Byte 187-190: Next record header (valid, but never reached)
```

## The Solution: Manual Record Iteration

### Strategy

Instead of catching exceptions outside the generator, **implement your own record iteration** that handles errors at the record level:

```python
def robust_records(reader):
    """Error-tolerant record iterator."""
    stream = reader._stream
    byte_order = reader._byte_order
    
    while True:
        try:
            # Step 1: Read record header
            header = stream.read(4)
            if not header or len(header) < 4:
                break
            
            rec_typ = header[2]
            rec_sub = header[3]
            rec_len = int.from_bytes(header[0:2], byte_order)
            
            # Step 2: Read FULL record payload
            payload = stream.read(rec_len)
            if len(payload) != rec_len:
                # Incomplete record - skip and continue
                # Stream is still aligned for next record
                continue
            
            # Step 3: Get record specification
            spec = RECORD_SPECS.get((rec_typ, rec_sub))
            if spec is None:
                continue  # Unknown record type
            
            # Step 4: Try to decode payload
            try:
                record = decode_record(spec, payload, byte_order)
                yield record  # Success!
                
            except Exception as e:
                # Decode failed, but we already read full payload
                # File stream is still aligned - can continue!
                log_error(f"Record decode failed: {e}")
                continue
                
        except Exception as e:
            # File-level error (I/O, corrupted headers, etc.)
            log_error(f"File read error: {e}")
            break
```

### Key Principles

#### 1. **Read Payload Before Decoding**

Always read the complete record payload based on the header's length field before attempting to parse:

```python
# ✓ CORRECT: Read full payload first
header = stream.read(4)
rec_len = int.from_bytes(header[0:2], byte_order)
payload = stream.read(rec_len)  # Read entire payload into memory

# Now decode from in-memory buffer
record = decode_record(payload)  # If this fails, stream is still aligned
```

```python
# ✗ WRONG: Decode directly from file stream
header = stream.read(4)
record = decode_record_from_stream(stream)  # If this fails, stream is misaligned
```

**Why:** If decoding fails partway through, the file stream position is corrupted and you can't find the next record header.

#### 2. **Use BytesIO for Payload Parsing**

Work with an in-memory copy of the payload:

```python
import io

payload = stream.read(rec_len)  # Read from file
body_stream = io.BytesIO(payload)  # Create in-memory buffer

# Parse fields from in-memory buffer
for field in record_spec.fields:
    value = decode_field(body_stream, field)  # Safe - can't corrupt file stream
```

**Why:** If field decoding fails, only the `BytesIO` position is affected. The file stream is untouched and still pointing to the next record header.

#### 3. **Validate Record Completeness**

Check if the full record was consumed:

```python
remaining = body_stream.read()
if remaining and len(remaining) > 0:
    log_warning(f"Record not fully consumed: {len(remaining)} extra bytes")
    # Decide: raise error, log and continue, or ignore
```

**Trade-off:**
- **Strict mode:** Reject records with extra bytes (spec compliance)
- **Lenient mode:** Accept records with extra bytes (real-world compatibility)

Choose based on your use case. For data analysis tools, lenient mode is often better.

#### 4. **Maintain Statistics**

Track what's happening during parsing:

```python
stats = {
    'total_records_attempted': 0,
    'records_parsed_successfully': 0,
    'records_skipped_unknown_type': 0,
    'records_failed_decode': 0,
    'records_incomplete': 0,
    'bytes_processed': 0,
}
```

**Why:** Users need to know:
- How much data was successfully extracted
- What percentage of the file was corrupted
- Whether the results are trustworthy

## Implementation Patterns

### Pattern 1: Wrapper Generator

Wrap an existing parser's generator with error handling:

```python
def error_tolerant_wrapper(strict_parser):
    """Wraps a strict parser with error recovery."""
    # This WON'T work - generator dies on first exception
    for record in strict_parser.records():
        try:
            yield record
        except Exception:
            continue  # Too late - generator already dead
```

**Verdict:** ❌ **Does not work** due to generator exhaustion.

### Pattern 2: Manual Iteration (Recommended)

Reimplement the iteration logic with error handling:

```python
def robust_iterator(parser):
    """Manually drives file reading with error recovery."""
    stream = parser._stream
    
    while True:
        # Read header
        header = stream.read(4)
        if not header:
            break
        
        # Read payload
        length = int.from_bytes(header[0:2], byte_order)
        payload = stream.read(length)
        
        # Try decode
        try:
            record = parser._decode_record(payload)
            yield record
        except Exception:
            continue  # Skip this record, stream still aligned
```

**Verdict:** ✅ **Works perfectly** - maintains stream alignment.

### Pattern 3: Callback-Based

Alternative approach using callbacks instead of generators:

```python
def parse_with_callbacks(file_path, on_record, on_error):
    """Parse STDF file with callback functions."""
    with open(file_path, 'rb') as stream:
        while True:
            header = stream.read(4)
            if not header:
                break
            
            length = int.from_bytes(header[0:2], byte_order)
            payload = stream.read(length)
            
            try:
                record = decode_record(payload)
                on_record(record)  # Callback for successful parse
            except Exception as e:
                on_error(record_num, e)  # Callback for errors
                continue

# Usage
records = []
errors = []

parse_with_callbacks(
    'data.stdf',
    on_record=lambda r: records.append(r),
    on_error=lambda num, e: errors.append((num, e))
)
```

**Verdict:** ✅ Works well, especially for languages without generators.

## Handling Specific STDF Issues

### Malformed Repeat Counts

**Problem:** Field claims more array elements than record space allows.

**Strict approach:**
```python
count = context.get('RTN_ICNT')  # Says 39
for i in range(count):
    item = decode_field(stream)  # Fails after ~10 items
```

**Lenient approach:**
```python
count = context.get('RTN_ICNT')
items = []
for i in range(count):
    try:
        items.append(decode_field(stream))
    except EndOfStream:
        break  # Got as many as possible
return items
```

**Best approach:**
```python
# Calculate maximum possible items based on remaining bytes
remaining_bytes = len(payload) - current_position
bytes_per_item = get_field_size(field_type)
max_possible = remaining_bytes // bytes_per_item

# Use smaller of claimed vs possible
actual_count = min(declared_count, max_possible)
return [decode_field(stream) for _ in range(actual_count)]
```

### Variable-Length Field Overflow

**Problem:** `Cn` or `Dn` field length exceeds remaining record space.

```python
def decode_variable_length(stream, remaining_bytes):
    """Safely decode Cn/Dn fields."""
    # Read declared length
    length_byte = stream.read(1)[0]
    
    # Validate against remaining space
    if length_byte > remaining_bytes - 1:
        # Truncate to available space
        actual_length = max(0, remaining_bytes - 1)
        log_warning(f"Cn field truncated: claimed {length_byte}, got {actual_length}")
    else:
        actual_length = length_byte
    
    # Read available data
    data = stream.read(actual_length)
    return data
```

### Byte Order Detection

**Critical:** Always detect byte order from FAR (File Attributes Record):

```python
# Read first record (FAR)
header = stream.read(4)

# FAR is always record type (0, 10) and length 2
if header[2] == 0 and header[3] == 10:
    len_little = int.from_bytes(header[0:2], 'little')  # = 2
    len_big = int.from_bytes(header[0:2], 'big')        # = 512
    
    if len_little == 2:
        byte_order = 'little'
    elif len_big == 2:
        byte_order = 'big'
    else:
        raise ValueError("Invalid FAR record")
```

**Why:** STDF files can be big-endian or little-endian. Getting this wrong will corrupt all multi-byte values.

## Testing Your Parser

### Test Files

Create test files with known corruption patterns:

1. **Malformed repeat count:**
   ```python
   # Build FTR record with PGM_ICNT = 100 but only space for 5 items
   ```

2. **Truncated variable-length field:**
   ```python
   # Build record with Cn field claiming 1000 bytes but record ends after 50
   ```

3. **Partial record:**
   ```python
   # Write record header (length=200) but only 100 bytes of payload
   ```

4. **Interleaved corruption:**
   ```python
   # Valid records → Corrupted → Valid records → Corrupted → Valid records
   ```

### Validation

Compare your parser against a known-good commercial parser:

```python
# Parse with your parser
your_results = your_parser.parse('test.stdf')

# Parse with commercial tool and export to CSV
commercial_csv = commercial_tool.export('test.stdf')

# Compare
assert your_results.device_count == commercial_csv.device_count
assert your_results.measurement_count == commercial_csv.measurement_count
assert your_results.unique_devices == commercial_csv.unique_devices
```

### Metrics to Track

```python
{
    'total_records_in_file': 1_071_498,
    'records_successfully_parsed': 1_070_702,
    'records_failed_decode': 796,
    'failure_rate_percent': 0.074,
    
    'records_by_type': {
        'PTR': 1_049_799,
        'FTR': 796,  # All failed
        'PIR': 1_205,
        'PRR': 1_205,
        # ...
    },
    
    'devices_extracted': 1_205,
    'measurements_extracted': 1_049_798,
    
    'errors_by_type': {
        'EndOfStream': 796,
        'InvalidEnum': 0,
        'MalformedUTF8': 0,
    }
}
```

## Real-World Example

### Case Study: Bhavik Test File

**File specs:**
- Size: 71.6 MB
- Total records: 1,071,498
- Expected devices: 1,205

**Corruption pattern:**
- 796 FTR (Functional Test Record) entries had malformed `PGM_ICNT` fields
- Repeat count claimed 39 items but record only had space for ~5-10 items
- FTR records scattered throughout file (first at record #3,709)

**Strict parser results:**
- Stopped at record #3,709 (first corrupted FTR)
- Extracted only 47 devices (last 47 in file before corruption)
- Lost 96% of data

**Robust parser results:**
- Processed all 1,071,498 records
- Skipped 796 corrupted FTR records (0.074%)
- Successfully parsed 1,070,702 records
- Extracted **all 1,205 devices** with 1,049,798 measurements
- Matched commercial parser (SEDana) extraction completeness

### Code Comparison

**Before (Generator exhaustion):**
```python
def parse_file(path):
    with STDFReader(path) as reader:
        for record in reader.records():  # Dies on first exception
            process(record)
            
# Result: 47 devices
```

**After (Manual iteration):**
```python
def parse_file(path):
    with STDFReader(path) as reader:
        stream = reader._stream
        
        while True:
            header = stream.read(4)
            if not header:
                break
            
            length = int.from_bytes(header[0:2], byte_order)
            payload = stream.read(length)
            
            try:
                record = reader._decode_record(payload)
                process(record)
            except Exception:
                continue  # Skip bad record, keep going
                
# Result: 1,205 devices
```

## Best Practices Summary

### Do's ✅

1. **Read full record payload before decoding**
   - Maintains file stream alignment
   - Enables recovery from decode failures

2. **Use in-memory buffers (BytesIO) for field parsing**
   - Protects file stream from corruption
   - Allows retry/recovery strategies

3. **Track detailed statistics**
   - Total records vs. parsed vs. failed
   - Error types and frequencies
   - Extraction completeness metrics

4. **Warn users about data quality issues**
   - Report corruption percentage
   - Indicate confidence in results
   - Suggest validation against commercial tools

5. **Test with real-world files**
   - Production test equipment generates imperfect files
   - Validate against commercial parser results

### Don'ts ❌

1. **Don't rely on generator exception handling**
   - Generators die on first exception
   - Can't resume iteration

2. **Don't decode directly from file stream**
   - Stream position gets corrupted on errors
   - Can't find next record header

3. **Don't fail silently**
   - Users need to know about data quality issues
   - Report statistics, log warnings

4. **Don't assume spec compliance**
   - Real test equipment has bugs
   - Firmware misinterprets spec
   - Choose practical compatibility over theoretical purity

5. **Don't skip validation testing**
   - Compare against commercial parsers
   - Verify device/measurement counts match
   - Check extraction completeness

## Performance Considerations

### Memory

**Trade-off:** Reading full record payloads uses more memory.

```python
# Memory usage = largest_record_size (typically < 10 KB)
payload = stream.read(rec_len)  # In-memory copy
```

**Impact:** Negligible - STDF records are typically < 1 KB, maximum ~10 KB.

### Speed

Manual iteration is slightly slower than pure generator approach:

```python
# Generator (fast): ~1M records/minute
for record in reader.records():
    process(record)

# Manual iteration (slightly slower): ~900K records/minute  
while True:
    header = stream.read(4)
    payload = stream.read(length)
    record = decode(payload)
```

**Impact:** ~10% slower, but you get **100% data extraction** vs **4% with naive approach** (96% data loss).

**Verdict:** Absolutely worth it.

### Optimization Tips

1. **Buffer file I/O:**
   ```python
   # Use buffered streams
   with open(path, 'rb', buffering=64*1024) as stream:
   ```

2. **Batch statistics updates:**
   ```python
   # Update every 10K records instead of every record
   if record_count % 10000 == 0:
       update_stats()
   ```

3. **Skip unneeded record types:**
   ```python
   # If you only need PTR records
   if rec_typ != 15 or rec_sub != 10:
       continue  # Skip without full decode
   ```

## Conclusion

Robust STDF parsing requires **manual record-level iteration** with proper error recovery. The key insights:

1. **Generator exhaustion** is the primary failure mode
2. **Reading full payloads first** maintains stream alignment
3. **In-memory parsing** protects file stream integrity
4. **Detailed statistics** ensure transparency
5. **Real-world compatibility** trumps strict spec compliance

By implementing these patterns, you can achieve commercial-grade STDF parsing that handles imperfect files from production test equipment while maintaining data integrity and user trust.

## References

- STDF V4 Specification: Standard Test Data Format Specification
- This implementation: `cpkanalysis/ingest.py` (robust_record_iterator)
- Case study details: `STDF_PARSING_INVESTIGATION.md`
