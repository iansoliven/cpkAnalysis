# STDF Reader Core: Architectural Design (Draft)

## 1. Vision & Motivation
Create a reusable, robust STDF Reader Core that:
- Ingests STDF V4 data (and is extensible to variants) with correct flag, limit, and site semantics.
- Provides multiple consumption modes: streaming records, chunked columnar batches, aggregated DataFrames, Parquet/Arrow output.
- Exposes stable, documented data contracts decoupled from downstream CPK analysis concerns (outliers, stats, workbook).
- Enables usage in other contexts (data lakes, characterization dashboards, anomaly detection, ETL pipelines) without pulling in analysis-specific dependencies.

## 2. Scope
IN SCOPE:
- Parsing PTR/PIR/PRR/SDR (initial minimal subset) + extensible record dispatch.
- Limit and unit resolution (OPT_FLAG semantics, scaling, SI prefix handling).
- Site topology extraction & validation.
- Error/warning accumulation (structured, machine-readable).
- Performance features: streaming API, memory-lean ingestion, optional selective field extraction.
- Schema versioning & metadata embedding.

OUT OF SCOPE (initial phase):
- Statistical calculations (CPK, yield, pareto).
- Outlier filtering.
- Workbook/report generation.
- What-if/spec limit overlays (treated as external augmentations).

## 3. Non-Goals / Deferrals
- Full STDF V4 record coverage (focus on those needed for measurements & topology first).
- Binary writing of STDF (reader only).
- Multi-process ingestion (optimize single-process first; design for later parallelization).
- Auto-detection of compressed archives (defer to caller or wrapper utilities).

## 4. Current State Summary
Existing module `cpkanalysis.ingest`:
- Combines parsing, catalog building, DataFrame aggregation, and Parquet writing in one function.
- Duplicates limit semantics logic across `_populate_test_catalog_from_ptr` and `_extract_measurement`.
- Accumulates all measurement frames in memory while also writing a single Parquet file (hybrid approach without streaming consumer).
- Provides no explicit schema version; column names hard-coded.
- Unit values are pre-scaled + SI-prefixed, with numeric values also scaled (risk of confusion in non-CPK contexts).

## 5. Design Principles
1. Single Source of Truth for limit/flag semantics.
2. Separation of concerns: parsing vs transformation vs materialization.
3. Explicit schemas (types, nullability, description) with version tags.
4. Extensibility via plugin hooks (record mappers, additional metadata enrichers).
5. Deterministic & idempotent ingestion outputs (identical for same input set).
6. Streaming-first; DataFrame as convenience layer.
7. Observability: counters, timings, warnings, anomaly flags.

## 6. High-Level Architecture
```
+-------------------+
|   STDF Core API   |
+---------+---------+
          |
          v
   +--------------+        +--------------------+
   | Record Layer | -----> |  Catalog Builder   |
   +--------------+        +--------------------+
          |                          |
          v                          v
   +--------------+          +-----------------+
   | Filter/Flag  |          | Site Topology   |
   | Resolution   |          +-----------------+
   +--------------+                  |
          |                           v
          v                   +---------------+
   +--------------+           | Metadata      |
   | Normalizer   |           | Accumulator   |
   +--------------+           +---------------+
          |                               |
          +---------------+---------------+
                          v
                +------------------+
                | Materialization  | (stream → batch → DataFrame / Parquet)
                +------------------+
```

## 7. Core Components
- `STDFReaderOptions`: ingestion configuration (ignore unknown, streaming batch size, validate flags, include_invalid, max_records, selected_fields).
- `STDFRecord`: lightweight internal representation (name, raw dict, offset, timestamp).
- `MeasurementBuilder`: stateful test metadata & limit resolution per test key.
- `CatalogBuilder`: collects per-test info, handles merges across files.
- `SiteTopology`: aggregates and validates SDR + observed site usage, exposes mapping.
- `IngestionStats`: counters (devices_total, measurements_valid, measurements_invalid, tests, files, bytes_read, time_per_file).
- `ErrorReporter`: structured warnings/errors list.
- `SchemaRegistry`: holds named schema versions (measurement v1, catalog v1) and can serialize metadata to Parquet key/value.
- `STDFIngestor`: façade orchestrating the above; entrypoint for callers.

## 8. Data Contracts (Draft v1)
### 8.1 Measurement Schema (v1)
Field | Type | Null? | Source | Notes
------|------|-------|--------|------
file | string | no | file path | basename only.
file_path | string | no | file path | absolute path.
device_id | string | no | PIR/PRR | synthesized fallback guaranteed unique per file.
device_sequence | int | no | implicit counter | 1-based sequence.
site | int | yes | PIR | None if absent.
test_name | string | no | PTR.TEST_TXT | may be empty string if absent.
test_number | string | no | PTR.TEST_NUM | stringified.
unit_raw | string | yes | PTR.UNITS | original base unit.
unit_scale | int | yes | PTR.RES_SCAL | canonical scale (may be 0/None).
unit_display | string | yes | derived | SI-prefix composition (optional detach later).
value_raw | float | yes | PTR.RESULT | before scaling.
value | float | yes | derived | scaled numeric used for stats; if scale applied.
stdf_lower | float | yes | PTR.LO_LIMIT | semantic after OPT_FLAG resolution.
stdf_upper | float | yes | PTR.HI_LIMIT | semantic after OPT_FLAG resolution.
limit_state | string | no | OPT_FLAG | enum: none, default, explicit, cleared.
result_format | string | yes | PTR.C_RESFMT | as-is.
lower_format | string | yes | PTR.C_LLMFMT | as-is.
upper_format | string | yes | PTR.C_HLMFMT | as-is.
result_scale | int | yes | PTR.RES_SCAL | preserve raw.
lower_scale | int | yes | PTR.LLM_SCAL | preserve raw.
upper_scale | int | yes | PTR.HLM_SCAL | preserve raw.
part_status | string | no | PRR.PART_FLG | PASS/FAIL derived.
measurement_index | int | no | internal counter | per (device_id, test key).
timestamp | float | yes | PTR.TEST_TIM | fallback measurement_index.
flags_parm | int | yes | PTR.PARM_FLG | raw integer.
flags_test | int | yes | PTR.TEST_FLG | raw integer.
flags_opt | int | yes | PTR.OPT_FLAG/OPT_FLG | raw integer.
invalid_reason | string | yes | derived | enumerated rejection cause if filtered.

### 8.2 Test Catalog Schema (v1)
Field | Type | Null? | Source | Notes
------|------|-------|--------|------
test_name | string | no | PTR | unified key.
test_number | string | no | PTR | unified key.
unit_raw | string | yes | PTR | base unit.
unit_scale | int | yes | PTR | scale used.
unit_display | string | yes | derived | convenience.
stdf_lower | float | yes | PTR | final resolved value.
stdf_upper | float | yes | PTR | final resolved value.
limit_state_lower | string | no | OPT_FLAG | none, default, explicit, cleared.
limit_state_upper | string | no | OPT_FLAG | none, default, explicit, cleared.
result_format | string | yes | PTR | first non-empty.
lower_format | string | yes | PTR | first non-empty.
upper_format | string | yes | PTR | first non-empty.
result_scale | int | yes | PTR | updated when explicit.
lower_scale | int | yes | PTR | updated when explicit.
upper_scale | int | yes | PTR | updated when explicit.
file_origins | list[string] | no | ingestion | files contributing metadata.

### 8.3 Site Topology Schema (v1)
Field | Type | Null? | Source | Notes
------|------|-------|--------|------
head_num | int | no | SDR | default 0 if missing.
site_group | int | yes | SDR | group id.
site_numbers | list[int] | no | SDR | sorted unique.
handler_type | string | yes | SDR | equipment meta.
... (other equipment fields) | string | yes | SDR | unchanged.

### 8.4 Transformation Semantics
This subsection codifies every transformation applied between raw STDF record fields and the canonical schemas, ensuring reproducibility, auditability, and clear separation of raw vs derived data.

| Aspect | Rule | Rationale | Repro / Audit Strategy |
|--------|------|-----------|------------------------|
| Test Identification | `test_number` stringified; `test_name` trimmed; empty becomes `""` (not `NULL`). | Stable grouping keys avoiding dtype drift. | Preserve original raw fields in raw record stream. |
| Device Identity | Prefer PIR `PART_ID`; if absent generate `SITE<site>_<seq>` with monotonic `device_sequence`. | Guarantees non-null key for yield math. | Record generation pattern & sequence in stats. |
| Measurement Scaling | If `scale_values=True`, numeric `value = value_raw * (10 ** result_scale)` when `result_scale` non-zero; else `value = value_raw`. | Align with human-readable engineering units. | Retain `value_raw` & `result_scale` columns; store transformation flag in metadata. |
| Unit Composition | `unit_display` = SI prefix + `unit_raw` using inverse exponent of `result_scale` when mapping found (e.g. scale=-3 → m). | Familiar engineering presentation; reversible. | Keep `unit_raw`, `unit_scale`; do not infer when `unit_raw` empty. |
| Limits Resolution | Apply precedence: (1) explicit clear bits 6/7 → none; (2) default bits 4/5 → preserve cached; (3) explicit numeric value updates; (4) ambiguous None → retain prior. | Enforces STDF semantics; prevents accidental limit loss. | Persist `limit_state_lower/upper` + original flags in catalog; snapshot tests over flag matrix. |
| Limit Scaling | Apply same scaling strategy as measurement if scaling enabled; preserve raw limit fields implicitly through raw record stream only. | Consistency with measurement domain. | Optionally add future `stdf_lower_raw` if audit mode required. |
| Invalid Measurements | If `include_invalid=True` retain rows with `invalid_reason` enumerated (`parm_flag_invalid`, `test_flag_mask`, etc.); else drop prior to materialization. | Support ETL completeness vs cleaned analytics. | `invalid_reason` non-null signals exclusion candidate; test ensures excluded from stats. |
| Timestamp Fallback | Use PTR `TEST_TIM` when finite; else fallback to `measurement_index` (float cast) to preserve ordering. | Ensures sortable temporal field even when absent. | Stats record count of fallback usage; expose ratio. |
| Site / Head / Group | `site` from PIR; `head_num/site_group` from most recent SDR in scope; if SDR absent use defaults (0 / NULL). | Rich multi-site analytics & equipment lineage. | Record presence counters; emit warning if measurement references no SDR. |
| Record Index & Offset | `record_index` increments per raw record; `byte_offset` recorded if reader exposes; missing offset stored as NULL. | Traceability & debugging. | Provide lookup helper to jump from measurement to raw record. |
| Partition Keys | Derived from MIR/WIR if present: `lot_id`, `wafer_id`; absent values replaced with `unknown`. Directory layout: `lot_id=<lot>/wafer_id=<wafer>/file=<basename>.parquet`. | Query pruning & organization. | Persist chosen keys in dataset-level `_dataset_meta.json`. |
| Ordering Guarantees | Within a file, output preserves PRR closure order; cross-file order not guaranteed unless `--stable-order` forces filename lexical sort. | Deterministic batching optional; performance by default. | Document flag; include `file_order` integer in stats. |
| Concurrency Effects | Parallel ingestion merges fragments; catalog merges first-wins for unit unless conflict triggers warning; limits unified via precedence rules. | Avoids race-induced nondeterminism while enabling speed. | Conflict warnings (`INTEGRITY.TEST.UNIT_CONFLICT`) captured with examples. |
| Schema Versioning | `schema_version` pinned in result; breaking column changes bump major (`measurement_v2`). | Client compatibility. | Maintain CHANGELOG; add version to Parquet metadata key `stdf_core.schema`. |
| Immutability Constraints | Columns defined in v1 not repurposed; new semantics require new columns or version bump. | Prevent silent semantic drift. | Enforced by tests comparing declared schema vs generated. |

Additional Notes:
- All derived columns must be computable from raw record stream + deterministic rules; no hidden global mutable state.
- Transformation parameters (scaling enabled, invalid retention) serialized into a run-level `ingestion_config` metadata block.
- Future "audit mode" will append raw limit columns and file checksum; deferred per decisions section.

Open Follow-Ups (to track before implementation freeze):
1. Decide whether to persist `stdf_lower_raw` / `stdf_upper_raw` pre-scale in Phase 1 (currently deferred; may be required by external validation tools).
2. Confirm acceptable default for missing `lot_id` / `wafer_id` (currently `unknown`).
3. Evaluate cost of capturing `byte_offset` (depends on underlying reader support). If unsupported, document limitation explicitly.


## 9. Core Public Interfaces (Draft)
```python
@dataclass
class STDFReaderOptions:
    ignore_unknown: bool = True
    batch_size: int | None = None  # number of devices or measurements per batch
    include_invalid: bool = False  # surface invalid measurements with invalid_reason
    validate_flags: bool = True
    max_records: int | None = None
    selected_fields: list[str] | None = None  # projection
    scale_values: bool = True  # apply value scaling
    compose_unit_display: bool = False  # optional

class STDFIngestor:
    def __init__(self, options: STDFReaderOptions = STDFReaderOptions()): ...

    def stream_measurements(self, path: Path) -> Iterator[RawMeasurement]: ...
    def ingest_files(self, sources: Sequence[Path]) -> STDFIngestResult: ...
    def detect_sites(self, sources: Sequence[Path]) -> SiteDetectionResult: ...

@dataclass
class STDFIngestResult:
    measurements: pd.DataFrame  # measurement schema v1
    catalog: pd.DataFrame       # test catalog schema v1
    sites: pd.DataFrame         # optional site topology
    stats: IngestionStats
    warnings: list[IngestionWarning]
    errors: list[IngestionError]
    schema_version: str = "measurement_v1"
```

## 10. Error & Warning Strategy
### 10.1 Taxonomy Overview
We define a stable, machine-readable taxonomy to classify all issues. Each emitted issue is a structured record enabling filtering, aggregation, suppression, and audit.

### 10.2 Severity Levels
| Level | Meaning | Action |
|-------|---------|--------|
| INFO | Informational events | No action; for context only |
| NOTICE | Non-problem noteworthy conditions | Monitor; potential future warning |
| WARNING | Potential data quality / semantic problem | Investigate; may affect analysis correctness |
| ERROR | Parsing / ingestion failure for a portion; ingestion continues | Review; may indicate partial data loss |
| FATAL | Ingestion cannot proceed | Abort run; user intervention required |

### 10.3 Categories & Naming Convention
Code format: `CATEGORY.SUBCATEGORY.KEYWORD` (uppercase, dot-separated). Primary categories:
- `RECORD` (low-level STDF record parsing)
- `LIMIT` (limit & flag semantics)
- `SITE` (site/topology extraction)
- `INGEST` (overall ingestion workflow/materialization)
- `INTEGRITY` (cross-file/test consistency)
- `PERFORMANCE` (resource & throughput concerns)
- `SYSTEM` (environment/dependency/state)

Examples: `LIMIT.OPTFLAG.CONTRADICTORY_BITS`, `SITE.DETECT.NONE_FOUND`, `INTEGRITY.TEST.UNIT_CONFLICT`.

### 10.4 Issue Record Schema
```json
{
    "code": "LIMIT.OPTFLAG.CONTRADICTORY_BITS",
    "level": "WARNING",
    "category": "LIMIT",
    "message": "OPT_FLAG bits indicate both default usage and no-limit for test",
    "detail": {
        "opt_flag": 80,
        "bits": ["BIT4_DEFAULT_LOW", "BIT6_NO_LOW_LIMIT"],
        "test_number": "100",
        "test_name": "VDD_CORE"
    },
    "file": "lot1.stdf",
    "file_path": "/data/lot1.stdf",
    "record_index": 4521,
    "byte_offset": 987654,
    "test_name": "VDD_CORE",
    "test_number": "100",
    "site": 3,
    "head_num": 1,
    "timestamp": "2025-11-12T04:15:23.456Z",
    "occurrence": 1,
    "correlation_id": "run-20251112-abc123",
    "memory_rss_mb": 512.4,
    "cpu_percent": 82.1
}
```

Required fields: `code`, `level`, `message`, `timestamp`. Optional depending on context: `file`, `record_index`, `byte_offset`, `test_name`, `test_number`, `site`, `head_num`, `detail`, `occurrence`.

### 10.5 Initial Code Set
| Code | Level | Description |
|------|-------|-------------|
| RECORD.PARSE.FAIL | ERROR | Exception converting STDF record to dict; record skipped |
| RECORD.PARSE.UNKNOWN_NAME | NOTICE | Unsupported record type ignored |
| RECORD.FIELD.MISSING_CRITICAL | WARNING | Measurement-relevant field absent |
| RECORD.FLAG.INVALID_RESULT | NOTICE | Measurement flagged invalid (retained if include_invalid) |
| LIMIT.OPTFLAG.CONTRADICTORY_BITS | WARNING | Bits imply conflicting semantics (default + no-limit) |
| LIMIT.CACHE.NO_DEFAULT_REFERENCED | WARNING | Default limit referenced before initialization (bit 4/5) |
| LIMIT.UPDATE.SCALE_VARIANCE | NOTICE | RES_SCAL changed within same test group |
| SITE.DETECT.NONE_FOUND | NOTICE | No site identifiers detected in probe |
| SITE.TOPOLOGY.DUPLICATE_HEAD_GROUP | WARNING | Conflicting SDR site sets for same head/group |
| INTEGRITY.TEST.UNIT_CONFLICT | WARNING | Same test has differing unit values across files |
| INTEGRITY.DEVICE.ID_DUPLICATE | WARNING | Duplicate device_id detected in file |
| INGEST.STREAM.INVALID_INCLUDED | INFO | Invalid measurements retained due to option |
| INGEST.PARTITION.WRITE_FAIL | ERROR | Failed writing partition Parquet fragment |
| PERFORMANCE.MEMORY.HIGH_WATERMARK | NOTICE | Memory usage exceeded threshold |
| PERFORMANCE.PARALLEL.WORKER_FAILURE | ERROR | Parallel worker crashed during parse |
| SYSTEM.DEPENDENCY.READER_UNAVAILABLE | FATAL | STDF reader not initialised; abort |
| SYSTEM.PATH.ACCESS_DENIED | ERROR | File access denied; skipped |

### 10.6 Deduplication & Occurrence Tracking
- Maintain `(code, file, test_number)` seen set.
- Increment `occurrence` on repeats.
- Suppression threshold confirmed: **25**. After 25 repeats emit `SYSTEM.LOG.SUPPRESSED_REPEATS` (NOTICE) with aggregated count.
- Continue tracking total suppressed per code in run summary.

### 10.7 Emission Strategy
1. Logging format: newline-delimited JSON (NDJSON); one JSON object per line for easy streaming ingestion (e.g. `grep`, `jq`, ELK).
2. Each record includes `correlation_id` (stable per run) to group multi-file batches.
3. Performance-related events populate `memory_rss_mb` and `cpu_percent` when measurable.
4. Collect issues into `warnings` / `errors` lists inside `STDFIngestResult` for programmatic access.
5. Optional sidecar file: `ingestion_issues.json` containing header block and all issue objects (not NDJSON) if `emit_issue_sidecar` is enabled.

### 10.8 Registry & Extensibility
Provide `IssueRegistry` with entries: `{code, default_level, description}`. Custom modules can register new codes; unknown codes rejected unless explicitly allowed.

### 10.9 Testing Approach
- Unit: each code path triggers one issue; assert schema fields.
- Golden: synthetic STDF triggering multiple flags; snapshot JSON stable.
- Performance: high-volume invalid measurements ensure dedupe works.
- Concurrency: parallel parse ensures no duplicate race emissions.

### 10.10 Migration Notes
Phase 1: Implement taxonomy + minimal set (above) alongside existing warnings.
Phase 2: Replace ad-hoc `warnings.warn` with taxonomy.
Phase 3: Add suppression + sidecar option.
Phase 4: Expand codes for MIR/WIR metadata anomalies.

### 10.11 Finalized Decisions (Taxonomy Specific)
1. Repeat suppression threshold: 25.
2. Logging format: newline-delimited JSON (NDJSON).
3. Correlation ID: included (`correlation_id` field) for multi-run tracing and aggregation.
4. Resource stats: `memory_rss_mb`, `cpu_percent` captured in PERFORMANCE events when available.
5. Sidecar optional; NDJSON remains canonical streaming form.


## 11. Flag & Limit Resolution (Unified Function)
Single function signature:
```python
def resolve_limits(opt_flag: int, lo_raw: float | None, hi_raw: float | None, cache: TestCache) -> LimitResolution:
    ...
```
Outputs structured result including state enums (`explicit`, `default`, `none`, `cleared`).

### 11.1 Unified Limit Resolver Detailed Specification
The current implementation spreads limit logic across `_populate_test_catalog_from_ptr` and `_extract_measurement`. We replace this with a single, deterministic resolver that:

Goals:
1. Centralize OPT_FLAG semantics (bits 4/5/6/7) for both catalog population and measurement extraction.
2. Provide consistent state labels for lower and upper limits separately.
3. Preserve historical limit values when defaults are referenced (bits 4/5) without re-reading stale raw fields.
4. Explicitly clear limits when no-limit bits (6/7) set, even if raw fields contain garbage values.
5. Avoid accidental clearing when raw limit field is None but no signalling flag asserts meaning.
6. Emit taxonomy issues for contradictions or misuse.

#### 11.1.1 Inputs
| Parameter | Type | Description |
|-----------|------|-------------|
| `opt_flag` | int | Raw OPT_FLAG (or 0 if absent) covering bits 0..7. |
| `lo_raw` | float | Raw LO_LIMIT (scaled or unscaled depending on early transform) or None. |
| `hi_raw` | float | Raw HI_LIMIT or None. |
| `cache` | `TestCache` | Mutable per-test metadata object holding previously resolved limits & state. |
| `scales` | `ScaleInfo` | (Optional) contains result/low/high scaling factors for audit logging. |
| `emit_issue` | Callable | Hook to log taxonomy issues (optional injection). |

#### 11.1.2 Cache Structure (`TestCache`)
```python
@dataclass
class TestCache:
    low_limit: float | None = None
    high_limit: float | None = None
    low_state: str = "none"      # one of: none, explicit, default, cleared
    high_state: str = "none"
    default_low_initialized: bool = False
    default_high_initialized: bool = False
```
Note: `default_*_initialized` set after first explicit limit (bits 4/5 refer back to this).

#### 11.1.3 Output (`LimitResolution`)
```python
@dataclass
class LimitResolution:
    low: float | None
    high: float | None
    low_state: str        # none | explicit | default | cleared | unchanged
    high_state: str
    cache_updated: bool
```
`unchanged` used when resolver performs no mutation (e.g. ambiguous None with no flags).

#### 11.1.4 Flag Bits (STDF Spec Focus)
| Bit | Hex | Meaning | Resolver Interpretation |
|-----|-----|---------|-------------------------|
| 4 | 0x10 | LO_LIMIT & LLM_SCAL invalid (use default) | Use cached low if initialized; else do not mutate (emit warning) |
| 5 | 0x20 | HI_LIMIT & HLM_SCAL invalid (use default) | Use cached high if initialized; else do not mutate (emit warning) |
| 6 | 0x40 | No low limit exists | Clear low: set `low=None`, `low_state=cleared`, mark no default available until new explicit value |
| 7 | 0x80 | No high limit exists | Clear high similarly |

#### 11.1.5 Precedence Order (Per Side)
1. Clear (bit 6 or 7) → state `cleared`.
2. Default (bit 4 or 5) → if initialized: state `default`; else state remains prior & emit `LIMIT.CACHE.NO_DEFAULT_REFERENCED`.
3. Explicit numeric (raw not None) → state `explicit`; updates cache and sets `default_*_initialized=True`.
4. Ambiguous None (raw None, no bits) → state `unchanged` (retain existing); no emission.

Contradictory bits (e.g. 0x10 + 0x40 simultaneously) resolved by precedence (Clear supersedes Default) and emit `LIMIT.OPTFLAG.CONTRADICTORY_BITS`.

#### 11.1.6 Pseudocode
```python
def resolve_limits(opt_flag: int, lo_raw: float | None, hi_raw: float | None, cache: TestCache, emit_issue: Callable | None = None) -> LimitResolution:
    def issue(code: str, level: str = "WARNING", detail: dict | None = None):
        if emit_issue: emit_issue(code, level, detail or {})

    low_clear = bool(opt_flag & 0x40)
    low_default = bool(opt_flag & 0x10)
    high_clear = bool(opt_flag & 0x80)
    high_default = bool(opt_flag & 0x20)

    # Contradictions
    if (low_clear and low_default) or (high_clear and high_default):
        issue("LIMIT.OPTFLAG.CONTRADICTORY_BITS", detail={"opt_flag": opt_flag})

    # Resolve lower
    if low_clear:
        cache.low_limit = None
        cache.low_state = "cleared"
        cache.default_low_initialized = False
    elif low_default:
        if cache.default_low_initialized and cache.low_limit is not None:
            cache.low_state = "default"
        else:
            # Referenced default before initialization
            issue("LIMIT.CACHE.NO_DEFAULT_REFERENCED")
            # Do not mutate state
        # ignore lo_raw even if present
    elif lo_raw is not None:
        cache.low_limit = lo_raw
        cache.low_state = "explicit"
        cache.default_low_initialized = True
    else:
        # ambiguous None; leave cache.low_state as-is or set to none if never initialized
        if cache.low_state == "none" and cache.low_limit is None:
            cache.low_state = "none"
        else:
            # mark unchanged explicitly in output
            pass

    # Resolve upper
    if high_clear:
        cache.high_limit = None
        cache.high_state = "cleared"
        cache.default_high_initialized = False
    elif high_default:
        if cache.default_high_initialized and cache.high_limit is not None:
            cache.high_state = "default"
        else:
            issue("LIMIT.CACHE.NO_DEFAULT_REFERENCED")
    elif hi_raw is not None:
        cache.high_limit = hi_raw
        cache.high_state = "explicit"
        cache.default_high_initialized = True
    else:
        if cache.high_state == "none" and cache.high_limit is None:
            cache.high_state = "none"

    # Determine output states (explicit unchanged tagging)
    low_out_state = cache.low_state if not (lo_raw is None and not (low_clear or low_default)) else (cache.low_state if cache.low_state != "none" else "none")
    high_out_state = cache.high_state if not (hi_raw is None and not (high_clear or high_default)) else (cache.high_state if cache.high_state != "none" else "none")

    return LimitResolution(low=cache.low_limit, high=cache.high_limit, low_state=low_out_state, high_state=high_out_state, cache_updated=True)
```

#### 11.1.7 State Transition Examples (Lower Side)
| Previous State | OPT_FLAG | lo_raw | Result State | Comments |
|----------------|---------|--------|--------------|----------|
| none           | 0x00    | 1.0    | explicit     | First explicit establishes default |
| explicit       | 0x10    | 999.0  | default      | Raw ignored; uses cached 1.0 |
| default        | 0x10    | None   | default      | Continues referencing cached |
| explicit       | 0x40    | 1.0    | cleared      | Clear overrides raw garbage |
| cleared        | 0x10    | None   | none (unchanged) | Warning: default requested but not initialized |
| explicit       | 0x00    | None   | unchanged    | Ambiguous None preserves existing explicit |

#### 11.1.8 Edge Cases
1. Contradictory flags (bits 4+6 or 5+7): Clear wins; emit warning.
2. Default referenced before initialization: output unchanged; emit warning.
3. Raw numeric with default bit set: raw ignored (space-saving intent). Consider INFO emission for debugging (optional toggle).
4. Simultaneous low & high clears: both limits cleared; states set individually.
5. Transition from cleared to explicit numeric: explicit re-establishes default initialization.
6. Scale variance independent: scale changes do not alter limit state; separate issue logged (`LIMIT.UPDATE.SCALE_VARIANCE`).

#### 11.1.9 Test Matrix (Lower Side Core Cases)
| Case | opt_flag | lo_raw | prior low_limit | prior state | default_initialized | Expected low_limit | Expected state | Issues |
|------|----------|--------|-----------------|-------------|---------------------|--------------------|---------------|--------|
| A    | 0x00     | 1.0    | None            | none        | False               | 1.0                | explicit       | None |
| B    | 0x10     | 999.0  | 1.0             | explicit    | True                | 1.0                | default        | None |
| C    | 0x40     | 5.0    | 1.0             | explicit    | True                | None               | cleared        | OPTFLAG contradiction if bit4 also set |
| D    | 0x10     | None   | None            | none        | False               | None               | none           | NO_DEFAULT_REFERENCED |
| E    | 0x00     | None   | 1.0             | explicit    | True                | 1.0                | unchanged      | None |
| F    | 0x40     | None   | None            | none        | False               | None               | cleared        | None |
| G    | 0x50     | 2.0    | 1.0             | explicit    | True                | None               | cleared        | CONTRADICTORY_BITS |

Replicate matrix for upper side with bit adjustments (5 default, 7 clear).

#### 11.1.10 Integration Points
- Catalog population: call resolver with raw limits; update test catalog row fields from returned state; unit assignment separate.
- Measurement extraction: call resolver first (so measurement row reflects current semantic limits) then include resulting `stdf_lower/stdf_upper`.
- Warning emission centralized (`emit_issue`).

#### 11.1.11 Performance Considerations
- Resolver is O(1); minimal branching. Called per PTR record only.
- Adds at most one warning emission per ambiguous event preventing log storms (suppression handles repeats).

#### 11.1.12 Future Extensions
- Add audit fields: `lo_raw_original`, `hi_raw_original`, `opt_flag_bits` after enabling audit mode.
- Support multi-valued limits (e.g. spec vs test limits) by layering additional resolvers or stacking states.

#### 11.1.13 Acceptance Criteria (Phase 1)
1. All existing limit-related tests pass unchanged or with minimal adaptation to new state names.
2. New tests cover matrix cases A–G (low) and equivalents for high.
3. Warnings emitted for contradictions & early default usage.
4. No regression in measurement counts or catalog rows on representative sample sets.
5. Resolver function documented and referenced exclusively; old scattered logic removed in a later migration step.

### 11.2 Integration Plan
This section defines the precise steps to introduce the unified resolver into the current codebase while maintaining backward compatibility and minimizing risk.

#### 11.2.1 Targeted Replacement Points
Current duplicated limit semantics appear in:
1. `_populate_test_catalog_from_ptr` (catalog population – lines ~400–470 in `ingest.py`).
2. `_extract_measurement` (measurement extraction – lines ~560–630 in `ingest.py`).

Both blocks manually interpret bits 4/5 (default) and 6/7 (no limit) with similar branching. They do not emit warnings for contradictions and silently ignore ambiguous None cases.

#### 11.2.2 Incremental Introduction Strategy
Phase 0 (Scaffolding):
1. Create new module: `cpkanalysis/limit_resolver.py` exporting `TestCache`, `LimitResolution`, and `resolve_limits` plus helper `resolve_for_record(opt_flag, lo_raw, hi_raw, cache, emit_issue)`.
2. Add lightweight unit tests for resolver only (matrix cases) – no integration changes yet.
3. Introduce feature toggle env var `CPK_ENABLE_LIMIT_RESOLVER` (default off) read early in `ingest.py`.

Phase 1 (Shadow Mode):
4. In `_populate_test_catalog_from_ptr`, after computing raw limit candidates & `opt_flg`, call resolver if toggle ON; compare resolver output with legacy outcome. If mismatch that is not due to a contradictory flag, emit INFO `LIMIT.SHADOW.DIVERGENCE` (suppressed after threshold). Do NOT alter existing catalog writes yet.
5. Capture aggregate divergence metrics (count per test) for evaluation.

Phase 2 (Dual Write):
6. Replace legacy branching with resolver output when toggle ON; still keep divergence comparison for 100 initial PTR records only (then disable to reduce overhead).
7. Update measurement path `_extract_measurement` similarly; maintain parity checks for first N (configurable, e.g. 200) measurements per file.

Phase 3 (Default On):
8. Flip default of `CPK_ENABLE_LIMIT_RESOLVER` to ON after green test matrix and zero unexpected divergences in sample lots.
9. Deprecate legacy code blocks (remove branching, keep a single call site). Document in CHANGELOG.

Phase 4 (Cleanup):
10. Remove shadow comparison logic and env guard after two minor releases; maintain resolver as authoritative.
11. Migrate tests referencing legacy semantics to use new states (`explicit`, `default`, `cleared`, `none`, `unchanged`).

#### 11.2.3 Code Modification Checklist
- Add `limit_resolver.py` with dataclasses & function.
- Inject resolver in `ingest.py` with feature toggle gating.
- Add issue codes:
    - `LIMIT.SHADOW.DIVERGENCE`
    - `LIMIT.OPTFLAG.CONTRADICTORY_BITS`
    - `LIMIT.CACHE.NO_DEFAULT_REFERENCED`
    - (Optional) `LIMIT.RAW.IGNORED_ON_DEFAULT` (INFO)
- Update `Architectural Plans/STDF_Core_Design.md` (this doc) referencing new module.
- Extend tests: `test_ingest_limits.py` add param set toggling env var.

#### 11.2.4 Testing Layers
1. Unit: Resolver matrix (low/high) + contradiction + early default reference.
2. Integration (ingest): Feed synthetic PTR stream covering sequences: explicit → default → cleared → explicit re-establish; assert catalog evolution states.
3. Regression: Run existing ingestion pipeline on representative STDF files; assert identical final numeric limit values and measurement counts pre vs post toggle.
4. Performance: Benchmark ingest throughput with and without resolver (expect negligible delta <1%). If >3% slowdown, profile branch conditions.

#### 11.2.5 Rollback Plan
If unexpected divergence occurs in production sample lots:
1. Disable via env `CPK_ENABLE_LIMIT_RESOLVER=0`.
2. Collect divergence logs for root-cause (store NDJSON `limit_resolver_shadow.log`).
3. Patch resolver or adjust precedence ordering.

#### 11.2.6 Risk Assessment & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Silent legacy assumptions differ from spec | Incorrect limit states | Shadow mode divergence logging |
| Performance regression | Slower ingest | O(1) resolver; micro-benchmark & optimize branch order |
| Overlogging contradictions | Log volume spike | Taxonomy suppression threshold (25) reused |
| Missed edge where default referenced pre-initialization | Inaccurate limit retention | Explicit warning + unchanged state logic |
| Env toggle forgotten | Inconsistent deployments | Include toggle presence in startup banner / run summary |

#### 11.2.7 Success Metrics
1. Zero unexpected divergences after Phase 2 on validation datasets.
2. No increase in ingestion warnings except mapped contradictions.
3. Test coverage >90% for resolver branch paths.
4. Documentation references updated (README limit semantics section).

#### 11.2.8 Follow-Up Enhancements
- Expose resolver statistics in debug summary (counts of explicit/default/cleared per test).
- Optional metrics export (Prometheus) for limit state churn.
- Extend resolver to handle multi-step hierarchical defaults (future STDF variants or test plan overlays).


## 12. Streaming & Batch Modes
- `stream_measurements` yields `RawMeasurement` items (low overhead, user decides materialization).
- `ingest_files` optionally builds DataFrame;
  - If `batch_size` set: internally aggregates and flushes to Arrow / Parquet before constructing final DataFrame (or returns path only).
- Future: memory mapping and direct Arrow dataset registration for analytic engines.

## 13. Performance Targets (Initial)
Metric | Target | Rationale
------|--------|----------
Throughput | >= 200k measurements/sec on single core | Competitive baseline.
Memory overhead | < 3x raw measurement footprint during parse | Avoid blow-up.
Invalid measurement tracking overhead | <5% of total ingest time | Acceptable cost.
Site detection | O(first 1k records) per file | Quick user feedback.

## 14. Migration Plan (Phased)
Phase | Action | Risk Mitigation
------|--------|----------------
0 | Snapshot current outputs (golden JSON fixtures). | Regression tests.
1 | Introduce unified limit resolver (shadow usage). | Dual-run comparison.
2 | Add `stdf_core` package with adapters; keep current API returning identical shapes. | Feature flags.
3 | Switch pipeline to new facade (`STDFIngestor`). | Backwards compatibility shim.
4 | Externalize package (separate repo, add dependency). | Semantic versioning.
5 | Enable streaming mode for large files (optional). | Opt-in rollout.
6 | Deprecate old ingestion functions and remove duplication. | Clear version notes.

## 15. Risks & Mitigations
Risk | Mitigation
-----|-----------
Silent schema drift | Embed schema version + enforce tests.
Unit scaling confusion | Provide both raw & scaled; document transformation.
Memory spike on huge files | Implement batch flush early (Phase 2/3).
Complex OPT_FLAG logic regressions | Maintain exhaustive flag matrix tests.
Multi-file limit conflicts | Add conflict detection + warning codes.

## 16. Testing Strategy
Test Class | Focus
-----------|------
Unit | Flag & limit resolution permutations.
Property | Idempotence (re-run ingestion yields same catalog).
Golden | Known STDF file outputs vs snapshot.
Performance | Large synthetic STDF stress (vary record mix).
Fault Injection | Truncated files, malformed flags.
Streaming Consistency | Stream vs bulk ingestion equivalence.

## 17. Open Questions (Please Provide Input)
1. Required minimum Python version / dependency constraints? (E.g. pin pandas/pyarrow versions?)
    - Decided: Target Python 3.11.1 for development; minimum supported version will be 3.10 to widen adoption. Pin baseline dependency floors (pandas >=2.1,<3; pyarrow >=15,<17) in later phase.
2. Should unit scaling default to raw values (defer display composition) for broader reuse?
    - Decision: Provide scaled values and scaled display units by default; retain raw numeric and raw unit fields as optional columns for audit/ML use.
3. Do we need binary offset or record index tracking for traceability/debugging?
    - Decision: Yes. Add `record_index` (sequential per file) and attempt `byte_offset` when available from underlying reader; include both in measurement and raw record streams.
4. Any need to expose additional STDF records now (e.g. TSR, MPR) for future analysis contexts?
    - Decision: Expose all records via a generic raw record streaming API; measurement extraction becomes a layer atop this. Non-measurement records surfaced unchanged.
5. How critical is Arrow dataset integration vs plain Parquet for your downstream tooling?
    - Decision: High importance. Adopt partitioned Parquet (Arrow Dataset) layout from the outset for scalability and pruning.
6. Should invalid measurements be optionally preserved with a flag instead of dropped (for some ETL consumers)?
    - Decision: Yes. Option `include_invalid` will retain invalid rows with `invalid_reason`; default remains to exclude from standard stats.
7. Acceptable trade-off: one Parquet per source vs single concatenated file? (Partitioning benefits?)
    - Decision: Use one Parquet file per STDF source under partitioned directory structure (e.g. `lot_id=<lot>/wafer_id=<wafer>/file=<basename>.parquet`). Provide optional compaction utility for very small fragments.
8. Need built-in parallel file parsing now or can defer to later phase?
    - Decision: Implement parallel parsing in Phase 1 with configurable `max_workers` (upper bound ~30 concurrent files). Choose threading if STDF reader releases GIL; fall back to process pool otherwise.
9. Preferred warning/error surface: return list objects, emit logging, or structured JSON sidecar?
    - Decision: Emit structured logging plus retain programmatic lists (`warnings`, `errors`) and optional JSON sidecar for reproducibility.
10. Any regulatory/compliance constraints (e.g. traceability requiring raw unchanged values)?
    - Deferred: Audit mode (raw values, file checksum, offsets) can be added in later phase; not required Phase 1.
11. How often do unit scale changes mid-test occur, and should we surface a warning in that case?
    - Decision: Rare event; emit `SemanticWarning` if scale changes for a test key and record both old/new values.
12. Should we capture HEAD_NUM and SITE_GRP per measurement row for richer multi-site analysis?
    - Decision: Yes. Add `head_num` and `site_group` columns to measurement schema; derive from SDR context.
13. Are there privacy/security considerations restricting absolute file paths in outputs?
    - Decision: Include absolute paths; emit `FileAccessWarning` if unreadable/locked. Provide optional `--strip-paths` future flag if needed.
14. Is per-wafer/lot metadata extraction required (e.g. from MIR/WIR) in near term?
    - Decision: Yes. Parse MIR/WIR for lot_id, wafer_id, test_program, start/end timestamps; include partition key alignment (`lot_id`, `wafer_id`).
15. Need language-agnostic schema export (Avro/Arrow IPC) for other ecosystems?
    - Deferred: Schema export (Avro/Arrow IPC/JSON) parked until ecosystem requirements clarified in later phase.

## 18. Next Steps (After Your Feedback)
- Finalize data contracts & enums.
- Implement unified limit/flag resolver in isolation with exhaustive tests.
- Build streaming ingestion prototype against a synthetic STDF generator.
- Integrate schema version tagging & Parquet metadata embedding.

---
*Please respond to the Open Questions section; your answers will drive the refinement of this design.*
