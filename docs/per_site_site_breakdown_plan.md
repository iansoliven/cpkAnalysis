Intent: Introduce optional per-site aggregation with minimal disruption by shipping it in staged, verifiable increments (Phases 1-6 implemented).

Lean Incremental Plan
- Phase 1 – Surface-Level Toggle & Detection
  - Add  `enable_site_breakdown` + `site_data_status` to `AnalysisInputs`. 
  - Implement lightweight  `detect_site_support(sources)` in `ingest` and expose a `has_site_data(frame)` helper. 
  - GUI/CLI: after STDF selection, probe once; if sites exist prompt the user, otherwise warn and gate the flag.
  - Tests: coverage for detection helper and UI prompt logic (CLI path).

- Phase 2 – Data Plumbing
  - Extend ingestion to capture  `SITE_NUM` into a new `site` column. 
  - Update parquet schema expectations and ingestion tests accordingly.
  - No behavioural change elsewhere; ensures existing pipeline still passes.

- Phase 3 – Computation Layer
  - Support dynamic grouping keys so outlier filtering, summary, yield, and Pareto can operate per site.
  - Add parallel \ by-site\ functions that return new DataFrames while keeping legacy outputs intact.
  - Pipeline context stores per-site outputs under the opt-in flag; emit warnings when requested but unavailable.
  - Metadata/result JSON updated to reflect site request/availability; add stats tests.

- Phase 4 – Charts & CPK Report
  - Introduce optional parameters in `workbook_builder` for per-site tables/plots.
  - Render per-site histograms/CDF/time-series alongside legacy plots (COL_STRIDE offsets).
  - Update CPK report with Site column + hyperlinks; maintain compatibility when site disabled.
  - Tests: workbook layout positioning, plot links, CPK report contents.

- Phase 5 – Yield/Pareto Layout
  - Append per-site yield and Pareto sections to the right of the legacy block.
  - Ensure table naming, chart placement, and widths remain consistent.
  - Tests: verify site sections render for multiple sites without overlapping.

- Phase 6 – Docs, UX, Metadata
  - Update README/docs, CLI/GUI messaging, and metadata docs for the per-site workflow.
  - Ensure CLI prints pipeline warnings (e.g., missing SITE_NUM) and metadata reports requested/available/generated with row counts.
  - Run full regression (pytest) and capture recommendations for deeper test coverage.

Additional Test Recommendations
- Site detection & validation (detect_site_support and has_site_data edge cases).
- Per-site statistics correctness (grouping, limit sources, Pareto).
- Workbook layout positioning (plot anchors, site blocks, hyperlinks).
- Pipeline integration (context fields, site flags, warnings).
- CLI/GUI prompts and flags (explicit vs. auto detection).
- Outlier filtering site-aware behaviour.
- Edge cases (single-device sites, mixed types, large site sets).
- Metadata/audit trail (row counts, availability).
- Backward compatibility (site disabled matches legacy output).
