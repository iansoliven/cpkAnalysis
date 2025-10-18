Intent: Introduce optional per-site aggregation with minimal disruption by shipping it in staged, verifiable increments.

Lean Incremental Plan
- Phase 1 – Surface-Level Toggle & Detection
  - Add `enable_site_breakdown` + `site_data_status` to `AnalysisInputs`.
  - Implement lightweight `detect_site_support(sources)` in `ingest` and expose a `has_site_data(frame)` helper.
  - GUI/CLI: after STDF selection, probe once; if sites exist prompt the user, otherwise warn and gate the flag.
  - Tests: coverage for detection helper and UI prompt logic (CLI path).

- Phase 2 – Data Plumbing Only
  - Extend ingestion to capture `SITE_NUM` into a new `site` column (no downstream usage yet).
  - Update parquet schema expectations and ingestion tests accordingly.
  - No behavioural change elsewhere; ensures existing pipeline still passes.

- Phase 3 – Computation Layer
  - Make outlier filtering, summary, yield, and Pareto helpers accept dynamic group keys.
  - Add parallel “by-site” functions that return new DataFrames but don’t affect the workbook yet.
  - Pipeline context stores per-site outputs under the opt-in flag; emit warnings when requested but unavailable.
  - Metadata/result JSON updated to reflect request/availability. Add tests around new stats outputs.

- Phase 4 – Workbook & Charts
  - Introduce optional parameters in `workbook_builder` for per-site tables/plots.
  - Implement side-by-side layout (legacy left, per-site blocks right) respecting COL_STRIDE, with clear site labels.
  - Adapt CPK report hyperlinks to include site context; keep legacy behaviour unchanged when flag off.
  - Unit/integration tests to inspect sheet structure and hyperlinks.

- Phase 5 – Yield/Pareto Sheet Enhancements
  - Append per-site yield/pareto blocks to the right of each file’s legacy section.
  - Validate row/column formatting and chart placement via tests.

- Phase 6 – Documentation & Final Polish
  - Update README/help text, CLI usage, and GUI tooltips.
  - Ensure metadata/console messaging clearly reports when per-site outputs are generated or skipped.
  - Run full regression tests (unit + integration) before merging.
