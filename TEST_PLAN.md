# CPK Analysis Post-Processing Test Plan

This document outlines the verification strategy for the post-processing pipeline, plugin framework, and supporting interfaces. It targets both human QA engineers and AI-powered testers.

---

## 1. Test Environment

- **Operating System**: Windows 11 (latest patches).
- **Python Version**: 3.11.x (required baseline); optional 3.12 smoke if available.
- **Dependencies**: Install via `pip install -r requirements.txt`; ensure `Submodules/istdf` initialized.
- **Workspace Layout**: Clone repository, verify `temp/` is writable, and create dedicated `cpk_plugins/` directories per scenario.

---

## 2. Test Scenarios

### 2.0 Execution Modes

- **Autonomous Mode**: Tasks that can be executed end-to-end by an AI agent through scripted commands and automated assertions. These scenarios avoid interactive prompts and rely on deterministic command outputs.
- **Human-Assisted Mode**: Scenarios requiring manual input, subjective verification, or bespoke plugin authoring beyond current automation hooks. These tests provide step-by-step guidance for a human tester.

### 2.1 Autonomous Scenarios

#### 2.1.1 Baseline CLI Regression

- Run `python -m cpkanalysis.cli run Sample/lot2.stdf Sample/lot3.stdf --output temp/Run.xlsx`.
- Verify spinner outputs, workbook structure, and JSON metadata (no plugins listed).

#### 2.1.2 Plugin Discovery and Execution

- Enable builtin plugin via profile; run CLI and confirm console log plus metadata entry.
- Copy `examples/plugins/demo_summary.py` and `examples/plugins/sample_summary.toml` into `./cpk_plugins/`; execute:
  - `--plugin enable:sample.summary_logger`
  - `--plugin param:sample.summary_logger:message="[override] rows={rows}"`
  - `--plugin priority:sample.summary_logger:42`
- For each run, ensure console reflects overrides and metadata records final parameters/priority.

#### 2.1.3 Validation Mode

- Command: `python -m cpkanalysis.cli run Sample/lot2.stdf --validate-plugins --plugin enable:builtin.summary_logger`.
- Confirm temporary workbook path is reported, temp files (including copied template) are deleted, and original template remains intact.

#### 2.1.4 Error Handling

- `--template nonexistent.xlsx` → expect clear error.
- `--plugin enable:unknown.plugin` → expect abort with available plugin list.
- Malformed `post_processing_profile.toml` → warning and default behaviour.
- Modify demo plugin to raise on handle → ensure pipeline surfaces exception (fail-fast).

#### 2.1.5 Performance Measurement

- Run pipeline on ≥10 STDF files with/without plugins; capture `stage_timings`.
- Ensure plugin-enabled run stays within ~10% of baseline (unless plugin intentionally heavy).

#### 2.1.6 Unit & Integration Automation

- Pytest suites for:
  - `PluginRegistry` discovery, manifest parsing, failure cases.
  - `plugin_profiles` load/save fidelity.
  - `_prepare_plugin_configs` overrides/conflict detection.
- Integration tests:
  - Mock plugin observing event dispatch and context updates.
  - Validation mode ensuring temp resources removed post-run.

### 2.2 Human-Assisted Scenarios

#### 2.2.1 Console Workflow

- Run `python -m cpkanalysis.gui` and walk through prompts manually (STDF paths, chart toggles, plugin selections).
- Evaluate usability of prompts and confirm resulting workbook/metadata.

#### 2.2.2 Persistence & Profiles

- Through console UI, adjust plugin settings; restart to confirm persistence.
- Run CLI without overrides to ensure profile honoured; then apply CLI overrides to confirm warnings and final metadata.

#### 2.2.3 Concurrency & Ordering (Future hooks)

- When concurrency controls land, create multi-plugin scenarios to validate priority ordering and thread-safety handling.

---

## 3. Tooling Checklist

- `python -m compileall cpkanalysis examples`
- `pip install -e .[dev]` (if test harness requirements defined)
- `pytest` (to be configured)
- Optionally integrate with CI (GitHub Actions) for multi-platform runs.

---

## 4. Reporting

For each scenario capture:
- Command executed.
- Environment details (OS, Python version).
- Console output snippet (especially plugin logs).
- Post-run artifacts (workbook path, metadata snippet).
- Timing metrics (from metadata).
- Any issues or discrepancies with references to logs and file paths.

---

## 5. Future Enhancements

- Add automated stress tests with synthetic STDF data.
- Implement plugin timeout configuration and verify enforcement.
- Extend validation to cover workbook mutations (e.g., verifying charts updated).
- Integrate linting/static analysis (ruff, mypy) for plugin modules.

---

*Maintained by CPK Analysis QA team. Update this plan whenever the plugin architecture or console/CLI behaviour changes.*
