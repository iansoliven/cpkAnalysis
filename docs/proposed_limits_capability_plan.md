# Proposed Limits (GRR-Based) Capability Plan

This document outlines the end-to-end design for adding a GRR-aware “calculate proposed limits” option within the post-processing phase of the CPK analysis toolchain. The plan spans user interaction (CLI & GUI), data ingestion, analytical logic, workbook updates, chart generation, and verification.

---

## 1. User Experience & Entry Points

### 1.1 Feature Access
- **CLI:** augment `cpkanalysis/cli.py` so both the primary `run` command (when `--postprocess` is set) and the standalone `post-process` sub-command expose a new action titled “Proposed Limits (GRR)”. Invocation drops the user into an interactive prompt sequence before calculations begin.
- **GUI:** extend `cpkanalysis/gui.py` to surface the same submenu option in the console-guided workflow. Ensure parity with CLI prompts and behavior.
- **Post-processing Menu:** update `cpkanalysis/postprocess/menu.py` to register a new `ActionDefinition` (e.g., key `calculate_proposed_limits_grr`). The existing three actions remain unchanged; the new option sits beneath “Calculate Proposed Limits” to preserve familiarity.

### 1.2 Prompt Flow
1. **GRR Availability:** use `PostProcessIO.confirm` to ask “Is GRR data available for proposed limit calculation?” If the user declines, abort gracefully (`ActionCancelled` with message).
2. **Directory Selection:** when confirmed, call `PostProcessIO.prompt` to capture the GRR directory path (support blank default). Resolve the path:
   - Validate that `Total_GRR.xlsx` exists within the directory.
   - If missing or unreadable, warn and allow the user to re-enter or cancel.
3. **CPK Targets:** prompt for numeric `Minimum FT Cpk` and `Maximum FT Cpk`. Enforce positivity and that `min <= max`. Persist the values in the action’s metadata/audit log.
4. **Scope Selection:** mirror the existing `calculate_proposed_limits` action by allowing “All tests” vs “Single test”. Reuse `_prompt_scope` where convenient or craft a sister helper to avoid target Cpk prompts from the legacy flow.

### 1.3 Replay & Automation
- Record all collected parameters (directory, min/max Cpk, scope, selected test identifiers) in the returned `audit` payload and `replay_params`. This enables re-running the same calculation without reprompting.

---

## 2. Data Ingestion & Preparation

### 2.1 GRR Workbook Parsing
- Implement a loader, e.g., `_load_grr_table(grr_dir: Path) -> pd.DataFrame`.
- Expected file: `<grr_dir>/Total_GRR.xlsx`.
- Default sheet: `"GRR data"` (fallback to first sheet if not found, but warn).
- Header normalization:
  - Test number columns: `["Test #", "Test Number", "Test Num", "Test"]`.
  - Test name columns: `["Test Name", "Description"]`.
  - Measurement units: `["Unit", "Units"]`.
  - Spec walls: `["Min\nSpec", "Spec Min", "Lower Spec"]` and `["Max\nSpec", "Spec Max", "Upper Spec"]`.
  - GRR magnitude: prefer `["Worst R&R"]`; fallback to normal R&R columns if necessary.
- Normalization steps:
  - Strip leading “T/t” from test numbers, remove leading zeros, convert missing numbers to empty strings.
  - Collapse repeated spaces and lowercase test names for consistent matching.
  - Ensure numeric fields are `float` (coerce errors to NaN and drop rows lacking both spec walls and GRR).
- Output columns:
  - `test_number`, `test_name`, `unit`, `spec_ll`, `spec_ul`, `grr_full`, plus optional metadata (e.g., `grr_source`).

### 2.2 Context Data Dependencies
- **Summary Sheet (`context.summary_frame`)**
  - Source of `mean`, `stdev`, original `CPK`, unit, and file/test identifiers.
  - Apply `_safe_float` and normalization analogous to the GRR loader.
  - Normalize unit strings (trim, uppercase, handle aliases like `V` → `VOLTS`). If the normalized value conflicts with the GRR unit, record the mismatch for later handling (§6.1).
- **Template Sheet (`context.template_sheet`)**
  - Use `sheet_utils.build_header_map` to detect columns: test name/number, existing spec/ATE/what-if/proposed columns.
  - Record column indexes for writing new values (see §4).
- **Limits Sheet (`context.limits_frame`)**
  - Reference current spec and FT limits for chart comparisons; not directly required for calculations but useful for verifying mismatches.
- **Measurements Frame (`context.measurements_frame`)**
  - Required for re-computing yield loss (same mechanics as the current proposed-limit action).
  - If empty, warn that yield cannot be recomputed but continue with limit proposals.

### 2.3 Record Assembly
- For each selected test (based on scope):
  1. Locate summary row by key `(file, test_name, test_number)`.
  2. Locate GRR row by `(test_number, test_name)` pair; fall back to name-only match if number blank.
  3. Locate template rows via `sheet_utils.find_rows_by_test`.
  4. If any component missing, issue warning and skip.
- Construct a structured record:
  ```python
  ProposedLimitRecord(
      descriptor: TestDescriptor,
      mean: float,
      stdev: float,
      spec_ll_orig: float,
      spec_ul_orig: float,
      ft_ll_orig: float,
      ft_ul_orig: float,
      grr_full: float,
      grr_half: float,
      unit: str,
      template_rows: list[int],
  )
  ```
  - `ft_ll_orig` / `ft_ul_orig` derived from existing ATE or PDS columns.
  - `grr_half = grr_full / 2`.

---

## 3. Analytical Logic

### 3.1 Guardband Calculation
For each record:
1. **Initial spec & Cpk:** compute `cpk_orig = min((spec_ul_orig - mean), (mean - spec_ll_orig)) / (3 * stdev)`. Treat zero/NaN stdev as blocking; warn and skip.
2. **Candidate guardbands:**
   - `spec_ll_gb_100 = spec_ll_orig + grr_full`
   - `spec_ul_gb_100 = spec_ul_orig - grr_full`
   - `spec_ll_gb_50 = spec_ll_orig + grr_half`
   - `spec_ul_gb_50 = spec_ul_orig - grr_half`
3. **Cpk evaluation:** compute `cpk_gb_100` and `cpk_gb_50` using the same formula but with guardbanded spec walls.

### 3.2 Guardband Selection Ladder
- **Preference Order:**
  1. Use 100 % GRR if `cpk_gb_100 >= CPK_min`.
  2. Else attempt 50 % GRR.
  3. If both violate the minimum, commit to 50 % GRR but plan to widen the spec (see below).
- Track the decision for audit: `guardband_choice` ∈ {"grr_full", "grr_half"}.

### 3.3 Spec Adjustment
When the chosen guardband leaves `cpk_guardband < CPK_min`:
1. Identify the limiting side (lower vs upper tail) by comparing deltas `(mean - ll)` vs `(ul - mean)`.
2. Solve for the required spec boundary:
   - For lower tail: `spec_ll_adj = mean - 3 * stdev * CPK_min`
   - For upper tail: `spec_ul_adj = mean + 3 * stdev * CPK_min`
3. Ensure the adjusted spec still accommodates the guardband shrinkage:
   - Example: `spec_ll_proposed = spec_ll_adj` and `ft_ll = spec_ll_proposed + guardband_choice`.
4. If both sides require widening (e.g., due to symmetric distribution), adjust both.
5. If widening pushes the FT window past the original spec walls even after adjustment, issue a warning that minimum Cpk cannot be satisfied; still record the best-effort limits.

### 3.4 FT Guardband Targets
Once spec walls are finalized:
1. Compute the FT guardbanded limits relative to mean to hit the target Cpk window:
   - Candidate `ft_ll_candidate = mean - 3 * stdev * target_cpk`
   - Candidate `ft_ul_candidate = mean + 3 * stdev * target_cpk`
2. Determine actual FT target:
   - If `cpk_guardband > CPK_max`: tighten (reduce interval) until Cpk equals `CPK_max`.
   - If `< CPK_min`: widen until equals `CPK_min`.
   - Otherwise, keep guardband implied by chosen GRR (i.e., FT Cpk equals current `cpk_guardband`).
3. Apply the guardband offset driven by GRR choice:
   - For lower side: `ft_ll_proposed = max(spec_ll_proposed + guardband_value, ft_ll_candidate)`.
   - For upper side: `ft_ul_proposed = min(spec_ul_proposed - guardband_value, ft_ul_candidate)`.
   - Enforce `ft_ll_proposed < ft_ul_proposed`; if violated, clamp symmetrically around mean and log warning.

### 3.5 Capability Metrics
Recalculate:
- `cpk_ft_proposed = min((ft_ul_proposed - mean), (mean - ft_ll_proposed)) / (3 * stdev)`
- `cpk_spec_proposed = min((spec_ul_proposed - mean), (mean - spec_ll_proposed)) / (3 * stdev)`
- Yield loss against the proposed FT window via `_compute_yield_loss`.

Store these metrics in the result bundle for workbook updates, audit trails, and chart annotations.

---

## 4. Workbook Mutations

### 4.1 Template Columns
- Existing columns: `LL_PROP`, `UL_PROP`, `CPK_PROP`, `%YLD LOSS_PROP`.
- Add four new columns two spaces to the right of `%YLD LOSS_PROP`:
  1. `Proposed Spec Lower`
  2. `Proposed Spec Upper`
  3. `CPK Proposed Spec`
  4. `Guardband Selection` (textual indicator such as `"100% GRR"` / `"50% GRR"`) – satisfies the request for a fourth column.
- Implementation details:
  - Determine base column index using `sheet_utils.build_header_map`.
  - If the header row lacks these titles, insert them in the appropriate cells.
  - For each target row write corresponding values; leave blank for skipped tests.

### 4.2 Proposed FT Limits & Metrics
- Write `ft_ll_proposed` and `ft_ul_proposed` into `LL_PROP` / `UL_PROP`.
- Update `CPK_PROP` and `%YLD LOSS_PROP` when proposals change or previously blank.
- Populate the new spec columns with `spec_ll_proposed`, `spec_ul_proposed`, `cpk_spec_proposed`, and `guardband_choice`.
- Record decisions in metadata via `_ensure_proposal_state` (extend to carry spec info and guardband type).
- Mark context as dirty whenever any cell is modified.

### 4.3 Metadata & Audit
- Extend stored metadata under `post_processing_state["proposed_limits"]` to include spec proposals and guardband choice, allowing change detection on subsequent runs.
- Audit payload (returned dict) should include:
  - Scope (all vs single)
  - Parameter set (GRR directory, min/max Cpk)
  - Per-test entries: guardband selection, whether spec was widened, old vs new FT/spec limits, resulting Cpks.

---

## 5. Chart Integration

### 5.1 LimitInfo Enhancements
- Update `postprocess/charts.py`’s `LimitInfo` dataclass to store `proposed_spec_lower` and `proposed_spec_upper`, along with the existing proposed FT limits.
- Modify `_collect_limit_info` to read the new columns from the template sheet and populate the extended fields.

### 5.2 Marker Rendering
- Expand `_build_markers` to add markers for proposed spec walls when `include_proposed` is true:
  - Label them distinctly (e.g., “Proposed Spec Lower/Upper”) using a dashed linestyle and a new color (reuse `PROPOSED_COLOR`).
  - Maintain existing proposed FT markers (“Proposed Lower/Upper”).

### 5.3 Chart Sheets
- Create a helper akin to `_refresh_all_tests` that, after refreshing the original charts, constructs parallel sheets for proposed guardband views:
  - Naming convention: append `" (Proposed)"` to the existing histogram/CDF/time-series sheet names, or create new sheets if duplicates would clash.
  - Charts should plot the original measurement data and overlay:
    - Original STDF/spec limits (as before).
    - Proposed FT limits.
    - Proposed spec limits.
  - Text annotations: display both `Original Cpk` and `Proposed FT Cpk`, along with `Proposed Spec Cpk`.
- Ensure axes are scaled to encompass both original and proposed limits; reuse existing `_load_axis_ranges` logic to maintain consistency across runs.

---

## 6. Error Handling & Messaging

- **Missing Data:** on absent summary stats, spec walls, or GRR values, warn via `io.warn` and skip the test gracefully.
- **Non-positive Stdev:** log warning that Cpk cannot be computed; do not write proposals.
- **Invalid GRR Directory:** prompt user again, allowing cancellation.
- **Cpk Target Violations:** if after adjustments the FT Cpk still falls outside `[min, max]`, note in warnings and continue with best effort.
- **Workbook Columns Missing:** if the template lacks required columns despite header scan, abort the action with an instructive message (similar to the existing action’s behavior).
### 6.1 Unit Mismatches
- Perform unit normalization when ingesting both Summary and GRR sources (strip whitespace, uppercase, convert common aliases).
- If the normalized units remain different, treat it as a hard mismatch:
  - Warn the user with details (`test key`, GRR unit, workbook unit) and skip the test by default so no stale proposals linger.
  - Offer an override prompt to proceed using the workbook unit when the user explicitly confirms; log the decision.
  - Record mismatches and override outcomes in the audit log and metadata so follow-up runs highlight the unresolved discrepancy.

---

## 7. Testing & Validation Strategy

1. **Unit Tests (if time allows):**
   - Create test cases around the guardband selection ladder using synthetic data frames.
   - Validate spec widening math and resulting Cpks.
2. **Integration Tests / Manual Verification:**
   - Use a sandbox workbook with known GRR data mirroring `LimitSetting_QUAL.xlsx`.
   - Run the new action with edge-case parameters:
     - High Cpk cap forcing additional tightening.
     - Low Cpk minimum requiring spec widening.
     - Missing GRR rows.
   - Inspect resulting workbook:
     - New columns populated correctly.
     - Proposed FT/spec limits match calculations.
     - Charts created/updated with new markers and annotations.
3. **Regression:** rerun existing “Calculate Proposed Limits” action to ensure no interference.

---

## 8. Implementation Breakdown

1. **Infrastructure:**
   - New module `postprocess/grr_limits.py` (optional) housing data models and calculation functions to keep `actions.py` lean.
   - Shared utilities for normalization (test keys, numeric parsing).
2. **Action Wiring:**
   - Register handler in `menu.py`.
   - CLI/GUI updates to trigger the action.
3. **Workbook Column Augmentation:**
   - Utility (`_ensure_proposed_spec_columns`) to create headers, invoked before writes.
4. **Core Calculation:**
   - Iterate selected tests, call calculator, write outputs, update metadata.
5. **Chart Refresh:**
   - Extend `charts.refresh_tests` to detect presence of proposed spec columns and adjust markers automatically; optionally add `refresh_proposed_charts` for dedicated sheets.
6. **Documentation & Help Text:**
   - Update README or CLI help message describing the new option and required GRR file.

---

## 9. Future Considerations

- Allow per-test overrides for Cpk targets via template metadata.
- Support ingesting multiple GRR files (e.g., per lot) with user selection.
- Add batch mode (non-interactive) by allowing parameters via CLI flags.
- Persist guardband choices back into metadata for cross-session auditing.

---

With the above plan we can integrate GRR-based proposed limit generation into the existing post-processing pipeline while maintaining backward compatibility and giving users clear visibility into both the FT and spec adjustments derived from their metrology studies.
