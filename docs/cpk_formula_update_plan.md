# CPK Formula Update Plan

## 1. Goals
- Convert the template workbook's manual CPK cells into formulas so they react to any edits in the stats or limit columns.
- Keep the existing pipeline output ("precalc CPK") untouched so charts and exports continue to read the same data.
- Surface a Spec CPK in the template without adding visible helper columns by storing intermediate work on a hidden sheet.

## 2. Workbook Scope
- Workbook: the user-supplied template path (e.g., `cpkTemplate/cpk_report_output_template.xlsx`).
- Primary worksheet: template sheet name determined earlier in the pipeline and passed into the script; data rows start at row 11.
- Reference worksheet: `calculation details` (describes column intent).
- Key inputs: limits (`E:F`, `G:H`, `AA:AB`), statistics (`J`, `K`, `L`, `M`), and CPK families (`N:P`, `R:X`, `AC`).

## 3. Formula Design

### 3.1 Limit Precedence
- Base CPK (`CPL`, `CPU`, `CPK`) favors ATE limits (`LL_ATE`, `UL_ATE`); if either side is missing it falls back to the datasheet spec (`LL_SPEC`, `UL_SPEC`).
- Spec-only calculations always use datasheet spec limits.
- Derived variants (`2.0`, `3IQR`, `Proposed`) keep using their current limit sources (`LL_2CPK`/`UL_2CPK`, `LL_3IQR`/`UL_3IQR`, `LL_PROP`/`UL_PROP`).

### 3.2 Per-Row Formulas (row index r >= 11)
| Column | Purpose | Formula Logic |
| --- | --- | --- |
| `N` (`CPL`) | Lower capability vs. active limits | `=IF(OR($Lr<=0,AND($Gr="",$Er="")),"",($Jr-IF($Gr<>"",$Gr,$Er))/(3*$Lr))` |
| `O` (`CPU`) | Upper capability vs. active limits | `=IF(OR($Lr<=0,AND($Hr="",$Fr="")),"",(IF($Hr<>"",$Hr,$Fr)-$Jr)/(3*$Lr))` |
| `P` (`CPK`) | Active capability | `=IF(OR($Nr="",$Or=""),"",MIN($Nr,$Or))` |
| `AE` (`CPK_SPEC`) | Datasheet capability | `=IF('cpk_helpers'!$Cr="","",'cpk_helpers'!$Cr)` |
| `R` (`LL_2CPK`) | Mean minus six stdev | `=IF($Lr>0,$Jr-6*$Lr,"")` |
| `S` (`UL_2CPK`) | Mean plus six stdev | `=IF($Lr>0,$Jr+6*$Lr,"")` |
| `T` (`CPK_2.0`) | Capability vs. mean +/- six stdev | `=IF(AND($Lr>0,$Rr<>"",$Sr<>""),MIN(($Jr-$Rr)/(3*$Lr),($Sr-$Jr)/(3*$Lr)),"")` |
| `V` (`LL_3IQR`) | Median minus three IQR | `=IF(AND($Kr<>"",$Mr<>""),$Kr-3*$Mr,"")` |
| `W` (`UL_3IQR`) | Median plus three IQR | `=IF(AND($Kr<>"",$Mr<>""),$Kr+3*$Mr,"")` |
| `X` (`CPK_3IQR`) | Capability vs. median +/- three IQR | `=IF(AND($Lr>0,$Vr<>"",$Wr<>""),MIN(($Jr-$Vr)/(3*$Lr),($Wr-$Jr)/(3*$Lr)),"")` |
| `AC` (`CPK_PROP`) | Capability vs. proposed limits | `=IF(AND($Lr>0,$AAr<>"",$ABr<>""),MIN(($Jr-$AAr)/(3*$Lr),($ABr-$Jr)/(3*$Lr)),"")` |

> Hidden helper sheet (`cpk_helpers`): columns `A`, `B`, and `C` hold the Spec CPL/CPU/CPK formulas so the primary sheet stays uncluttered.

## 4. Implementation Steps
1. **Template Audit**  
   - Confirm the target columns do not already carry formulas or named ranges that would conflict.  
   - Verify the header row (row 10) and maximum data row (currently 10030).
2. **Scripted Injection**  
   - Use Python + openpyxl to populate formulas from row 11 to the last populated row, taking the pipeline-provided template sheet name as input.  
   - Rebuild the hidden `cpk_helpers` sheet on each run, writing the Spec CPL/CPU/CPK helper formulas there and marking the sheet hidden.  
   - Update the visible sheet headers (e.g., ensure `AE10` is `CPK_SPEC`) without moving existing columns.
3. **Manual Validation**  
   - Load a copy of the workbook in Excel, enable calculation, and edit sample stats/limits to confirm every CPK column reacts immediately.  
   - Spot-check at least one row where ATE limits are blank to ensure the spec fallback works.  
   - Confirm the hidden helper sheet stays hidden after saving.
4. **Documentation & Handoff**  
   - Share the updated usage notes with the post-processing team.  
   - Let testers know the spreadsheet now updates automatically so they can focus on verifying values rather than typing them.

## 5. Testing Strategy
- **Automated**: No code-based comparison yet; deferred until priorities allow.
- **Manual**: QA spot-check of representative rows covering ATE-only, spec-only, and proposed-limit scenarios.

## 6. Dependencies & Risks
- Requires `openpyxl` (already in the project).  
- If limit precedence expectations change, formulas must be updated to match.  
- Downstream tooling must tolerate the additional hidden sheet and the new `CPK_SPEC` column; validate before release.

## 7. Follow-Up Items
- Decide whether `%YLD LOSS_SPEC` should also become formula-driven.  
- Revisit the automated CPK comparison once bandwidth opens up.  
- Update Option 4 documentation to mention the `CPK_SPEC` column and hidden helper sheet.
