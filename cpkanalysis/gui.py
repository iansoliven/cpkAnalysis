"""Minimal text-driven GUI scaffold for the CPK workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .cli import _select_template
from .models import AnalysisInputs, OutlierOptions, SourceFile
from .pipeline import run_analysis


@dataclass
class ApplicationState:
    """In-memory representation of the current GUI configuration."""

    workspace_path: Path = Path.cwd()
    sources: list[Path] = field(default_factory=list)
    template_root: Path | None = None
    output_path: Path = Path("CPK_Workbook.xlsx")
    outlier_method: str = "none"
    outlier_k: float = 1.5
    include_histogram: bool = True
    include_cdf: bool = True
    include_time_series: bool = True


class CPKAnalysisGUI:
    """Very small text-based facade mirroring the future GUI flow."""

    def __init__(self, state: ApplicationState | None = None) -> None:
        self.state = state or ApplicationState()

    def launch(self) -> None:
        """Launch the interactive console session."""
        print("=== CPK Analysis Console ===")
        try:
            self._collect_sources()
            self._collect_template()
            self._collect_outlier_settings()
            self._collect_chart_preferences()
            self._collect_output_path()
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled; exiting without running analysis.")
            return

        config = AnalysisInputs(
            sources=[SourceFile(path=path) for path in self.state.sources],
            output=self.state.output_path,
            template=self.state.template_root,
            outliers=OutlierOptions(method=self.state.outlier_method, k=self.state.outlier_k),
            generate_histogram=self.state.include_histogram,
            generate_cdf=self.state.include_cdf,
            generate_time_series=self.state.include_time_series,
        )

        result = run_analysis(config)
        print("\nAnalysis complete.")
        print(f"Workbook: {result['output']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Summary rows: {result['summary_rows']} | Measurements: {result['measurement_rows']}")

    def shutdown(self) -> None:
        """Placeholder to mirror expected GUI lifecycle."""
        print("Shutting down CPK console.")

    def _collect_sources(self) -> None:
        print("Enter STDF file paths (blank line to finish):")
        sources: list[Path] = []
        while True:
            entry = input("> ").strip()
            if not entry:
                break
            path = Path(entry).expanduser().resolve()
            if path.is_dir():
                discovered = sorted(p for p in path.glob("*.stdf") if p.is_file())
                if not discovered:
                    print(f"  ! No .stdf files found in directory: {path}")
                    continue
                print(f"  > Added {len(discovered)} file(s) from {path}")
                sources.extend(discovered)
                continue
            if not path.exists() or not path.is_file():
                print(f"  ! File not found: {path}")
                continue
            sources.append(path)
        if not sources:
            raise SystemExit("At least one STDF file is required.")
        self.state.sources = sources

    def _collect_template(self) -> None:
        entry = input("Template directory (optional): ").strip()
        if entry:
            root = Path(entry).expanduser().resolve()
            if not root.exists():
                raise SystemExit(f"Template path not found: {root}")
            if root.is_dir():
                self.state.template_root = _select_template(root)
            else:
                self.state.template_root = root

    def _collect_outlier_settings(self) -> None:
        method = input("Outlier method [none/iqr/stdev] (default: none): ").strip().lower()
        if method in {"iqr", "stdev", "none"}:
            self.state.outlier_method = method or "none"
        else:
            self.state.outlier_method = "none"
        if self.state.outlier_method != "none":
            try:
                k_value = float(input("Outlier multiplier k (default 1.5): ").strip() or "1.5")
            except ValueError:
                k_value = 1.5
            self.state.outlier_k = k_value

    def _collect_chart_preferences(self) -> None:
        self.state.include_histogram = _yes_no("Generate histograms? [Y/n]: ", default=True)
        self.state.include_cdf = _yes_no("Generate CDF charts? [Y/n]: ", default=True)
        self.state.include_time_series = _yes_no("Generate time-series charts? [Y/n]: ", default=True)

    def _collect_output_path(self) -> None:
        entry = input(f"Output workbook path (default: {self.state.output_path}): ").strip()
        if entry:
            self.state.output_path = Path(entry).expanduser().resolve()


def _yes_no(prompt: str, *, default: bool) -> bool:
    value = input(prompt).strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def main() -> int:
    """Entry point for launching the console workflow."""
    CPKAnalysisGUI().launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
