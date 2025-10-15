"""Minimal text-driven GUI scaffold for the CPK workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .cli import _select_template, _prepare_output_path
from .models import AnalysisInputs, OutlierOptions, SourceFile, PluginConfig
from .pipeline import run_analysis
from .plugins import PluginRegistry, PluginRegistryError
from .plugin_profiles import load_plugin_profile, save_plugin_profile, PROFILE_FILENAME
from . import postprocess

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .postprocess.context import PostProcessContext


@dataclass
class ApplicationState:
    """In-memory representation of the current GUI configuration."""

    workspace_path: Path = Path.cwd()
    sources: list[Path] = field(default_factory=list)
    template_root: Path | None = None
    template_sheet: str | None = None
    output_path: Path = Path("CPK_Workbook.xlsx")
    outlier_method: str = "none"
    outlier_k: float = 1.5
    include_histogram: bool = True
    include_cdf: bool = True
    include_time_series: bool = True
    generate_yield_pareto: bool = False
    display_decimals: int = 4
    plugins: list[PluginConfig] = field(default_factory=list)


class CPKAnalysisGUI:
    """Very small text-based facade mirroring the future GUI flow."""

    def __init__(self, state: ApplicationState | None = None) -> None:
        self.state = state or ApplicationState()
        self.state.workspace_path = self.state.workspace_path.expanduser().resolve()
        self.registry = PluginRegistry(workspace_dir=self.state.workspace_path / "cpk_plugins")
        self._post_context: Optional["PostProcessContext"] = None

    def launch(self) -> None:
        """Launch the interactive console session."""
        print("=== CPK Analysis Console ===")
        try:
            self._collect_sources()
            self._collect_template()
            self._collect_outlier_settings()
            self._collect_chart_preferences()
            self._collect_format_preferences()
            self._collect_plugins()
            self._collect_output_path()
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled; exiting without running analysis.")
            return

        config = AnalysisInputs(
            sources=[SourceFile(path=path) for path in self.state.sources],
            output=self.state.output_path,
            template=self.state.template_root,
            template_sheet=self.state.template_sheet,
            outliers=OutlierOptions(method=self.state.outlier_method, k=self.state.outlier_k),
            generate_histogram=self.state.include_histogram,
            generate_cdf=self.state.include_cdf,
            generate_time_series=self.state.include_time_series,
            generate_yield_pareto=self.state.generate_yield_pareto,
            display_decimals=self.state.display_decimals,
            plugins=self.state.plugins,
        )

        result = run_analysis(config, registry=self.registry)
        print("\nAnalysis complete.")
        print(f"Workbook: {result['output']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Summary rows: {result['summary_rows']} | Measurements: {result['measurement_rows']}")
        if result.get("plugins"):
            print(f"Post-processing plugins: {', '.join(result['plugins'])}")

        try:
            metadata_path = Path(result["metadata"]).expanduser().resolve() if result.get("metadata") else None
        except Exception:
            metadata_path = None
        try:
            self._post_context = postprocess.create_context(
                workbook_path=Path(result["output"]).expanduser().resolve(),
                metadata_path=metadata_path,
                analysis_inputs=config,
            )
        except Exception as exc:
            print(f"Unable to initialise post-processing context: {exc}")
            self._post_context = None

        if self._post_context is not None:
            if _yes_no("Open post-processing menu now? [Y/n]: ", default=True):
                self._open_postprocess_menu()
            self._postprocess_command_prompt()

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
            sheet = input("Template sheet (optional): ").strip()
            self.state.template_sheet = sheet

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
        self.state.generate_yield_pareto = _yes_no("Generate Yield & Pareto analysis? [y/N]: ", default=False)

    def _collect_format_preferences(self) -> None:
        prompt = f"Fallback decimal places when STDF hints are missing (0-9, default {self.state.display_decimals}): "
        entry = input(prompt).strip()
        if not entry:
            return
        try:
            value = int(entry)
        except ValueError:
            print("  ! Invalid integer; keeping existing setting.")
            return
        if value < 0:
            value = 0
        if value > 9:
            value = 9
        self.state.display_decimals = value

    def _collect_plugins(self) -> None:
        try:
            descriptors = self.registry.descriptors()
        except PluginRegistryError as exc:
            print(f"\nPost-processing plugins unavailable: {exc}")
            self.state.plugins = []
            return

        if not descriptors:
            print("\nPost-processing plugins: none available.")
            self.state.plugins = []
            save_plugin_profile(self._profile_path(), [])
            return

        sorted_descriptors = sorted(descriptors.values(), key=lambda d: d.name.lower())
        profile_path = self._profile_path()
        stored = load_plugin_profile(profile_path)

        print("\nPost-processing plugins:")
        configs: list[PluginConfig] = []

        for descriptor in sorted_descriptors:
            print(f"\nPlugin: {descriptor.name} ({descriptor.plugin_id})")
            if descriptor.description:
                print(f"  {descriptor.description}")
            print(f"  Events: {', '.join(descriptor.events or ('PipelineEvent',))}")
            stored_cfg = stored.get(descriptor.plugin_id)
            enabled_default = stored_cfg.enabled if stored_cfg else descriptor.default_enabled
            enabled = _yes_no(
                f"  Enable? [{'Y/n' if enabled_default else 'y/N'}]: ",
                default=enabled_default,
            )

            priority_default = (
                stored_cfg.priority
                if stored_cfg and stored_cfg.priority is not None
                else descriptor.default_priority
            )
            priority_value: Optional[int] = priority_default
            parameters: Dict[str, Any] = dict(stored_cfg.parameters) if stored_cfg else {}

            if enabled:
                priority_input = input(f"  Priority (default {priority_default}): ").strip()
                if priority_input:
                    try:
                        priority_value = int(priority_input)
                    except ValueError:
                        print("  ! Invalid integer; keeping default priority.")
                parameters = self._prompt_parameters(parameters)

            configs.append(
                PluginConfig(
                    plugin_id=descriptor.plugin_id,
                    enabled=enabled,
                    priority=priority_value,
                    parameters=parameters,
                )
            )

        self.state.plugins = configs
        save_plugin_profile(profile_path, configs)

    def _collect_output_path(self) -> None:
        entry = input(f"Output workbook path (default: {self.state.output_path}): ").strip()
        if entry:
            self.state.output_path = _prepare_output_path(Path(entry))

    def _open_postprocess_menu(self) -> None:
        if self._post_context is None:
            print("Post-processing context unavailable.")
            return
        postprocess.open_gui_menu(self._post_context, input_fn=input, output_fn=print)

    def _postprocess_command_prompt(self) -> None:
        if self._post_context is None:
            return
        print("\nEnter 'post' to reopen the post-processing menu or 'quit' to exit this session.")
        while True:
            command = input("post> ").strip().lower()
            if command in {"", "quit", "exit"}:
                break
            if command in {"post", "p"}:
                self._open_postprocess_menu()
                continue
            if command in {"help", "?"}:
                print("Commands: 'post' to open the menu, 'quit' to exit this prompt.")
                continue
            print("Unknown command. Type 'help' for options.")

    def _profile_path(self) -> Path:
        return self.state.workspace_path / PROFILE_FILENAME

    def _prompt_parameters(self, existing: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(existing)
        if params:
            print("  Current parameters:")
            for key, value in params.items():
                print(f"    {key} = {value}")
        else:
            print("  No parameters set.")
        print("  Enter key=value pairs to update parameters (blank to continue).")
        print("  Use key= (empty value) to remove an existing parameter.")
        while True:
            entry = input("    parameter> ").strip()
            if not entry:
                return params
            if "=" not in entry:
                print("    ! Expected key=value format.")
                continue
            key, value = entry.split("=", 1)
            key = key.strip()
            if not key:
                print("    ! Key cannot be empty.")
                continue
            value = value.strip()
            if value:
                params[key] = value
            else:
                params.pop(key, None)


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
