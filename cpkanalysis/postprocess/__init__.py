"""Shared entry points for post-pipeline capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ..models import AnalysisInputs, OutlierOptions, PluginConfig, SourceFile
from . import menu
from .context import PostProcessContext, load_metadata
from .io_adapters import CliIO, GuiIO, PostProcessIO

__all__ = [
    "create_context",
    "open_cli_menu",
    "open_gui_menu",
]


def create_context(
    *,
    workbook_path: Path,
    metadata_path: Optional[Path] = None,
    analysis_inputs: Optional[AnalysisInputs] = None,
    metadata_strict: bool = True,
) -> PostProcessContext:
    """Build a :class:`PostProcessContext` from the supplied artefacts."""
    workbook_path = workbook_path.expanduser().resolve()
    if workbook_path.exists() and workbook_path.is_dir():
        print(f"Warning: Expected workbook file but received directory: {workbook_path}")
        raise SystemExit(f"Workbook path must be a file: {workbook_path}")
    metadata_path = (
        metadata_path.expanduser().resolve() if metadata_path is not None else workbook_path.with_suffix(".json")
    )
    if metadata_path.exists() and metadata_path.is_dir():
        print(f"Warning: Expected metadata file but received directory: {metadata_path}")
        raise SystemExit(f"Metadata path must be a file: {metadata_path}")
    metadata = load_metadata(metadata_path, strict=metadata_strict)

    if analysis_inputs is None:
        analysis_inputs = _analysis_inputs_from_metadata(workbook_path, metadata)

    return PostProcessContext(
        analysis_inputs=analysis_inputs,
        workbook_path=workbook_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def open_cli_menu(context: PostProcessContext, *, scripted_choices: Optional[Sequence[str]] = None) -> None:
    """Launch the post-processing menu using a CLI IO adapter."""
    io = CliIO(scripted_choices=scripted_choices)
    _open_menu(context, io)


def open_gui_menu(context: PostProcessContext, *, input_fn=None, output_fn=None) -> None:
    """Launch the post-processing menu using a GUI IO adapter (console based)."""
    io = GuiIO(input_fn=input_fn, output_fn=output_fn)
    _open_menu(context, io)


def _open_menu(context: PostProcessContext, io: PostProcessIO) -> None:
    menu.loop(context, io=io)


def _analysis_inputs_from_metadata(workbook_path: Path, metadata: dict) -> AnalysisInputs:
    """Rehydrate a minimal ``AnalysisInputs`` record from metadata."""
    output_path = workbook_path
    template_path = None

    template_value = metadata.get("template")
    if template_value:
        try:
            template_path = Path(template_value).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            template_path = None

    sources = []
    for entry in metadata.get("sources", []):
        path_text = entry.get("path") or entry.get("file") or entry.get("source")
        if not path_text:
            continue
        try:
            sources.append(SourceFile(path=Path(path_text)))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue

    plugin_configs = [
        PluginConfig(
            plugin_id=record.get("id", ""),
            enabled=True,
            priority=record.get("priority"),
            parameters=record.get("parameters", {}),
        )
        for record in metadata.get("plugins", [])
        if record.get("id")
    ]

    options = metadata.get("analysis_options", {})
    def _bool_option(name: str, default: bool) -> bool:
        value = options.get(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    display_decimals = options.get("display_decimals", 4)

    return AnalysisInputs(
        sources=sources,
        output=output_path,
        template=template_path,
        template_sheet=metadata.get("template_sheet"),
        outliers=OutlierOptions(),
        generate_histogram=_bool_option("generate_histogram", True),
        generate_cdf=_bool_option("generate_cdf", True),
        generate_time_series=_bool_option("generate_time_series", True),
        generate_yield_pareto=_bool_option("generate_yield_pareto", False),
        pareto_first_failure_only=_bool_option("pareto_first_failure_only", False),
        display_decimals=display_decimals,
        plugins=plugin_configs,
        enable_site_breakdown=_bool_option("enable_site_breakdown", False),
        include_site_rows_in_cpk=_bool_option("include_site_rows_in_cpk", False),
    )
