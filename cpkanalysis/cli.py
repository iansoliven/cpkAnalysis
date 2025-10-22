from __future__ import annotations

import argparse
import json
import time
import shutil
import os
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from .models import AnalysisInputs, OutlierOptions, SourceFile, PluginConfig
from .pipeline import run_analysis
from .move_to_template import run as run_move_to_template
from .plugins import PluginRegistry, PluginRegistryError, PluginDescriptor
from .plugin_profiles import load_plugin_profile, PROFILE_FILENAME
from . import ingest, postprocess


def _warn_path_type_mismatch(path: Path, *, expect_file: bool, description: str) -> None:
    if not path.exists():
        return
    if expect_file and path.is_dir():
        print(f"Warning: {description} should be a file, but a directory was provided: {path}")
    if not expect_file and path.is_file():
        print(f"Warning: {description} should be a directory, but a file was provided: {path}")


def _prepare_output_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists() and path.is_dir():
        print(f"Warning: Output path {path} is a directory; writing CPK_Workbook.xlsx inside it.")
        path = path / "CPK_Workbook.xlsx"
    if path.suffix:
        if path.suffix.lower() != ".xlsx":
            print(f"Warning: Output workbook should use '.xlsx'; replacing extension for {path}.")
            path = path.with_suffix(".xlsx")
    else:
        path = path.with_suffix(".xlsx")
    return path.resolve()


def _resolve_render_process_limit(cli_value: Optional[int]) -> Optional[int]:
    if cli_value is not None:
        return cli_value if cli_value > 0 else None
    env_value = os.environ.get("CPKANALYSIS_MAX_RENDER_PROCS")
    if env_value:
        try:
            value = int(env_value)
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpkanalysis",
        description="CPK analysis workflow for STDF sources.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Discover STDF files in a directory and write metadata JSON.")
    scan_parser.add_argument("directory", nargs="?", type=Path, default=Path.cwd())
    scan_parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("stdf_sources.json"),
        help="Destination metadata JSON path (default: stdf_sources.json).",
    )
    scan_parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for STDF files.",
    )

    run_parser = subparsers.add_parser("run", help="Execute the full CPK workflow.")
    run_parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="STDF file paths to process.",
    )
    run_parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional metadata JSON listing STDF files to process.",
    )
    run_parser.add_argument(
        "--output",
        type=Path,
        default=Path("CPK_Workbook.xlsx"),
        help="Output Excel workbook path.",
    )
    run_parser.add_argument(
        "--template",
        type=Path,
        default=Path("cpkTemplate"),
        help="Path to the Excel template directory or file (default: ./cpkTemplate).",
    )
    run_parser.add_argument(
        "--template-sheet",
        type=str,
        help="Optional template sheet name to receive CPK data.",
    )
    run_parser.add_argument(
        "--outlier-method",
        choices=("none", "iqr", "stdev"),
        default="none",
        help="Outlier filtering strategy.",
    )
    run_parser.add_argument(
        "--outlier-k",
        type=float,
        default=1.5,
        help="Multiplier for outlier filtering heuristics.",
    )
    run_parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Skip histogram chart generation.",
    )
    run_parser.add_argument(
        "--no-cdf",
        action="store_true",
        help="Skip CDF chart generation.",
    )
    run_parser.add_argument(
        "--no-time-series",
        action="store_true",
        help="Skip time-series chart generation.",
    )
    run_parser.add_argument(
        "--max-render-procs",
        type=int,
        help="Maximum parallel processes for chart rendering (default leaves 2 cores free).",
    )
    run_parser.add_argument(
        "--histogram-rug",
        action="store_true",
        help="Add rug markers beneath histograms (may increase processing time).",
    )
    run_parser.add_argument(
        "--generate-yield-pareto",
        action="store_true",
        help="Generate the Yield and Pareto analysis sheet with associated charts.",
    )
    run_parser.add_argument(
        "--site-breakdown",
        action="store_true",
        help="Generate per-site aggregates when SITE_NUM data is available.",
    )
    run_parser.add_argument(
        "--no-site-breakdown",
        action="store_true",
        help="Skip per-site aggregation even if SITE_NUM data is available.",
    )
    run_parser.add_argument(
        "--display-decimals",
        type=int,
        help="Override fallback decimal places when STDF formats are absent (0-9, default: 4).",
    )
    run_parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help=(
            "Plugin override directive. "
            "Formats: enable:<id>, disable:<id>, priority:<id>:value, param:<id>:key=value. "
            "May be specified multiple times."
        ),
    )
    run_parser.add_argument(
        "--plugin-profile",
        type=Path,
        help=f"Optional plugin profile file (default: ./{PROFILE_FILENAME}).",
    )
    run_parser.add_argument(
        "--validate-plugins",
        action="store_true",
        help="Run plugins against newly generated data without persisting the workbook.",
    )
    run_parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Launch the post-processing menu after the workbook is generated.",
    )
    run_parser.add_argument(
        "--keep-session",
        action="store_true",
        help="Retain the temporary session directory for inspection after the run completes.",
    )
    run_parser.add_argument(
        "--cpk-include-site-rows",
        action="store_true",
        help="Include per-site rows in the CPK Report when site breakdown data is available.",
    )
    run_parser.add_argument(
        "--no-cpk-include-site-rows",
        action="store_true",
        help="Omit per-site rows from the CPK Report (default behavior).",
    )

    prune_parser = subparsers.add_parser(
        "prune-sessions",
        help="Delete temporary session directories created by previous runs.",
    )
    prune_parser.add_argument(
        "--root",
        type=Path,
        default=Path("temp"),
        help="Root directory containing session folders (default: ./temp).",
    )
    prune_parser.add_argument(
        "--older-than",
        type=float,
        help="Prune sessions older than the specified number of hours (default: remove all sessions).",
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sessions that would be removed without deleting them.",
    )

    move_template_parser = subparsers.add_parser(
        "move-template",
        help="Copy CPK Report contents into the template sheet.",
    )
    move_template_parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("CPK_Workbook.xlsx"),
        help="Workbook path whose CPK Report should be copied (default: CPK_Workbook.xlsx).",
    )
    move_template_parser.add_argument(
        "--sheet",
        type=str,
        help="Target sheet within the workbook; defaults to the first non-CPK Report sheet.",
    )

    post_parser = subparsers.add_parser(
        "post-process",
        help="Open the post-processing menu for an existing workbook.",
    )
    post_parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("CPK_Workbook.xlsx"),
        help="Workbook path to process (default: CPK_Workbook.xlsx).",
    )
    post_parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional metadata JSON path (defaults to <workbook>.json).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        sources = _scan_directory(args.directory, recursive=args.recursive)
        _write_metadata(args.metadata, sources)
        print(f"Recorded {len(sources)} STDF file(s) to {args.metadata}")
        return 0

    if args.command == "post-process":
        workbook_path = args.workbook.expanduser().resolve()
        if workbook_path.exists() and workbook_path.is_dir():
            _warn_path_type_mismatch(workbook_path, expect_file=True, description="Workbook path")
            raise SystemExit(f"Workbook path must be a file: {workbook_path}")
        metadata_path = args.metadata
        if metadata_path is not None:
            metadata_path = metadata_path.expanduser().resolve()
            if metadata_path.exists() and metadata_path.is_dir():
                _warn_path_type_mismatch(metadata_path, expect_file=True, description="Metadata path")
                raise SystemExit(f"Metadata path must be a file: {metadata_path}")
        context = postprocess.create_context(
            workbook_path=workbook_path,
            metadata_path=metadata_path,
        )
        postprocess.open_cli_menu(context)
        return 0

    if args.command == "prune-sessions":
        from .session_prune import prune_sessions

        root = args.root.expanduser().resolve()
        if not root.exists():
            print(f"No session directory found at {root}")
            return 0
        removed = prune_sessions(
            root,
            older_than_hours=args.older_than,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            if removed:
                print("Sessions eligible for removal:")
                for path in removed:
                    print(f"  {path}")
            else:
                print("No session directories meet the removal criteria.")
        else:
            if removed:
                print("Removed session directories:")
                for path in removed:
                    print(f"  {path}")
            else:
                print("No session directories were removed.")
        return 0

    if args.command == "run":
        paths: list[Path] = []
        if args.metadata:
            paths.extend(_read_metadata(args.metadata))
        paths.extend(args.inputs or [])
        if not paths:
            discovered = _scan_directory(Path.cwd(), recursive=True)
            if not discovered:
                parser.error("No STDF inputs supplied and no STDF files found in the current directory.")
            print(f"Auto-discovered {len(discovered)} STDF file(s) under {Path.cwd()}.")
            paths.extend(discovered)

        unique_paths = list(dict.fromkeys(Path(path).expanduser().resolve() for path in paths))
        sources = [SourceFile(path=path) for path in unique_paths]

        template_path: Path | None = args.template
        if template_path is not None:
            template_path = template_path.expanduser().resolve()
            if template_path.is_dir():
                template_path = _select_template(template_path)
            elif not template_path.exists():
                raise SystemExit(f"Template path not found: {template_path}")

        workspace_root = Path.cwd()
        if args.plugin_profile:
            profile_path = args.plugin_profile.expanduser().resolve()
            if profile_path.exists() and profile_path.is_dir():
                _warn_path_type_mismatch(profile_path, expect_file=True, description="Plugin profile")
                profile_path = profile_path / PROFILE_FILENAME
        else:
            profile_path = workspace_root / PROFILE_FILENAME
        if args.plugin_profile and not profile_path.exists():
            print(f"Warning: plugin profile {profile_path} not found; using defaults.")

        registry = PluginRegistry(workspace_dir=workspace_root / "cpk_plugins")
        plugins, override_conflicts, missing_plugins = _prepare_plugin_configs(
            registry=registry,
            profile_path=profile_path,
            overrides=args.plugin,
            parser=parser,
        )
        for plugin_id in missing_plugins:
            print(f"Warning: plugin '{plugin_id}' referenced in profile is unavailable and will be ignored.")
        for plugin_id in override_conflicts:
            print(f"Warning: command-line overrides applied to plugin '{plugin_id}' (profile preference superseded).")

        output_path: Path = _prepare_output_path(args.output)
        temp_template_path: Optional[Path] = None
        if args.validate_plugins:
            temp_dir = workspace_root / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time() * 1000)
            output_path = temp_dir / f"validation_{timestamp}.xlsx"
            if template_path is not None and template_path.exists():
                temp_template_path = temp_dir / f"template_{timestamp}.xlsx"
                shutil.copy2(template_path, temp_template_path)
                template_path = temp_template_path

        if args.histogram_rug and args.no_histogram:
            print("Warning: --histogram-rug ignored because histograms are disabled.")
            histogram_rug = False
        else:
            histogram_rug = bool(args.histogram_rug and not args.no_histogram)
            if histogram_rug:
                print("Warning: Rug plots may significantly increase processing time on large datasets.")

        if args.site_breakdown and args.no_site_breakdown:
            parser.error("--site-breakdown and --no-site-breakdown cannot be used together.")

        site_status = "unknown"
        enable_site_breakdown = False
        site_available = False
        site_message: Optional[str] = None
        try:
            site_available, site_message = ingest.detect_site_support(sources)
        except Exception as exc:  # pragma: no cover - defensive guard
            site_available = False
            site_message = f"Site detection failed: {exc}"
        site_status = "available" if site_available else "unavailable"
        if site_message:
            print(f"Site detection: {site_message}")

        if site_available:
            if args.no_site_breakdown:
                enable_site_breakdown = False
            elif args.site_breakdown:
                enable_site_breakdown = True
            else:
                enable_site_breakdown = _prompt_yes_no(
                    "SITE_NUM detected. Generate per-site aggregates? [y/N]: ",
                    default=False,
                )
        else:
            if args.site_breakdown:
                proceed = _prompt_yes_no(
                    "SITE_NUM data unavailable; proceed without per-site aggregation? [y/N]: ",
                    default=False,
                )
                if not proceed:
                    raise SystemExit("Aborted: per-site aggregation requested but SITE_NUM data is unavailable.")
            enable_site_breakdown = False

        if args.cpk_include_site_rows and args.no_cpk_include_site_rows:
            parser.error("--cpk-include-site-rows and --no-cpk-include-site-rows cannot be used together.")

        include_site_rows = False
        if args.cpk_include_site_rows:
            if enable_site_breakdown:
                include_site_rows = True
            else:
                print("Warning: --cpk-include-site-rows ignored because per-site aggregation is disabled.")
        elif args.no_cpk_include_site_rows:
            include_site_rows = False

        config = AnalysisInputs(
            sources=sources,
            output=output_path,
            template=template_path,
            template_sheet=args.template_sheet,
            outliers=OutlierOptions(method=args.outlier_method, k=args.outlier_k),
            generate_histogram=not args.no_histogram,
            generate_cdf=not args.no_cdf,
            generate_time_series=not args.no_time_series,
            generate_yield_pareto=args.generate_yield_pareto,
            display_decimals=args.display_decimals if args.display_decimals is not None else 4,
            plugins=plugins,
            max_render_processes=_resolve_render_process_limit(args.max_render_procs),
            histogram_rug=histogram_rug,
            enable_site_breakdown=enable_site_breakdown,
            site_data_status=site_status,
            keep_session=bool(args.keep_session),
            include_site_rows_in_cpk=include_site_rows,
        )
        result = run_analysis(config, registry=registry)
        if args.validate_plugins:
            print(f"Plugin validation completed using temporary workbook {result['output']}.")
            _cleanup_file(result.get("metadata", ""))
            _cleanup_file(result.get("output", ""))
            if temp_template_path is not None:
                _cleanup_file(str(temp_template_path))
        else:
            warnings = result.get("warnings") or []
            for warning in warnings:
                print(f"Warning: {warning}")
            print(
                f"Workbook written to {result['output']} ({result['summary_rows']} summary rows; "
                f"{result['measurement_rows']} measurements; outliers removed: {result['outlier_removed']})"
            )
            print(f"Metadata captured at {result['metadata']}")
            if result.get('template_sheet'):
                print(f"Template sheet updated: {result['template_sheet']}")
            elapsed = result.get("elapsed_seconds")
            if isinstance(elapsed, (int, float)):
                print(f"Total elapsed time: {elapsed:.2f}s")
            stage_details = result.get("stage_details") or {}
            workbook_details = stage_details.get("workbook") if isinstance(stage_details, dict) else None
            if isinstance(workbook_details, dict) and workbook_details:
                print("Workbook timing breakdown:")
                if "charts.render" in workbook_details:
                    print(f"  Chart rendering: {workbook_details['charts.render']:.2f}s")
                if "charts.embed" in workbook_details:
                    print(f"  Chart embedding: {workbook_details['charts.embed']:.2f}s")
                if "charts.total" in workbook_details:
                    print(f"  Chart stage total: {workbook_details['charts.total']:.2f}s")
                if "workbook.save" in workbook_details:
                    print(f"  Workbook save: {workbook_details['workbook.save']:.2f}s")
        if result.get("plugins"):
            print(f"Post-processing plugins: {', '.join(result['plugins'])}")
        if args.postprocess:
            metadata_path = Path(result["metadata"]).expanduser().resolve() if result.get("metadata") else None
            context_obj = postprocess.create_context(
                workbook_path=Path(result["output"]).expanduser().resolve(),
                metadata_path=metadata_path,
                analysis_inputs=config,
            )
            postprocess.open_cli_menu(context_obj)
        return 0

    if args.command == "move-template":
        workbook_path = args.workbook.expanduser().resolve()
        if workbook_path.exists() and workbook_path.is_dir():
            _warn_path_type_mismatch(workbook_path, expect_file=True, description="Workbook path")
            raise SystemExit(f"Workbook path must be a file: {workbook_path}")
        target_sheet = run_move_to_template(
            workbook_path=workbook_path,
            sheet_name=args.sheet,
        )
        print(f"Copied CPK Report data into sheet '{target_sheet}' in {workbook_path}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


def _prompt_yes_no(prompt: str, *, default: bool) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        return default
    try:
        response = input(prompt)
    except EOFError:
        return default
    if response is None:
        return default
    response = response.strip().lower()
    if not response:
        return default
    return response in {"y", "yes"}

def _scan_directory(directory: Path, *, recursive: bool) -> list[Path]:
    directory = directory.expanduser().resolve()
    if directory.exists() and directory.is_file():
        _warn_path_type_mismatch(directory, expect_file=False, description="Scan directory")
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")
    if recursive:
        results: list[Path] = []
        for path in directory.rglob("*.stdf"):
            if any(part.lower() == "submodules" for part in path.parts):
                continue
            if path.is_file():
                results.append(path.resolve())
        return sorted(results)
    return sorted(path.resolve() for path in directory.glob("*.stdf") if path.is_file())


def _write_metadata(path: Path, sources: Iterable[Path]) -> None:
    if path.exists() and path.is_dir():
        print(f"Warning: Metadata path {path} is a directory; writing stdf_sources.json inside it.")
        path = path / "stdf_sources.json"
    payload = [{"path": str(Path(src).expanduser().resolve())} for src in sources]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_metadata(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.exists() and path.is_dir():
        _warn_path_type_mismatch(path, expect_file=True, description="Metadata path")
        raise SystemExit(f"Metadata path is a directory, expected a file: {path}")
    if not path.exists():
        raise SystemExit(f"Metadata file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Path(entry["path"]).expanduser().resolve() for entry in data]


def _select_template(template_root: Path) -> Path:
    candidates = sorted(template_root.glob("*.xlsx"))
    if not candidates:
        raise SystemExit(f"No .xlsx template found under {template_root}")
    return candidates[0]


def _cleanup_file(path_str: str) -> None:
    if not path_str:
        return
    path = Path(path_str)
    try:
        if path.exists():
            path.unlink()
    except (PermissionError, OSError) as exc:
        logging.getLogger(__name__).warning("Failed to remove temporary file '%s': %s", path, exc)


def _prepare_plugin_configs(
    *,
    registry: PluginRegistry,
    profile_path: Path,
    overrides: Sequence[str],
    parser: argparse.ArgumentParser,
) -> tuple[List[PluginConfig], List[str], List[str]]:
    try:
        descriptors = registry.descriptors()
    except PluginRegistryError as exc:
        parser.error(f"Plugin discovery failed: {exc}")

    profile_map = load_plugin_profile(profile_path)
    plugin_configs: Dict[str, PluginConfig] = {}

    for plugin_id, descriptor in descriptors.items():
        base = profile_map.get(plugin_id)
        if base is not None:
            priority = base.priority if base.priority is not None else descriptor.default_priority
            plugin_configs[plugin_id] = PluginConfig(
                plugin_id=plugin_id,
                enabled=base.enabled,
                priority=priority,
                parameters=dict(base.parameters),
            )
        else:
            plugin_configs[plugin_id] = PluginConfig(
                plugin_id=plugin_id,
                enabled=descriptor.default_enabled,
                priority=descriptor.default_priority,
                parameters={},
            )

    conflicts: Set[str] = set()

    for token in overrides or []:
        kind, sep, remainder = token.partition(":")
        if not sep or not remainder:
            parser.error(f"Invalid --plugin directive '{token}'.")
        kind = kind.lower()
        current_id: Optional[str] = None

        if kind in {"enable", "disable"}:
            plugin_id = remainder.strip()
            descriptor = descriptors.get(plugin_id)
            if descriptor is None:
                parser.error(_unknown_plugin_message(plugin_id, descriptors))
            cfg = plugin_configs.get(plugin_id)
            if cfg is None:
                cfg = PluginConfig(
                    plugin_id=plugin_id,
                    enabled=descriptor.default_enabled,
                    priority=descriptor.default_priority,
                    parameters={},
                )
            plugin_configs[plugin_id] = PluginConfig(
                plugin_id=plugin_id,
                enabled=(kind == "enable"),
                priority=cfg.priority,
                parameters=dict(cfg.parameters),
            )
            current_id = plugin_id
        elif kind == "priority":
            plugin_id, sep2, value_text = remainder.partition(":")
            if not sep2 or not value_text:
                parser.error(f"Invalid priority directive '{token}'. Expected priority:<id>:value.")
            descriptor = descriptors.get(plugin_id)
            if descriptor is None:
                parser.error(_unknown_plugin_message(plugin_id, descriptors))
            cfg = plugin_configs.get(plugin_id)
            if cfg is None:
                cfg = PluginConfig(
                    plugin_id=plugin_id,
                    enabled=descriptor.default_enabled,
                    priority=descriptor.default_priority,
                    parameters={},
                )
            try:
                priority_value = int(value_text.strip())
            except ValueError:
                parser.error(f"Invalid priority value in '{token}'.")
            plugin_configs[plugin_id] = PluginConfig(
                plugin_id=plugin_id,
                enabled=cfg.enabled,
                priority=priority_value,
                parameters=dict(cfg.parameters),
            )
            current_id = plugin_id
        elif kind == "param":
            plugin_id, sep2, assignment = remainder.partition(":")
            if not sep2 or "=" not in assignment:
                parser.error(f"Invalid parameter directive '{token}'. Expected param:<id>:key=value.")
            key, value = assignment.split("=", 1)
            key = key.strip()
            if not key:
                parser.error("Plugin parameter key cannot be empty.")
            descriptor = descriptors.get(plugin_id)
            if descriptor is None:
                parser.error(_unknown_plugin_message(plugin_id, descriptors))
            cfg = plugin_configs.get(plugin_id)
            if cfg is None:
                cfg = PluginConfig(
                    plugin_id=plugin_id,
                    enabled=descriptor.default_enabled,
                    priority=descriptor.default_priority,
                    parameters={},
                )
            params = dict(cfg.parameters)
            value = value.strip()
            if value:
                params[key] = value
            else:
                params.pop(key, None)
            plugin_configs[plugin_id] = PluginConfig(
                plugin_id=plugin_id,
                enabled=cfg.enabled,
                priority=cfg.priority,
                parameters=params,
            )
            current_id = plugin_id
        else:
            parser.error(f"Unsupported --plugin directive '{token}'.")

        if current_id:
            base_cfg = profile_map.get(current_id)
            if base_cfg and plugin_configs[current_id] != base_cfg:
                conflicts.add(current_id)

    missing_from_registry = sorted(set(profile_map.keys()) - set(descriptors.keys()))
    ordered = [
        plugin_configs[plugin_id]
        for plugin_id in sorted(descriptors.keys(), key=lambda pid: descriptors[pid].name.lower())
    ]
    return ordered, sorted(conflicts), missing_from_registry


def _unknown_plugin_message(plugin_id: str, descriptors: Dict[str, PluginDescriptor]) -> str:
    available = ", ".join(sorted(descriptors.keys()))
    suffix = available if available else "none available"
    return f"Unknown plugin id '{plugin_id}'. Available plugins: {suffix}."


if __name__ == "__main__":
    raise SystemExit(main())
