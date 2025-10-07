from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from .models import AnalysisInputs, OutlierOptions, SourceFile
from .pipeline import run_analysis
from .move_to_template import run as run_move_to_template


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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        sources = _scan_directory(args.directory, recursive=args.recursive)
        _write_metadata(args.metadata, sources)
        print(f"Recorded {len(sources)} STDF file(s) to {args.metadata}")
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

        config = AnalysisInputs(
            sources=sources,
            output=args.output,
            template=template_path,
            outliers=OutlierOptions(method=args.outlier_method, k=args.outlier_k),
            generate_histogram=not args.no_histogram,
            generate_cdf=not args.no_cdf,
            generate_time_series=not args.no_time_series,
        )
        result = run_analysis(config)
        print(f"Workbook written to {result['output']} ({result['summary_rows']} summary rows; "
              f"{result['measurement_rows']} measurements; outliers removed: {result['outlier_removed']})")
        print(f"Metadata captured at {result['metadata']}")
        return 0

    if args.command == "move-template":
        target_sheet = run_move_to_template(
            workbook_path=args.workbook,
            sheet_name=args.sheet,
        )
        print(f"Copied CPK Report data into sheet '{target_sheet}' in {args.workbook.resolve()}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


def _scan_directory(directory: Path, *, recursive: bool) -> list[Path]:
    directory = directory.expanduser().resolve()
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
    payload = [{"path": str(Path(src).expanduser().resolve())} for src in sources]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_metadata(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Metadata file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Path(entry["path"]).expanduser().resolve() for entry in data]


def _select_template(template_root: Path) -> Path:
    candidates = sorted(template_root.glob("*.xlsx"))
    if not candidates:
        raise SystemExit(f"No .xlsx template found under {template_root}")
    return candidates[0]


if __name__ == "__main__":
    raise SystemExit(main())
