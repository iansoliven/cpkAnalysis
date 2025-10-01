from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

from .models import MeasurementRow, SourceFile, SummaryRow
from .stages import calculate_shift, convert_to_data, generate_plot, read_directory

DEFAULT_METADATA_PATH = Path("shift_sources.json")
DEFAULT_OUTPUT_PATH = Path("DataWorkBook.xlsx")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shiftanalysis",
        description="Shift analysis workflow orchestrator.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=f"Path to metadata JSON file (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output workbook path (default: {DEFAULT_OUTPUT_PATH}).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    readdir_parser = subparsers.add_parser(
        "readdir",
        help="Scan a directory for candidate source files and confirm metadata.",
    )
    readdir_parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Directory to scan (defaults to current working directory).",
    )
    readdir_parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Skip confirmation prompt (non-interactive).",
    )

    convert_parser = subparsers.add_parser(
        "convert",
        help="Populate the DataWorkBook summary and measurement sheets.",
    )
    convert_parser.add_argument(
        "--values-only",
        action="store_true",
        help="Skip formatting while processing Excel files.",
    )

    calc_parser = subparsers.add_parser(
        "calc-shift",
        help="Append shift calculations to the workbook.",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="Generate boxplot and histogram sheets.",
    )
    plot_parser.add_argument(
        "--BoxChartOnly",
        action="store_true",
        help="Generate only boxplot sheets.",
    )
    plot_parser.add_argument(
        "--HistoChartOnly",
        action="store_true",
        help="Generate only histogram sheets.",
    )

    auto_parser = subparsers.add_parser(
        "auto",
        help="Run the full pipeline sequentially.",
    )
    auto_parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Optional directory to scan before processing.",
    )
    auto_parser.add_argument(
        "--assume-yes",
        action="store_true",
        help="Skip confirmation prompt during the directory scan.",
    )
    auto_parser.add_argument(
        "--values-only",
        action="store_true",
        help="Skip Excel formatting when converting to data.",
    )
    auto_parser.add_argument(
        "--BoxChartOnly",
        action="store_true",
        help="Generate only boxplot sheets during the plotting stage.",
    )
    auto_parser.add_argument(
        "--HistoChartOnly",
        action="store_true",
        help="Generate only histogram sheets during the plotting stage.",
    )

    return parser


def _load_metadata(path: Path) -> list[SourceFile]:
    if not path.exists():
        raise SystemExit(f"Metadata file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        SourceFile(
            path=Path(entry["path"]),
            lot=entry["lot"],
            event=entry["event"],
            interval=entry["interval"],
            file_type=entry["file_type"],
        )
        for entry in data
    ]


def _save_metadata(path: Path, sources: Sequence[SourceFile]) -> None:
    payload = [
        {
            "path": str(source.path),
            "lot": source.lot,
            "event": source.event,
            "interval": source.interval,
            "file_type": source.file_type,
        }
        for source in sources
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_readdir(args: argparse.Namespace) -> list[SourceFile]:
    sources = read_directory.run(
        directory=args.directory,
        metadata_out=args.metadata,
        assume_yes=args.assume_yes,
    )
    return sources


def run_convert(args: argparse.Namespace, sources: Sequence[SourceFile]) -> tuple[list[SummaryRow], list[MeasurementRow]]:
    summaries, measurements = convert_to_data.run(
        sources=sources,
        output_path=args.output,
        values_only=getattr(args, "values_only", False),
    )
    return summaries, measurements


def run_calc_shift(args: argparse.Namespace, measurements: Sequence[MeasurementRow]) -> None:
    calculate_shift.run(
        measurements=measurements,
        workbook_path=args.output,
    )


def run_plot(args: argparse.Namespace, measurements: Sequence[MeasurementRow]) -> None:
    generate_plot.run(
        measurements=measurements,
        workbook_path=args.output,
        box_only=getattr(args, "BoxChartOnly", False),
        hist_only=getattr(args, "HistoChartOnly", False),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "readdir":
        sources = run_readdir(args)
        print(f"Discovered {len(sources)} source files; metadata saved to {args.metadata}.")
        return 0

    if args.command == "convert":
        sources = _load_metadata(args.metadata)
        summaries, measurements = run_convert(args, sources)
        print(f"Wrote DataWorkBook with {len(summaries)} summary rows and {len(measurements)} measurements.")
        return 0

    if args.command == "calc-shift":
        sources = _load_metadata(args.metadata)
        _, measurements = convert_to_data.run(sources, args.output)
        run_calc_shift(args, measurements)
        print("Shift calculations appended.")
        return 0

    if args.command == "plot":
        sources = _load_metadata(args.metadata)
        _, measurements = convert_to_data.run(sources, args.output)
        run_plot(args, measurements)
        print("Plot sheets generated.")
        return 0

    if args.command == "auto":
        sources = run_readdir(argparse.Namespace(
            directory=args.directory,
            assume_yes=args.assume_yes,
            metadata=args.metadata,
        ))
        summaries, measurements = run_convert(args, sources)
        run_calc_shift(args, measurements)
        run_plot(args, measurements)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
