from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..models import MeasurementRow


def run(
    measurements: Sequence[MeasurementRow],
    workbook_path: Path,
    *,
    box_only: bool = False,
    hist_only: bool = False,
) -> None:
    if not measurements:
        return

    if box_only and hist_only:
        box_only = False
        hist_only = False

    workbook_path = workbook_path.expanduser().resolve()
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found for plotting: {workbook_path}")

    try:
        from ..charts import boxplots, histograms  # pylint: disable=import-error
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency handling
        missing = exc.name or 'required plotting dependency'
        raise RuntimeError(
            f"Plot generation requires additional dependency '{missing}'. Install matplotlib and numpy to proceed."
        ) from exc

    if not hist_only:
        events = boxplots.load_measurement_tests(workbook_path)
        boxplots.add_boxplots_to_workbook(
            workbook_path,
            workbook_path,
            events,
            max_lots=0,
        )

    if not box_only:
        events = histograms.load_measurement_tests(workbook_path)
        histograms.add_charts_to_workbook(
            workbook_path,
            workbook_path,
            events,
            max_lots=0,
        )

