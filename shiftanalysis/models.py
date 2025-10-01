from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SourceFile:
    path: Path
    lot: str
    event: str
    interval: str
    file_type: Literal["xlsx", "stdf"]


@dataclass
class SummaryRow:
    lot: str
    event: str
    interval: str
    file_name: str
    source_path: Path
    pass_count: int
    fail_count: int


@dataclass
class MeasurementRow:
    lot: str
    event: str
    interval: str
    source: str
    status: str
    test_number: str
    test_name: str
    test_unit: str
    low_limit: float | None
    high_limit: float | None
    measurement: float | str | None
    serial_number: str

