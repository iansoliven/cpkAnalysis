from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import pandas as pd

LimitSource = Literal["what_if", "spec", "stdf", "unset"]
OutlierMethod = Literal["none", "iqr", "stdev"]


@dataclass(frozen=True)
class SourceFile:
    """Container describing an input STDF file."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.expanduser().resolve())

    @property
    def file_name(self) -> str:
        return self.path.name


@dataclass
class OutlierOptions:
    """User-controlled outlier filtering configuration."""

    method: OutlierMethod = "none"
    k: float = 1.5

    def is_active(self) -> bool:
        return self.method != "none" and self.k > 0


@dataclass
class AnalysisInputs:
    """Aggregated configuration for an analysis run."""

    sources: list[SourceFile]
    output: Path
    template: Path | None = None
    outliers: OutlierOptions = field(default_factory=OutlierOptions)
    generate_histogram: bool = True
    generate_cdf: bool = True
    generate_time_series: bool = True

    def __post_init__(self) -> None:
        self.output = self.output.expanduser().resolve()
        if self.template is not None:
            self.template = self.template.expanduser().resolve()


@dataclass
class IngestResult:
    """Result bundle produced by STDF ingestion."""

    frame: "pd.DataFrame"
    test_catalog: "pd.DataFrame"
    per_file_stats: list[dict[str, Any]]
    raw_store_path: Path

