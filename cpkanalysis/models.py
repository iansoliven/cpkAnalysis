from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import pandas as pd

LimitSource = Literal["what_if", "spec", "stdf", "unset"]
OutlierMethod = Literal["none", "iqr", "stdev"]
SiteDataStatus = Literal["available", "unavailable", "unknown"]


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


@dataclass(frozen=True)
class PluginConfig:
    """Configuration block describing an enabled pipeline plugin."""

    plugin_id: str
    enabled: bool = True
    priority: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def normalized_parameters(self) -> dict[str, Any]:
        return dict(self.parameters)


@dataclass
class AnalysisInputs:
    """Aggregated configuration for an analysis run."""

    sources: list[SourceFile]
    output: Path
    template: Path | None = None
    template_sheet: str | None = None
    outliers: OutlierOptions = field(default_factory=OutlierOptions)
    generate_histogram: bool = True
    generate_cdf: bool = True
    generate_time_series: bool = True
    generate_yield_pareto: bool = False
    pareto_first_failure_only: bool = False
    display_decimals: int = 4
    plugins: list[PluginConfig] = field(default_factory=list)
    max_render_processes: int | None = None
    histogram_rug: bool = False
    enable_site_breakdown: bool = False
    site_data_status: SiteDataStatus = "unknown"
    keep_session: bool = False
    include_site_rows_in_cpk: bool = False

    def __post_init__(self) -> None:
        self.output = self.output.expanduser().resolve()
        if self.template is not None:
            self.template = self.template.expanduser().resolve()
        if self.template_sheet:
            self.template_sheet = self.template_sheet.strip()
            if not self.template_sheet:
                self.template_sheet = None
        try:
            decimals = int(self.display_decimals)
        except (TypeError, ValueError):
            decimals = 4
        if decimals < 0:
            decimals = 0
        if decimals > 9:
            decimals = 9
        self.display_decimals = decimals
        if self.max_render_processes is not None:
            try:
                value = int(self.max_render_processes)
            except (TypeError, ValueError):
                value = None
            else:
                if value <= 0:
                    value = None
            self.max_render_processes = value
        self.histogram_rug = bool(self.histogram_rug)
        self.enable_site_breakdown = bool(self.enable_site_breakdown)
        if self.site_data_status not in ("available", "unavailable", "unknown"):
            self.site_data_status = "unknown"
        self.keep_session = bool(self.keep_session)
        self.include_site_rows_in_cpk = bool(self.include_site_rows_in_cpk)
        self.pareto_first_failure_only = bool(self.pareto_first_failure_only)


@dataclass(frozen=True)
class SiteDescription:
    """Configuration extracted from an SDR record."""

    head_num: int
    site_group: int | None
    site_numbers: tuple[int, ...]
    handler_type: str | None = None
    handler_id: str | None = None
    card_type: str | None = None
    card_id: str | None = None
    load_type: str | None = None
    load_id: str | None = None
    dib_type: str | None = None
    dib_id: str | None = None
    cable_type: str | None = None
    cable_id: str | None = None
    contactor_type: str | None = None
    contactor_id: str | None = None
    laser_type: str | None = None
    laser_id: str | None = None
    extractor_type: str | None = None
    extractor_id: str | None = None


@dataclass
class IngestResult:
    """Result bundle produced by STDF ingestion."""

    frame: "pd.DataFrame"
    test_catalog: "pd.DataFrame"
    per_file_stats: list[dict[str, Any]]
    raw_store_path: Path
    site_descriptions: tuple[SiteDescription, ...] = field(default_factory=tuple)

