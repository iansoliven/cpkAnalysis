from __future__ import annotations

import itertools
import json
import shutil
import sys
import threading
import time
from dataclasses import dataclass, replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Protocol, List, Type
import logging

import pandas as pd

from . import ingest, outliers, stats, workbook_builder
from .models import AnalysisInputs, IngestResult
from .plugins import PluginRegistry, PluginRegistryError
from .move_to_template import run as move_to_template, apply_template
from .tools.update_cpk_formulas import update_workbook as apply_cpk_formulas
from openpyxl import Workbook
from .event_names import (
    DEFAULT_EVENT_NAMES,
    FILTERED_READY_EVENT,
    INGEST_READY_EVENT,
    METADATA_WRITTEN_EVENT,
    PIPELINE_EVENT,
    SUMMARY_READY_EVENT,
    TEMPLATE_APPLIED_EVENT,
    WORKBOOK_READY_EVENT,
    YIELD_PARETO_READY_EVENT,
)

logger = logging.getLogger(__name__)

StageHandler = Callable[["PipelineContext"], "PipelineContext"]


@dataclass(frozen=True)
class PluginExecutionRecord:
    """Immutable record capturing runtime plugin state for metadata."""

    plugin_id: str
    name: str
    events: tuple[str, ...]
    thread_safe: bool
    priority: int
    source: str
    parameters: tuple[tuple[str, Any], ...]


class PipelineListener(Protocol):
    """Listener interface for reacting to pipeline events."""

    def handle(self, event: "PipelineEvent") -> Optional["PipelineContext"]:
        """Process an event and optionally return a replacement context."""


@dataclass(order=True)
class _RegisteredListener:
    priority: int
    order: int
    listener: PipelineListener


class EventBus:
    """Minimal event bus for pipeline lifecycle notifications."""

    def __init__(self) -> None:
        self._listeners: Dict[Type["PipelineEvent"], List[_RegisteredListener]] = {}
        self._order_counter = itertools.count()

    def register(
        self,
        event_type: Type["PipelineEvent"],
        listener: PipelineListener,
        *,
        priority: int = 0,
    ) -> None:
        registered = _RegisteredListener(priority=-priority, order=next(self._order_counter), listener=listener)
        self._listeners.setdefault(event_type, []).append(registered)

    def dispatch(self, event: "PipelineEvent", context: "PipelineContext") -> "PipelineContext":
        """Emit an event to registered listeners, returning the (potentially updated) context."""
        listeners = self._listeners_for(type(event))
        current_context = context
        current_event = event
        for registration in listeners:
            result = registration.listener.handle(current_event)
            if result is not None and result is not current_context:
                current_context = result
                current_event = replace(current_event, context=current_context)
        return current_context

    def _listeners_for(self, event_type: Type["PipelineEvent"]) -> List[_RegisteredListener]:
        collected: List[_RegisteredListener] = []
        for cls in event_type.__mro__:
            if not issubclass(cls, PipelineEvent):
                continue
            collected.extend(self._listeners.get(cls, []))
        collected.sort()
        return collected


@dataclass(frozen=True)
class PipelineEvent:
    """Base event emitted after a pipeline stage completes."""

    stage: str
    elapsed: float
    context: "PipelineContext"


@dataclass(frozen=True)
class IngestReadyEvent(PipelineEvent):
    ingest_result: IngestResult


@dataclass(frozen=True)
class FilteredReadyEvent(PipelineEvent):
    measurements: pd.DataFrame
    outlier_summary: dict[str, Any]


@dataclass(frozen=True)
class SummaryReadyEvent(PipelineEvent):
    summary: pd.DataFrame
    limit_sources: dict[tuple[str, str, str], dict[str, str]]


@dataclass(frozen=True)
class YieldParetoReadyEvent(PipelineEvent):
    yield_summary: pd.DataFrame
    pareto_summary: pd.DataFrame


@dataclass(frozen=True)
class WorkbookReadyEvent(PipelineEvent):
    output_path: Path


@dataclass(frozen=True)
class TemplateAppliedEvent(PipelineEvent):
    template_sheet: Optional[str]


@dataclass(frozen=True)
class MetadataWrittenEvent(PipelineEvent):
    metadata_path: Path


_EVENT_NAME_MAP: dict[str, Type[PipelineEvent]] = {
    PIPELINE_EVENT: PipelineEvent,
    INGEST_READY_EVENT: IngestReadyEvent,
    FILTERED_READY_EVENT: FilteredReadyEvent,
    SUMMARY_READY_EVENT: SummaryReadyEvent,
    YIELD_PARETO_READY_EVENT: YieldParetoReadyEvent,
    WORKBOOK_READY_EVENT: WorkbookReadyEvent,
    TEMPLATE_APPLIED_EVENT: TemplateAppliedEvent,
    METADATA_WRITTEN_EVENT: MetadataWrittenEvent,
}


@dataclass(frozen=True)
class PipelineContext:
    """Immutable snapshot of the pipeline's state at a point in time."""

    config: AnalysisInputs
    session_dir: Path
    ingest_result: IngestResult | None = None
    filtered_measurements: pd.DataFrame | None = None
    outlier_summary: dict[str, Any] | None = None
    summary: pd.DataFrame | None = None
    limit_sources: dict[tuple[str, str, str], dict[str, str]] | None = None
    site_summary: pd.DataFrame | None = None
    site_limit_sources: dict[tuple[Any, ...], dict[str, str]] | None = None
    yield_summary: pd.DataFrame | None = None
    pareto_summary: pd.DataFrame | None = None
    site_yield_summary: pd.DataFrame | None = None
    site_pareto_summary: pd.DataFrame | None = None
    template_sheet: str | None = None
    metadata_path: Path | None = None
    stage_timings: dict[str, float] = None  # type: ignore[assignment]
    stage_details: dict[str, dict[str, float]] = None  # type: ignore[assignment]
    active_plugins: tuple[str, ...] = ()
    plugins_metadata: tuple[PluginExecutionRecord, ...] = ()
    workbook_obj: Workbook | None = None
    site_data_available: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage_timings is None:
            object.__setattr__(self, "stage_timings", {})
        if self.stage_details is None:
            object.__setattr__(self, "stage_details", {})
        if self.warnings is None:
            object.__setattr__(self, "warnings", ())
        elif not isinstance(self.warnings, tuple):
            object.__setattr__(self, "warnings", tuple(self.warnings))

    def with_updates(self, **changes: Any) -> "PipelineContext":
        """Return a new context with the supplied field updates."""
        if "stage_timings" in changes and changes["stage_timings"] is None:
            # Prevent accidental None assignment; force callers to supply a mapping.
            changes["stage_timings"] = {}
        if "stage_details" in changes and changes["stage_details"] is None:
            changes["stage_details"] = {}
        if "warnings" in changes and changes["warnings"] is not None and not isinstance(changes["warnings"], tuple):
            changes["warnings"] = tuple(changes["warnings"])
        return replace(self, **changes)


class Pipeline:
    """Orchestrates the sequential execution of the analysis stages."""

    def __init__(
        self,
        config: AnalysisInputs,
        *,
        event_bus: Optional[EventBus] = None,
        registry: Optional[PluginRegistry] = None,
    ) -> None:
        self._config = config
        self._session_dir = _create_session_dir()
        self._event_bus = event_bus or EventBus()
        self._registry = registry or PluginRegistry()
        self._stage_handlers: list[tuple[str, str, StageHandler]] = [
            ("ingest", "Ingesting STDF sources", self._stage_ingest),
            ("outliers", "Applying outlier filters", self._stage_outliers),
            ("statistics", "Computing summary statistics", self._stage_statistics),
            ("yield_pareto", "Computing yield and Pareto analysis", self._stage_yield_pareto),
            ("workbook", "Building Excel workbook", self._stage_workbook),
            ("template", "Updating template sheet", self._stage_template),
            ("metadata", "Writing metadata sidecar", self._stage_metadata),
        ]

    def run(self) -> dict[str, Any]:
        """Execute the configured pipeline and return summary information."""
        context = PipelineContext(config=self._config, session_dir=self._session_dir)
        try:
            context = self._install_plugins(context)
            for stage_name, message, handler in self._stage_handlers:
                if stage_name == "yield_pareto" and not self._config.generate_yield_pareto:
                    continue
                if stage_name == "template" and not (self._config.template or self._config.template_sheet):
                    continue
                context = self._run_stage(stage_name, message, handler, context)
            return self._build_result(context)
        finally:
            if self._config.keep_session:
                logger.info("Retaining session directory at '%s' per configuration.", self._session_dir)
            else:
                _cleanup_session_dir(self._session_dir)

    def _run_stage(
        self,
        stage_name: str,
        message: str,
        handler: StageHandler,
        context: PipelineContext,
    ) -> PipelineContext:
        start = time.perf_counter()
        with _spinner(message):
            updated_context = handler(context)
        elapsed = time.perf_counter() - start
        timings = dict(context.stage_timings)
        timings[stage_name] = elapsed
        updated_context = updated_context.with_updates(stage_timings=timings)
        event = self._create_event(stage_name, elapsed, updated_context)
        if event is not None:
            updated_context = self._event_bus.dispatch(event, updated_context)
        return updated_context

    def _install_plugins(self, context: PipelineContext) -> PipelineContext:
        plugin_configs = getattr(context.config, "plugins", [])
        if not plugin_configs:
            return context.with_updates(active_plugins=(), plugins_metadata=())

        self._registry.discover()
        active_ids: list[str] = []
        records: list[PluginExecutionRecord] = []

        for plugin_cfg in plugin_configs:
            if not plugin_cfg.enabled:
                continue
            descriptor, listener = self._registry.create_listener(
                plugin_cfg.plugin_id,
                plugin_cfg.parameters,
            )
            event_names = descriptor.events or DEFAULT_EVENT_NAMES
            event_types: list[Type[PipelineEvent]] = []
            for event_name in event_names:
                event_cls = _EVENT_NAME_MAP.get(event_name)
                if event_cls is None:
                    raise PluginRegistryError(
                        f"Plugin '{descriptor.plugin_id}' references unknown event '{event_name}'."
                    )
                event_types.append(event_cls)
            priority = (
                plugin_cfg.priority
                if plugin_cfg.priority is not None
                else descriptor.default_priority
            )
            for event_cls in event_types:
                self._event_bus.register(event_cls, listener, priority=priority)
            active_ids.append(descriptor.plugin_id)
            raw_params = plugin_cfg.parameters or {}
            sorted_params = tuple(sorted(((str(key), value) for key, value in raw_params.items()), key=lambda item: item[0]))
            records.append(
                PluginExecutionRecord(
                    plugin_id=descriptor.plugin_id,
                    name=descriptor.name,
                    events=tuple(event_names),
                    thread_safe=bool(descriptor.thread_safe),
                    priority=int(priority),
                    source=descriptor.source,
                    parameters=sorted_params,
                )
            )

        return context.with_updates(
            active_plugins=tuple(active_ids),
            plugins_metadata=tuple(records),
        )

    def _stage_ingest(self, context: PipelineContext) -> PipelineContext:
        ingest_result = ingest.ingest_sources(context.config.sources, context.session_dir)
        site_available = ingest.has_site_data(ingest_result.frame)
        warnings = list(context.warnings)
        if context.config.enable_site_breakdown and not site_available:
            warnings.append(
                "Per-site aggregation requested but SITE_NUM data was not detected; proceeding without site breakdown."
            )
        context.config.site_data_status = "available" if site_available else "unavailable"
        return context.with_updates(
            ingest_result=ingest_result,
            site_data_available=site_available,
            warnings=tuple(warnings),
        )

    def _stage_outliers(self, context: PipelineContext) -> PipelineContext:
        assert context.ingest_result is not None, "Ingest stage must run before outlier filtering."
        group_keys = None
        if context.config.enable_site_breakdown and context.site_data_available:
            group_keys = ["file", "site", "test_name", "test_number"]
        filtered_frame, outlier_summary = outliers.apply_outlier_filter(
            context.ingest_result.frame,
            context.config.outliers.method,
            context.config.outliers.k,
            group_keys=group_keys,
        )
        filtered_path = context.session_dir / "filtered_measurements.parquet"
        filtered_frame.to_parquet(filtered_path, engine="pyarrow", index=False)
        return context.with_updates(filtered_measurements=filtered_frame, outlier_summary=outlier_summary)

    def _stage_statistics(self, context: PipelineContext) -> PipelineContext:
        assert context.ingest_result is not None, "Ingest stage must run before statistics."
        assert context.filtered_measurements is not None, "Outlier stage must run before statistics."
        summary_df, limit_sources = stats.compute_summary(
            context.filtered_measurements, context.ingest_result.test_catalog
        )
        site_summary = None
        site_limit_sources = None
        if context.config.enable_site_breakdown and context.site_data_available:
            site_summary, site_limit_sources = stats.compute_summary_by_site(
                context.filtered_measurements,
                context.ingest_result.test_catalog,
            )
        return context.with_updates(
            summary=summary_df,
            limit_sources=limit_sources,
            site_summary=site_summary,
            site_limit_sources=site_limit_sources,
        )

    def _stage_yield_pareto(self, context: PipelineContext) -> PipelineContext:
        if not context.config.generate_yield_pareto:
            return context
        assert context.ingest_result is not None, "Ingest stage must run before yield analysis."
        assert context.filtered_measurements is not None, "Outlier stage must run before yield analysis."
        # Use the unfiltered measurements so yield/pareto stay aligned with the raw data set.
        yield_df, pareto_df = stats.compute_yield_pareto(
            context.ingest_result.frame, context.ingest_result.test_catalog
        )
        site_yield = None
        site_pareto = None
        if context.config.enable_site_breakdown and context.site_data_available:
            site_yield, site_pareto = stats.compute_yield_pareto_by_site(
                context.ingest_result.frame,
                context.ingest_result.test_catalog,
            )
        return context.with_updates(
            yield_summary=yield_df,
            pareto_summary=pareto_df,
            site_yield_summary=site_yield,
            site_pareto_summary=site_pareto,
        )

    def _stage_workbook(self, context: PipelineContext) -> PipelineContext:
        assert context.ingest_result is not None, "Ingest stage must run before workbook."
        assert context.filtered_measurements is not None, "Outlier stage must run before workbook."
        assert context.summary is not None, "Statistics stage must run before workbook."
        assert context.limit_sources is not None, "Statistics stage must run before workbook."
        assert context.outlier_summary is not None, "Outlier stage must run before workbook."
        workbook_timings: dict[str, float] = {}
        workbook_obj = workbook_builder.build_workbook(
            summary=context.summary,
            measurements=context.filtered_measurements,
            test_limits=context.ingest_result.test_catalog,
            limit_sources=context.limit_sources,
            outlier_summary=context.outlier_summary,
            per_file_stats=context.ingest_result.per_file_stats,
            output_path=context.config.output,
            template_path=context.config.template,
            include_histogram=context.config.generate_histogram,
            include_cdf=context.config.generate_cdf,
            include_time_series=context.config.generate_time_series,
            include_yield_pareto=context.config.generate_yield_pareto,
            yield_summary=context.yield_summary,
            pareto_summary=context.pareto_summary,
            site_summary=context.site_summary,
            site_limit_sources=context.site_limit_sources,
            site_yield_summary=context.site_yield_summary,
            site_pareto_summary=context.site_pareto_summary,
            site_enabled=context.config.enable_site_breakdown and context.site_data_available,
            fallback_decimals=context.config.display_decimals,
            temp_dir=context.session_dir,
            timing_collector=workbook_timings,
            max_render_processes=context.config.max_render_processes,
            histogram_rug=context.config.generate_histogram and context.config.histogram_rug,
            workbook_obj=context.workbook_obj,
            defer_save=True,
            include_site_rows=context.config.include_site_rows_in_cpk,
        )
        if workbook_timings:
            details = dict(context.stage_details)
            details["workbook"] = workbook_timings
            return context.with_updates(stage_details=details, workbook_obj=workbook_obj)
        return context.with_updates(workbook_obj=workbook_obj)

    def _stage_template(self, context: PipelineContext) -> PipelineContext:
        template_sheet_used: str | None = None
        workbook_obj = context.workbook_obj

        if workbook_obj is not None:
            if context.config.template or context.config.template_sheet:
                template_sheet_used = apply_template(workbook_obj, context.config.template_sheet)
            save_start = time.perf_counter()
            workbook_obj.save(context.config.output)
            save_elapsed = time.perf_counter() - save_start

            stage_details = dict(context.stage_details)
            workbook_detail = dict(stage_details.get("workbook", {}))
            workbook_detail["workbook.save"] = workbook_detail.get("workbook.save", 0.0) + save_elapsed
            stage_details["workbook"] = workbook_detail

            workbook_obj.close()
            update_sheet = template_sheet_used or context.config.template_sheet
            if update_sheet:
                try:
                    apply_cpk_formulas(context.config.output, update_sheet)
                except ValueError as exc:
                    logger.warning("Skipping CPK formula update: %s", exc)
            return context.with_updates(template_sheet=template_sheet_used, workbook_obj=None, stage_details=stage_details)

        if context.config.template or context.config.template_sheet:
            template_sheet_used = move_to_template(context.config.output, context.config.template_sheet)
        update_sheet = template_sheet_used or context.config.template_sheet
        if update_sheet:
            try:
                apply_cpk_formulas(context.config.output, update_sheet)
            except ValueError as exc:
                logger.warning("Skipping CPK formula update: %s", exc)
        return context.with_updates(template_sheet=template_sheet_used, workbook_obj=None)

    def _stage_metadata(self, context: PipelineContext) -> PipelineContext:
        assert context.ingest_result is not None, "Ingest stage must run before metadata."
        assert context.summary is not None, "Statistics stage must run before metadata."
        assert context.outlier_summary is not None, "Outlier stage must run before metadata."
        metadata = _build_metadata(
            config=context.config,
            ingest_result=context.ingest_result,
            outlier_summary=context.outlier_summary,
            limit_sources=context.limit_sources or {},
            summary=context.summary,
            yield_summary=context.yield_summary,
            pareto_summary=context.pareto_summary,
            site_summary=context.site_summary,
            site_yield_summary=context.site_yield_summary,
            site_pareto_summary=context.site_pareto_summary,
            site_breakdown_requested=context.config.enable_site_breakdown,
            site_breakdown_available=context.site_data_available,
            template_sheet=context.template_sheet,
            plugins=context.plugins_metadata,
            warnings=context.warnings,
        )
        metadata_path = context.config.output.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return context.with_updates(metadata_path=metadata_path)

    def _create_event(
        self,
        stage_name: str,
        elapsed: float,
        context: PipelineContext,
    ) -> Optional[PipelineEvent]:
        if stage_name == "ingest" and context.ingest_result is not None:
            return IngestReadyEvent(
                stage=stage_name,
                elapsed=elapsed,
                context=context,
                ingest_result=context.ingest_result,
            )
        if (
            stage_name == "outliers"
            and context.filtered_measurements is not None
            and context.outlier_summary is not None
        ):
            return FilteredReadyEvent(
                stage=stage_name,
                elapsed=elapsed,
                context=context,
                measurements=context.filtered_measurements,
                outlier_summary=context.outlier_summary,
            )
        if stage_name == "statistics" and context.summary is not None and context.limit_sources is not None:
            return SummaryReadyEvent(
                stage=stage_name,
                elapsed=elapsed,
                context=context,
                summary=context.summary,
                limit_sources=context.limit_sources,
            )
        if (
            stage_name == "yield_pareto"
            and context.yield_summary is not None
            and context.pareto_summary is not None
        ):
            return YieldParetoReadyEvent(
                stage=stage_name,
                elapsed=elapsed,
                context=context,
                yield_summary=context.yield_summary,
                pareto_summary=context.pareto_summary,
            )
        if stage_name == "workbook":
            return WorkbookReadyEvent(
                stage=stage_name,
                elapsed=elapsed,
                context=context,
                output_path=context.config.output,
            )
        if stage_name == "template":
            return TemplateAppliedEvent(
                stage=stage_name,
                elapsed=elapsed,
                context=context,
                template_sheet=context.template_sheet,
            )
        if stage_name == "metadata" and context.metadata_path is not None:
            return MetadataWrittenEvent(
                stage=stage_name,
                elapsed=elapsed,
                context=context,
                metadata_path=context.metadata_path,
            )
        return None

    def _build_result(self, context: PipelineContext) -> dict[str, Any]:
        assert context.ingest_result is not None, "Pipeline must ingest sources before completing."
        assert context.filtered_measurements is not None, "Filtered measurements missing at completion."
        assert context.summary is not None, "Summary statistics missing at completion."
        assert context.outlier_summary is not None, "Outlier summary missing at completion."
        result = {
            "output": str(context.config.output),
            "metadata": str(context.metadata_path) if context.metadata_path else "",
            "session_dir": str(context.session_dir),
            "session_retained": bool(context.config.keep_session),
            "summary_rows": int(len(context.summary)),
            "measurement_rows": int(len(context.filtered_measurements)),
            "outlier_removed": context.outlier_summary.get("removed", 0),
            "template_sheet": context.template_sheet,
            "plugins": list(context.active_plugins),
            "yield_pareto_enabled": context.config.generate_yield_pareto,
            "site_breakdown_requested": context.config.enable_site_breakdown,
            "site_data_available": context.site_data_available,
            "cpk_site_rows_requested": context.config.include_site_rows_in_cpk,
        }
        result["yield_rows"] = int(len(context.yield_summary)) if context.yield_summary is not None else 0
        result["pareto_rows"] = int(len(context.pareto_summary)) if context.pareto_summary is not None else 0
        result["site_summary_rows"] = int(len(context.site_summary)) if context.site_summary is not None else 0
        if context.config.enable_site_breakdown:
            result["site_yield_rows"] = (
                int(len(context.site_yield_summary)) if context.site_yield_summary is not None else 0
            )
            result["site_pareto_rows"] = (
                int(len(context.site_pareto_summary)) if context.site_pareto_summary is not None else 0
            )
        result["site_breakdown_generated"] = context.site_summary is not None
        result["cpk_site_rows_included"] = bool(
            context.config.include_site_rows_in_cpk and context.site_summary is not None
        )
        if context.warnings:
            result["warnings"] = list(context.warnings)
        stage_timings = dict(context.stage_timings)
        result["stage_timings"] = stage_timings
        stage_details = {name: dict(values) for name, values in context.stage_details.items()}
        if stage_details:
            result["stage_details"] = stage_details
        result["elapsed_seconds"] = float(sum(stage_timings.values()))
        return result


def run_analysis(config: AnalysisInputs, *, registry: Optional[PluginRegistry] = None) -> dict[str, Any]:
    """Execute the end-to-end analysis pipeline."""
    return Pipeline(config, registry=registry).run()


def _build_metadata(
    config: AnalysisInputs,
    ingest_result: IngestResult,
    outlier_summary: dict[str, Any],
    limit_sources: dict[tuple[str, str, str], dict[str, str]],
    summary: pd.DataFrame,
    yield_summary: pd.DataFrame | None = None,
    pareto_summary: pd.DataFrame | None = None,
    site_summary: pd.DataFrame | None = None,
    site_yield_summary: pd.DataFrame | None = None,
    site_pareto_summary: pd.DataFrame | None = None,
    *,
    site_breakdown_requested: bool = False,
    site_breakdown_available: bool = False,
    template_sheet: str | None = None,
    plugins: Iterable[PluginExecutionRecord] = (),
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    yield_info: dict[str, Any] | None = None
    if yield_summary is not None:
        yield_info = {
            "rows": int(len(yield_summary)),
        }

    pareto_info: dict[str, Any] | None = None
    if pareto_summary is not None:
        pareto_info = {
            "rows": int(len(pareto_summary)),
        }

    site_summary_info: dict[str, Any] | None = None
    if site_summary is not None:
        site_summary_info = {"rows": int(len(site_summary))}

    site_yield_info: dict[str, Any] | None = None
    if site_yield_summary is not None:
        site_yield_info = {"rows": int(len(site_yield_summary))}

    site_pareto_info: dict[str, Any] | None = None
    if site_pareto_summary is not None:
        site_pareto_info = {"rows": int(len(site_pareto_summary))}

    site_configuration: list[dict[str, Any]] = []
    for description in ingest_result.site_descriptions:
        entry: dict[str, Any] = {
            "head": description.head_num,
            "sites": list(description.site_numbers),
        }
        if description.site_group is not None:
            entry["site_group"] = description.site_group
        optional_fields = {
            "handler_type": description.handler_type,
            "handler_id": description.handler_id,
            "card_type": description.card_type,
            "card_id": description.card_id,
            "load_type": description.load_type,
            "load_id": description.load_id,
            "dib_type": description.dib_type,
            "dib_id": description.dib_id,
            "cable_type": description.cable_type,
            "cable_id": description.cable_id,
            "contactor_type": description.contactor_type,
            "contactor_id": description.contactor_id,
            "laser_type": description.laser_type,
            "laser_id": description.laser_id,
            "extractor_type": description.extractor_type,
            "extractor_id": description.extractor_id,
        }
        entry.update({key: value for key, value in optional_fields.items() if value is not None})
        site_configuration.append(entry)

    return {
        "output": str(config.output),
        "template": str(config.template) if config.template else None,
        "sources": ingest_result.per_file_stats,
        "outlier_filter": outlier_summary,
        "template_sheet": template_sheet,
        "limit_sources": {
            f"{file}|{test}|{number}": sources
            for (file, test, number), sources in limit_sources.items()
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary_counts": {
            "rows": int(len(summary)),
            "tests": int(summary["Test Name"].nunique()) if not summary.empty else 0,
        },
        "analysis_options": {
            "generate_histogram": config.generate_histogram,
            "generate_cdf": config.generate_cdf,
            "generate_time_series": config.generate_time_series,
            "generate_yield_pareto": config.generate_yield_pareto,
            "display_decimals": config.display_decimals,
            "enable_site_breakdown": site_breakdown_requested,
            "include_site_rows_in_cpk": config.include_site_rows_in_cpk,
        },
        "yield_summary": yield_info,
        "pareto_summary": pareto_info,
        "site_summary": site_summary_info,
        "site_yield_summary": site_yield_info,
        "site_pareto_summary": site_pareto_info,
        "site_configuration": site_configuration or None,
        "site_breakdown": {
            "requested": site_breakdown_requested,
            "available": site_breakdown_available,
            "generated": site_summary is not None,
        },
        "warnings": list(warnings),
        "plugins": [
            {
                "id": record.plugin_id,
                "name": record.name,
                "events": list(record.events),
                "thread_safe": record.thread_safe,
                "priority": record.priority,
                "source": record.source,
                "parameters": {key: value for key, value in record.parameters},
            }
            for record in plugins
        ],
    }


def _create_session_dir() -> Path:
    root = Path("temp")
    root.mkdir(exist_ok=True)
    session = root / f"session_{int(time.time() * 1000)}"
    session.mkdir()
    return session


def _cleanup_session_dir(path: Path, *, retries: int = 3, delay: float = 0.5) -> None:
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if attempt == retries - 1:
                logger.warning("Failed to remove session directory '%s': %s", path, exc)
            else:
                time.sleep(delay * (attempt + 1))


class _Spinner:
    BRAILLE_FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834", "\u2826", "\u2827", "\u2807", "\u280f"]
    ASCII_FRAMES = ["-", "\\", "|", "/"]

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._line_length = 0
        self._unicode_supported = _supports_output("".join(self.BRAILLE_FRAMES) + "\u2714\u2716")
        self._frames = self.BRAILLE_FRAMES if self._unicode_supported else self.ASCII_FRAMES
        self._success_symbol = "\u2714" if self._unicode_supported else "[OK]"
        self._failure_symbol = "\u2716" if self._unicode_supported else "[FAIL]"

    def start(self) -> None:
        self._thread.start()

    def stop(self, final_message: str) -> None:
        self._stop.set()
        self._thread.join()
        text = final_message
        pad = max(self._line_length - len(text), 0)
        self._write(f"\r{text}{' ' * pad}\n")

    def success_text(self, message: str) -> str:
        return f"{self._success_symbol} Completed: {message}"

    def failure_text(self, message: str) -> str:
        return f"{self._failure_symbol} Failed: {message}"

    def _spin(self) -> None:
        for frame in itertools.cycle(self._frames):
            if self._stop.is_set():
                break
            text = f"{frame} {self.message}"
            self._line_length = len(text)
            self._write(f"\r{text}")
            time.sleep(0.1)

    @staticmethod
    def _write(text: str) -> None:
        try:
            sys.stdout.write(text)
        except UnicodeEncodeError:
            sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
        sys.stdout.flush()


@contextmanager
def _spinner(message: str):
    spinner = _Spinner(message)
    spinner.start()
    try:
        yield
    except BaseException:
        spinner.stop(spinner.failure_text(message))
        raise
    else:
        spinner.stop(spinner.success_text(message))


def _supports_output(sample: str) -> bool:
    encoding = sys.stdout.encoding or "utf-8"
    try:
        sample.encode(encoding)
        return True
    except (UnicodeEncodeError, LookupError):
        return False
