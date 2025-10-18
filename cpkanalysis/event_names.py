"""Shared string constants for pipeline event identifiers."""

from __future__ import annotations

PIPELINE_EVENT = "PipelineEvent"
INGEST_READY_EVENT = "IngestReadyEvent"
FILTERED_READY_EVENT = "FilteredReadyEvent"
SUMMARY_READY_EVENT = "SummaryReadyEvent"
YIELD_PARETO_READY_EVENT = "YieldParetoReadyEvent"
WORKBOOK_READY_EVENT = "WorkbookReadyEvent"
TEMPLATE_APPLIED_EVENT = "TemplateAppliedEvent"
METADATA_WRITTEN_EVENT = "MetadataWrittenEvent"

DEFAULT_EVENT_NAMES: tuple[str, ...] = (PIPELINE_EVENT,)
