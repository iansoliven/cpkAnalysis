from __future__ import annotations

from typing import Any

from cpkanalysis.plugins import PluginDescriptor


def descriptor() -> PluginDescriptor:
    """Return a descriptor for the sample summary logger plugin."""
    return PluginDescriptor(
        plugin_id="sample.summary_logger",
        name="Sample Summary Logger",
        description="Writes the summary row count to stdout after statistics.",
        factory=_create_listener,
        events=("SummaryReadyEvent",),
        default_enabled=False,
        default_priority=0,
        thread_safe=True,
        source="Sample/plugins/demo_summary.py",
    )


def _create_listener(parameters: dict[str, Any]) -> Any:
    message_template = parameters.get("message", "[sample] Summary rows: {rows}")

    class _SummaryLogger:
        def __init__(self, template: str) -> None:
            self._template = template

        def handle(self, event: Any) -> None:
            summary = getattr(event, "summary", None)
            if summary is None:
                return None
            try:
                rows = len(summary)
            except Exception:
                rows = "unknown"
            print(self._template.format(rows=rows))
            return None

    return _SummaryLogger(message_template)
