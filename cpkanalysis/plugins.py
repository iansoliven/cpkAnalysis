from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

try:  # pragma: no cover - stdlib name in 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    try:
        import tomli as tomllib  # type: ignore[assignment]
    except ImportError:
        tomllib = None  # type: ignore[assignment]
        warnings.warn(
            "TOML support requires Python 3.11+ or the 'tomli' package. Plugin profiles may not load.",
            RuntimeWarning,
        )

try:  # pragma: no cover - Python 3.10+
    from importlib.metadata import entry_points, EntryPoint
except ImportError:  # pragma: no cover - fallback for <3.10
    from importlib_metadata import entry_points, EntryPoint  # type: ignore[assignment]


class PluginRegistryError(Exception):
    """Raised when plugin discovery or instantiation fails."""


@dataclass
class PluginDescriptor:
    """Metadata describing a pipeline plugin."""

    plugin_id: str
    name: str
    description: str
    factory: Callable[[Dict[str, Any]], Any]
    events: Tuple[str, ...] = ("PipelineEvent",)
    default_enabled: bool = True
    default_priority: int = 0
    thread_safe: bool = False
    source: str = "builtin"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Registry that discovers and instantiates pipeline plugins."""

    ENTRYPOINT_GROUP = "cpkanalysis.pipeline_plugins"

    def __init__(self, *, workspace_dir: Optional[Path] = None) -> None:
        self._workspace_dir = workspace_dir
        self._descriptors: Dict[str, PluginDescriptor] = {}
        self._discovered = False
        self._register_builtin_plugins()

    def register_descriptor(self, descriptor: PluginDescriptor) -> None:
        self._descriptors[descriptor.plugin_id] = descriptor

    def discover(self) -> None:
        if self._discovered:
            return
        self._load_entry_points()
        self._load_workspace_manifests()
        self._discovered = True

    def descriptors(self) -> Dict[str, PluginDescriptor]:
        self.discover()
        return dict(self._descriptors)

    def get(self, plugin_id: str) -> Optional[PluginDescriptor]:
        self.discover()
        return self._descriptors.get(plugin_id)

    def create_listener(
        self,
        plugin_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PluginDescriptor, Any]:
        descriptor = self.get(plugin_id)
        if descriptor is None:
            raise PluginRegistryError(f"Unknown plugin id '{plugin_id}'.")
        params = dict(parameters or {})
        listener = descriptor.factory(params)
        if not hasattr(listener, "handle"):
            raise PluginRegistryError(
                f"Plugin '{plugin_id}' returned object without 'handle' method."
            )
        return descriptor, listener

    # Discovery helpers -------------------------------------------------

    def _load_entry_points(self) -> None:
        try:
            group = entry_points().select(group=self.ENTRYPOINT_GROUP)  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - older API surface
            group = entry_points().get(self.ENTRYPOINT_GROUP, [])  # type: ignore[index]
        for ep in group:
            self._consume_entry_point(ep)

    def _consume_entry_point(self, ep: EntryPoint) -> None:
        loaded = ep.load()
        self._consume_descriptor_object(loaded, source=f"entry_point:{ep.name}")

    def _load_workspace_manifests(self) -> None:
        if self._workspace_dir is None or not self._workspace_dir.exists():
            return
        for path in sorted(self._workspace_dir.glob("*.toml")):
            descriptor = self._descriptor_from_manifest(path)
            if descriptor is not None:
                self.register_descriptor(descriptor)

    def _descriptor_from_manifest(self, path: Path) -> Optional[PluginDescriptor]:
        if tomllib is None:
            raise PluginRegistryError(
                f"TOML support unavailable for manifest at {path}; "
                "Python 3.11+ or tomllib module required."
            )
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - user supplied data
            raise PluginRegistryError(f"Failed to parse plugin manifest {path}: {exc}") from exc

        plugin_section = payload.get("plugin") or payload
        factory_spec = plugin_section.get("factory")
        if not factory_spec:
            raise PluginRegistryError(f"Manifest {path} missing 'factory' attribute.")
        factory = self._resolve_object(factory_spec)
        descriptor_obj = factory()
        descriptor = self._normalize_descriptor(descriptor_obj, source=str(path))
        explicit_id = plugin_section.get("id")
        if explicit_id:
            descriptor.plugin_id = explicit_id
        descriptor.name = plugin_section.get("name", descriptor.name)
        descriptor.description = plugin_section.get("description", descriptor.description)
        events = plugin_section.get("events")
        if events:
            descriptor.events = tuple(events)
        if "default_enabled" in plugin_section:
            descriptor.default_enabled = bool(plugin_section["default_enabled"])
        if "default_priority" in plugin_section:
            descriptor.default_priority = int(plugin_section["default_priority"])
        if "thread_safe" in plugin_section:
            descriptor.thread_safe = bool(plugin_section["thread_safe"])
        descriptor.source = str(path)
        return descriptor

    def _consume_descriptor_object(self, obj: Any, *, source: str) -> None:
        if obj is None:
            return
        descriptors: Iterable[Any]
        if isinstance(obj, PluginDescriptor):
            descriptors = [obj]
        elif isinstance(obj, Sequence):
            descriptors = obj
        else:
            descriptors = [obj()]

        for descriptor_obj in descriptors:
            descriptor = self._normalize_descriptor(descriptor_obj, source=source)
            self.register_descriptor(descriptor)

    def _normalize_descriptor(self, descriptor_obj: Any, *, source: str) -> PluginDescriptor:
        if isinstance(descriptor_obj, PluginDescriptor):
            descriptor = descriptor_obj
        elif isinstance(descriptor_obj, dict):
            descriptor = PluginDescriptor(
                plugin_id=descriptor_obj["plugin_id"],
                name=descriptor_obj.get("name", descriptor_obj["plugin_id"]),
                description=descriptor_obj.get("description", ""),
                factory=descriptor_obj["factory"],
                events=tuple(descriptor_obj.get("events", ("PipelineEvent",))),
                default_enabled=bool(descriptor_obj.get("default_enabled", True)),
                default_priority=int(descriptor_obj.get("default_priority", 0)),
                thread_safe=bool(descriptor_obj.get("thread_safe", False)),
                source=descriptor_obj.get("source", source),
                metadata=dict(descriptor_obj.get("metadata", {})),
            )
        else:
            raise PluginRegistryError(
                f"Unsupported plugin descriptor object from {source}: {type(descriptor_obj)!r}"
            )

        if not descriptor.plugin_id:
            raise PluginRegistryError(f"Descriptor from {source} missing plugin_id.")
        descriptor.source = descriptor.source or source
        return descriptor

    def _resolve_object(self, spec: str) -> Callable[[], Any]:
        module_name, _, attr = spec.partition(":")
        if not module_name or not attr:
            raise PluginRegistryError(f"Invalid factory spec '{spec}'. Expected 'module:callable'.")
        module = importlib.import_module(module_name)
        try:
            target = getattr(module, attr)
        except AttributeError as exc:  # pragma: no cover - user supplied data
            raise PluginRegistryError(f"Factory '{spec}' not found: {exc}") from exc
        if not callable(target):
            raise PluginRegistryError(f"Factory '{spec}' is not callable.")
        return target

    # Built-in sample plugins -------------------------------------------

    def _register_builtin_plugins(self) -> None:
        descriptor = PluginDescriptor(
            plugin_id="builtin.summary_logger",
            name="Summary Logger",
            description="Logs summary row counts after the statistics stage.",
            factory=self._create_summary_logger,
            events=("SummaryReadyEvent",),
            default_enabled=False,
            default_priority=0,
            thread_safe=True,
            source="builtin",
            metadata={"category": "demo"},
        )
        self.register_descriptor(descriptor)

    @staticmethod
    def _create_summary_logger(parameters: Dict[str, Any]) -> Any:
        message_template = parameters.get("message", "Summary rows: {rows}")

        class _SummaryLogger:
            def __init__(self, template: str) -> None:
                self._template = template

            def handle(self, event: Any) -> None:
                summary = getattr(event, "summary", None)
                if summary is None:
                    return None
                try:
                    row_count = len(summary)
                except Exception:  # pragma: no cover - defensive fallback
                    row_count = "unknown"
                try:
                    print(self._template.format(rows=row_count))
                except Exception:
                    print(f"Summary rows: {row_count}")
                return None

        return _SummaryLogger(message_template)
