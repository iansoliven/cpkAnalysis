from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis import plugin_profiles, plugins
from cpkanalysis.models import PluginConfig


def manifest_factory() -> plugins.PluginDescriptor:
    return plugins.PluginDescriptor(
        plugin_id="manifest.factory",
        name="Manifest Factory",
        description="Created via manifest",
        factory=lambda params: _HandleRecorder(params),
        events=("summary_ready",),
        default_enabled=False,
        default_priority=5,
        thread_safe=True,
        source="",
    )


class _HandleRecorder:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = dict(params)

    def handle(self, event: Any) -> None:  # pragma: no cover - smoke behaviour
        return None


def test_registry_includes_builtin_summary_logger() -> None:
    registry = plugins.PluginRegistry()
    descriptors = registry.descriptors()
    assert "builtin.summary_logger" in descriptors

    descriptor, listener = registry.create_listener("builtin.summary_logger", {"message": "Rows: {rows}"})
    assert descriptor.plugin_id == "builtin.summary_logger"
    assert hasattr(listener, "handle")


def test_registry_loads_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEntryPoint:
        def __init__(self) -> None:
            self.name = "demo_plugin"

        def load(self) -> plugins.PluginDescriptor:
            return plugins.PluginDescriptor(
                plugin_id="entry.point.plugin",
                name="Entry Plugin",
                description="Loaded from entry point",
                factory=lambda params: _HandleRecorder(params),
                source="",
            )

    class FakeEntryPoints:
        def select(self, group: str):
            assert group == plugins.PluginRegistry.ENTRYPOINT_GROUP
            return [FakeEntryPoint()]

    monkeypatch.setattr(plugins, "entry_points", lambda: FakeEntryPoints())

    registry = plugins.PluginRegistry()
    descriptors = registry.descriptors()
    assert "entry.point.plugin" in descriptors
    assert descriptors["entry.point.plugin"].source == "entry_point:demo_plugin"


def test_registry_loads_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        "\n".join(
            [
                "[plugin]",
                'factory = "tests.test_plugin_system:manifest_factory"',
                'id = "manifest.override"',
                'name = "Manifest Override"',
                'description = "Manifest description"',
                "events = [\"summary_ready\"]",
                "default_enabled = false",
                "default_priority = 7",
                "thread_safe = true",
            ]
        ),
        encoding="utf-8",
    )

    registry = plugins.PluginRegistry(workspace_dir=tmp_path)
    descriptors = registry.descriptors()
    descriptor = descriptors["manifest.override"]
    assert descriptor.name == "Manifest Override"
    assert descriptor.description == "Manifest description"
    assert descriptor.default_priority == 7
    assert descriptor.thread_safe is True
    assert descriptor.events == ("summary_ready",)
    _, listener = registry.create_listener("manifest.override", {"alpha": 1})
    assert hasattr(listener, "handle")
    assert getattr(listener, "params") == {"alpha": 1}


def test_create_listener_validates_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_factory(params: dict[str, Any]) -> object:
        return object()

    descriptor = plugins.PluginDescriptor(
        plugin_id="invalid.handle",
        name="Broken plugin",
        description="Returns object without handle",
        factory=bad_factory,
    )
    registry = plugins.PluginRegistry()
    registry.register_descriptor(descriptor)

    with pytest.raises(plugins.PluginRegistryError):
        registry.create_listener("invalid.handle")


def test_plugin_profile_round_trip(tmp_path: Path) -> None:
    configs = [
        PluginConfig(plugin_id="p1", enabled=True, priority=10, parameters={"threshold": "0.5"}),
        PluginConfig(plugin_id="p2", enabled=False, priority=None, parameters={}),
    ]
    profile_path = tmp_path / "profiles" / "post_processing_profile.toml"
    plugin_profiles.save_plugin_profile(profile_path, configs)
    loaded = plugin_profiles.load_plugin_profile(profile_path)
    assert set(loaded.keys()) == {"p1", "p2"}
    assert loaded["p1"].enabled is True
    assert loaded["p1"].parameters == {"threshold": "0.5"}
    assert loaded["p2"].enabled is False
    assert loaded["p2"].priority is None
