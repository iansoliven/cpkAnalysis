from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis import cli, plugins
from cpkanalysis.models import PluginConfig, AnalysisInputs


class DummyRegistry:
    def __init__(self, descriptors: dict[str, plugins.PluginDescriptor]) -> None:
        self._descriptors = descriptors
        self.discovered = False

    def discover(self) -> None:
        self.discovered = True

    def descriptors(self) -> dict[str, plugins.PluginDescriptor]:
        return dict(self._descriptors)

    def get(self, plugin_id: str) -> plugins.PluginDescriptor | None:
        return self._descriptors.get(plugin_id)


def test_cli_scan_writes_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "one.stdf").write_text("data")
    (input_dir / "two.stdf").write_text("data")

    metadata_path = tmp_path / "sources.json"
    exit_code = cli.main(["scan", str(input_dir), "--metadata", str(metadata_path)])
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Recorded 2 STDF file(s)" in captured

    data = json.loads(metadata_path.read_text())
    recorded_paths = {Path(entry["path"]) for entry in data}
    assert recorded_paths == {p.resolve() for p in input_dir.glob("*.stdf")}


def test_cli_run_invokes_pipeline_with_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "demo.xlsx"
    template_file.write_text("template")

    metadata_file = tmp_path / "meta.json"
    inputs = [{"path": str((tmp_path / "input.stdf").resolve())}]
    metadata_file.write_text(json.dumps(inputs))

    recorded: dict[str, Any] = {}

    def fake_run_analysis(config: AnalysisInputs, *, registry: Any | None = None) -> dict[str, Any]:
        recorded["config"] = config
        recorded["registry"] = registry
        return {
            "output": str(config.output),
            "metadata": "",
            "summary_rows": 0,
            "measurement_rows": 0,
            "outlier_removed": 0,
            "yield_rows": 0,
            "pareto_rows": 0,
            "stage_timings": {},
        }

    descriptor = plugins.PluginDescriptor(
        plugin_id="plugin.a",
        name="Plugin A",
        description="Test plugin",
        factory=lambda params: object(),  # pragma: no cover - behaviour not needed
        default_enabled=False,
        default_priority=2,
    )

    def fake_registry(*, workspace_dir: Path | None = None) -> DummyRegistry:
        assert workspace_dir is not None
        return DummyRegistry({"plugin.a": descriptor})

    monkeypatch.setattr(cli, "PluginRegistry", fake_registry)
    monkeypatch.setattr(cli, "run_analysis", fake_run_analysis)

    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        "\n".join(
            [
                "[[plugins]]",
                'id = "plugin.a"',
                "enabled = false",
                "priority = 1",
                '[parameters]',
                'alpha = "0.1"',
            ]
        )
    )

    output_path = tmp_path / "output.xlsx"
    exit_code = cli.main(
        [
            "run",
            "--metadata",
            str(metadata_file),
            "--output",
            str(output_path),
            "--template",
            str(template_dir),
            "--plugin-profile",
            str(profile_path),
            "--plugin",
            "enable:plugin.a",
            "--plugin",
            "param:plugin.a:alpha=0.7",
        ]
    )
    assert exit_code == 0
    config: AnalysisInputs = recorded["config"]
    assert config.output.name == "output.xlsx"
    assert len(config.sources) == 1
    assert config.plugins[0] == PluginConfig(plugin_id="plugin.a", enabled=True, priority=1, parameters={"alpha": "0.7"})

    captured = capsys.readouterr().out
    assert "command-line overrides applied to plugin 'plugin.a'" in captured


def test_cli_post_process_calls_menu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "report.xlsx"
    workbook_path.write_text("wb")
    metadata_path = tmp_path / "report.json"
    metadata_path.write_text("{}")

    captured: dict[str, Any] = {}

    def fake_create_context(*, workbook_path: Path, metadata_path: Path | None) -> object:
        captured["context_args"] = (workbook_path, metadata_path)
        return {"workbook": workbook_path}

    def fake_open_cli_menu(context: Any) -> None:
        captured["menu_context"] = context

    monkeypatch.setattr(cli.postprocess, "create_context", fake_create_context)
    monkeypatch.setattr(cli.postprocess, "open_cli_menu", fake_open_cli_menu)

    exit_code = cli.main(
        [
            "post-process",
            "--workbook",
            str(workbook_path),
            "--metadata",
            str(metadata_path),
        ]
    )
    assert exit_code == 0
    assert captured["context_args"][0] == workbook_path.resolve()
    assert captured["context_args"][1] == metadata_path.resolve()
    assert captured["menu_context"] == {"workbook": workbook_path.resolve()}


def test_cli_move_template_invokes_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workbook_path = tmp_path / "out.xlsx"
    workbook_path.write_text("wb")

    called: dict[str, Any] = {}

    def fake_move(*, workbook_path: Path, sheet_name: str | None) -> str:
        called["args"] = (workbook_path, sheet_name)
        return "Template Sheet"

    monkeypatch.setattr(cli, "run_move_to_template", fake_move)

    exit_code = cli.main(
        [
            "move-template",
            "--workbook",
            str(workbook_path),
            "--sheet",
            "Template Sheet",
        ]
    )
    assert exit_code == 0
    assert called["args"] == (workbook_path.resolve(), "Template Sheet")
