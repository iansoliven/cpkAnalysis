from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpkanalysis import gui


def test_gui_main_invokes_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {}

    class DummyGUI(gui.CPKAnalysisGUI):
        def launch(self) -> None:  # pragma: no cover - invoked via main
            called["launch"] = True

    monkeypatch.setattr(gui, "CPKAnalysisGUI", DummyGUI)
    # Pass an empty argv list to avoid pytest flags leaking into sys.argv
    assert gui.main([]) == 0
    assert called.get("launch") is True


def test_yes_no_uses_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert gui._yes_no("prompt", default=True) is True
    monkeypatch.setattr("builtins.input", lambda prompt: "No")
    assert gui._yes_no("prompt", default=True) is False


def test_prompt_parameters_adds_and_updates(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    responses = iter(["alpha=0.7", "beta= ", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))
    instance = gui.CPKAnalysisGUI()
    results = instance._prompt_parameters({})
    assert results == {"alpha": "0.7"}
    capsys.readouterr()
