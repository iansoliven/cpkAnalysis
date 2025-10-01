"""Placeholder for future graphical interface integration (TBD)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ApplicationState:
    """TBD: container for GUI runtime state."""

    workspace_path: str = ''
    last_run_output: str = ''


class ShiftAnalysisGUI:
    """TBD: GUI scaffolding to be replaced with concrete implementation."""

    def __init__(self, state: ApplicationState | None = None) -> None:
        self.state = state or ApplicationState()

    def launch(self) -> None:
        """TBD: initialize and display GUI widgets."""
        raise NotImplementedError("TBD: GUI launch workflow")

    def shutdown(self) -> None:
        """TBD: perform cleanup before application exit."""
        raise NotImplementedError("TBD: GUI shutdown workflow")

