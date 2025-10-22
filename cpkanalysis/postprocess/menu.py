"""Menu handling for post-processing capabilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Dict, List, Optional, Tuple

from .context import PostProcessContext
from .io_adapters import PostProcessIO
from . import actions

__all__ = ["loop"]


ActionHandler = Callable[[PostProcessContext, PostProcessIO, Optional[dict]], dict]


@dataclass(frozen=True)
class ActionDefinition:
    key: str
    label: str
    handler: ActionHandler


ACTION_DEFINITIONS: Tuple[ActionDefinition, ...] = (
    ActionDefinition("update_stdf_limits", "Update STDF Limits", actions.update_stdf_limits),
    ActionDefinition("apply_spec_limits", "Apply Spec / What-If Limits", actions.apply_spec_limits),
    ActionDefinition("calculate_proposed_limits", "Calculate Proposed Limits", actions.calculate_proposed_limits),
    ActionDefinition(
        "calculate_proposed_limits_grr",
        "Calculate Proposed Limits (GRR)",
        actions.calculate_proposed_limits_grr,
    ),
)

logger = logging.getLogger(__name__)


def loop(context: PostProcessContext, *, io: PostProcessIO) -> None:
    """Run the menu loop until the user exits."""
    last_action: Optional[Tuple[ActionDefinition, dict]] = None

    while True:
        io.print()
        io.print("Post-Processing Menu")
        io.print("--------------------")
        for index, action in enumerate(ACTION_DEFINITIONS, start=1):
            io.print(f"{index}. {action.label}")
        offset = len(ACTION_DEFINITIONS)
        extra_labels = [
            "Re-run last action" + ("" if last_action else " (n/a)"),
            "Reload workbook from disk",
            "View audit log",
            "Exit",
        ]
        for index, label in enumerate(extra_labels, start=offset + 1):
            io.print(f"{index}. {label}")

        choice = io.prompt_choice(
            "Select an option:",
            [a.label for a in ACTION_DEFINITIONS] + extra_labels,
            show_options=False,
        )
        if choice < len(ACTION_DEFINITIONS):
            action_def = ACTION_DEFINITIONS[choice]
            params: Optional[dict] = None
            result = _execute_action(context, io, action_def, params)
            if result is not None:
                last_action = (action_def, result.get("replay_params", params) or {})
            continue

        remaining = choice - len(ACTION_DEFINITIONS)
        if remaining == 0:
            # Re-run last action
            if not last_action:
                io.warn("No previous action to re-run.")
                continue
            action_def, params = last_action
            result = _execute_action(context, io, action_def, params)
            if result is not None:
                last_action = (action_def, result.get("replay_params", params) or {})
            continue

        if remaining == 1:
            io.info("Reloading workbook...")
            context.reload()
            io.info("Workbook reloaded.")
            continue

        if remaining == 2:
            _print_audit_log(context, io)
            continue

        # Exit
        if context.dirty:
            if io.confirm("You have unsaved changes. Save before exiting?", default=True):
                try:
                    context.save()
                    io.info("Changes saved.")
                except (OSError, ValueError) as exc:
                    logger.exception("Failed to save post-processing changes before exit")
                    io.warn(f"Failed to save changes: {exc}")
        io.info("Exiting post-processing menu.")
        break


def _execute_action(
    context: PostProcessContext,
    io: PostProcessIO,
    action_def: ActionDefinition,
    params: Optional[dict],
) -> Optional[dict]:
    try:
        result = action_def.handler(context, io, params)
    except actions.ActionCancelled as exc:
        # Display the cancellation reason if provided
        message = str(exc) if str(exc) else "Action cancelled."
        io.warn(message)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Action '%s' failed", action_def.key)
        io.warn(f"{action_def.label} failed: {exc}")
        return None

    summary = result.get("summary")
    if summary:
        io.info(summary)
    for warning in result.get("warnings", []):
        io.warn(warning)

    if result.get("mark_dirty", True):
        context.mark_dirty()

    audit_entry = result.get("audit")
    if audit_entry:
        audit_entry.setdefault("action", action_def.key)
        context.add_audit_entry(audit_entry)

    if context.dirty and result.get("auto_save", True):
        try:
            context.save()
            io.info("Changes saved.")
        except (OSError, ValueError) as exc:
            logger.exception("Failed to auto-save post-processing changes")
            io.warn(f"Failed to save changes automatically: {exc}")
    return result


def _print_audit_log(context: PostProcessContext, io: PostProcessIO) -> None:
    runs = context.metadata.get("post_processing", {}).get("runs", [])
    if context.audit_log:
        runs = runs + context.audit_log
    if not runs:
        io.info("No post-processing runs recorded.")
        return
    io.print("Recorded post-processing runs:")
    for entry in runs:
        action = entry.get("action", "unknown")
        timestamp = entry.get("timestamp", "n/a")
        scope = entry.get("scope", "n/a")
        tests = entry.get("tests", [])
        test_summary = ", ".join(tests) if tests else "none"
        io.print(f"- {timestamp}: {action} (scope={scope}; tests={test_summary})")
