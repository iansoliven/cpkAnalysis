"""Input/output abstractions for post-processing menu."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

__all__ = [
    "PostProcessIO",
    "CliIO",
    "GuiIO",
]


class PostProcessIO:
    """Abstract user-interaction surface."""

    def prompt(self, message: str, *, default: str | None = None) -> str:
        raise NotImplementedError

    def prompt_float(self, message: str, *, default: float | None = None) -> float | None:
        text = self.prompt(message, default="" if default is None else str(default))
        if not text.strip():
            return default
        try:
            return float(text)
        except ValueError:
            self.warn("Please enter a numeric value.")
            return self.prompt_float(message, default=default)

    def prompt_choice(self, message: str, options: Sequence[str]) -> int:
        if not options:
            raise ValueError("No options provided.")
        while True:
            choice = self.prompt(message)
            if not choice:
                continue
            if choice.isdigit():
                index = int(choice)
                if 1 <= index <= len(options):
                    return index - 1
            self.warn(f"Select a value between 1 and {len(options)}.")

    def confirm(self, message: str, *, default: bool = False) -> bool:
        suffix = "[Y/n]" if default else "[y/N]"
        text = self.prompt(f"{message} {suffix}", default="y" if default else "n").strip().lower()
        if not text:
            return default
        return text in {"y", "yes"}

    def print(self, message: str = "") -> None:
        raise NotImplementedError

    def warn(self, message: str) -> None:
        self.print(f"WARNING: {message}")

    def info(self, message: str) -> None:
        self.print(message)


@dataclass
class CliIO(PostProcessIO):
    """Simple console IO adapter."""

    scripted_choices: Optional[Sequence[str]] = None
    _script_index: int = 0

    def prompt(self, message: str, *, default: str | None = None) -> str:
        prompt_text = f"{message.strip()} "
        if default not in (None, ""):
            prompt_text = f"{message.strip()} [{default}] "

        if self.scripted_choices is not None:
            if self._script_index >= len(self.scripted_choices):
                raise RuntimeError("Scripted input exhausted.")
            response = self.scripted_choices[self._script_index]
            self._script_index += 1
            self.print(f"{prompt_text}{response}")
            return response

        try:
            response = input(prompt_text)
        except EOFError:
            response = ""
        if not response and default is not None:
            return default
        return response

    def print(self, message: str = "") -> None:
        print(message)


@dataclass
class GuiIO(PostProcessIO):
    """Adapter used by the console-based GUI facade."""

    input_fn: Optional[callable] = None
    output_fn: Optional[callable] = None

    def prompt(self, message: str, *, default: str | None = None) -> str:
        fn = self.input_fn or input
        prompt_text = f"{message.strip()} "
        if default not in (None, ""):
            prompt_text = f"{message.strip()} [{default}] "
        text = fn(prompt_text)
        if not text and default is not None:
            return default
        return text

    def print(self, message: str = "") -> None:
        fn = self.output_fn or print
        fn(message)
