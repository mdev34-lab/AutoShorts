"""Console output policy for the CLI.

Typer draws the help screens with rich, which keeps its colours even when
the output is piped: rich is told to force a terminal so CI logs are
coloured. A pipe is not a terminal though, so the help is rendered as plain
text there - the very same text, without the escape sequences, that a
terminal-less run always produced. Terminals keep their colours.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import typer.rich_utils
from typer.core import TyperCommand, TyperGroup


def stdout_is_terminal() -> bool:
    """Whether stdout is an interactive terminal."""
    stream = sys.stdout
    try:
        return bool(stream is not None and stream.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - closed stream
        return False


@contextmanager
def plain_output() -> Iterator[None]:
    """Draw the current output without colours when stdout is not a terminal.

    Rich's colour system is what turns the help screens into coloured
    panels; disabling it keeps every character of the text and only drops
    the escape sequences.
    """
    if stdout_is_terminal():
        yield
        return

    color_system = getattr(typer.rich_utils, "COLOR_SYSTEM", None)
    typer.rich_utils.COLOR_SYSTEM = None
    try:
        yield
    finally:
        typer.rich_utils.COLOR_SYSTEM = color_system


class PlainCommand(TyperCommand):
    """A command whose help is plain text when stdout is not a terminal."""

    def format_help(self, ctx: Any, formatter: Any) -> None:
        with plain_output():
            super().format_help(ctx, formatter)


class PlainGroup(TyperGroup):
    """A group whose help is plain text when stdout is not a terminal."""

    def format_help(self, ctx: Any, formatter: Any) -> None:
        with plain_output():
            super().format_help(ctx, formatter)
