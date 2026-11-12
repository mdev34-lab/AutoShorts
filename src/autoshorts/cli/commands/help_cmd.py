from collections.abc import Mapping
from typing import Any

import click
import typer
from typer.main import get_command


def _typer_context_class() -> Any | None:
    """Return the Context class of typer's own click, when it ships one.

    Typer >= 0.27 vendors its own copy of click, so the commands it builds
    are not ``click.Command`` instances and need the matching Context class.
    """
    try:
        from typer import _click as typer_click
    except ImportError:  # pragma: no cover - older typer uses click directly
        return None
    return getattr(typer_click, "Context", None)


def _make_help_context(command: Any, info_name: str) -> Any:
    """Build the context used to render the help of ``command``."""
    context_class: Any = click.Context
    if not isinstance(command, click.Command):
        vendored_class = _typer_context_class()
        if vendored_class is not None:
            context_class = vendored_class
    return context_class(command, info_name=info_name)


def _resolve_command(root_cmd: Any, path: list[str]) -> Any | None:
    """Resolve a command path such as ``["new", "explainer"]``.

    Returns the command the path points at, or None when the path does not
    exist. Groups are detected by their ``commands`` mapping, because typer
    builds its own group type that is not a ``click.Group``.
    """
    current = root_cmd
    for name in path:
        commands = getattr(current, "commands", None)
        if not isinstance(commands, Mapping) or name not in commands:
            return None
        current = commands[name]
    return current


def make_help_command(app: typer.Typer):
    def help_command(
        args: list[str] = typer.Argument(None, help="Command path to show help for"),
    ):
        # Typer's command object is consumed here through click's Command/Group API.
        root_cmd = get_command(app)
        path = list(args or [])
        target = _resolve_command(root_cmd, path)

        if target is None:
            typer.secho(
                f"Error: No such command: autoshorts {' '.join(path)}",
                fg="red",
            )
            raise typer.Exit(1)

        help_ctx = _make_help_context(target, " ".join(["autoshorts", *path]))
        typer.echo(target.get_help(help_ctx))
        raise typer.Exit(0)

    return help_command
