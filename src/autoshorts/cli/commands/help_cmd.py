import click
import typer
from typer.main import get_command


def make_help_command(app: typer.Typer):
    def help_command(
        args: list[str] = typer.Argument(None, help="Command path to show help for"),
    ):
        root_cmd = get_command(app)
        target = root_cmd
        info_parts = ["autoshorts"]

        if args:
            for name in args:
                if isinstance(target, click.Group) and name in target.commands:
                    target = target.commands[name]
                    info_parts.append(name)
                else:
                    typer.secho(
                        f"Error: No such command: autoshorts {' '.join(args)}",
                        fg="red",
                    )
                    raise typer.Exit(1)

        help_ctx = click.Context(target, info_name=" ".join(info_parts))
        typer.echo(target.get_help(help_ctx))
        raise typer.Exit(0)

    return help_command
