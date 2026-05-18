import typer

from .commands import explainer as explainer
from .commands.help_cmd import make_help_command
from .new import new_app

app = typer.Typer(
    name="autoshorts",
    help="Automated video generation tool",
    no_args_is_help=True,
)
app.add_typer(new_app, name="new")

app.command(name="help")(make_help_command(app))
