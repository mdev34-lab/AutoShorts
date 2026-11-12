import typer

from .commands import explainer as explainer
from .commands import product as product
from .commands.help_cmd import make_help_command
from .console import PlainCommand, PlainGroup
from .new import new_app

app = typer.Typer(
    name="autoshorts",
    help="Automated video generation tool",
    no_args_is_help=True,
    cls=PlainGroup,
)
app.add_typer(new_app, name="new")

app.command(name="help", cls=PlainCommand)(make_help_command(app))
