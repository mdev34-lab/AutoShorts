import typer

from .console import PlainGroup

new_app = typer.Typer(help="Create a new video (explainer, etc.)", cls=PlainGroup)
