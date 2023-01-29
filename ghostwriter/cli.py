"""CLI app to enable training and interacting with models and datasets."""

import click
from ghostwriter.commands import train
from ghostwriter.commands import generate


@click.group(help="CLI tool to interface with ghostwriter.")
def cli():
    """Entrypoint for the cli app."""
    pass


cli.add_command(train.train)
cli.add_command(generate.generate)


if __name__ == "__main__":
    cli()
