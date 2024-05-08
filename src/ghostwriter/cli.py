"""Ghostwriter CLI."""

import typer
from ghostwriter.commands.generate_dataset import generate_dataset_command
from ghostwriter.model.generate_lyrics import generate_lyrics_command
from ghostwriter.model.train import train_qlora_model_command
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def main(name: str):
    """CLI app to enable training and interacting with models and datasets."""
    typer.echo(f"Hello {name}")


@app.command()
def generate_dataset(
    artist: Annotated[str, "The name of the artist to generate a dataset for."],
):
    """Generate dataset from artist lyrics."""
    generate_dataset_command(artist)


@app.command()
def train_model():
    """Train a model."""
    train_qlora_model_command()


@app.command()
def generate_lyrics():
    """Evaluate the model."""
    generate_lyrics_command()


if __name__ == "__main__":
    app()
