"""Generate a dataset for a given artist."""

import typer
from datasets import Dataset
from ghostwriter.dataset.common import DATASET_OUTPUT_DIR
from ghostwriter.dataset.scrape_lyrics import (
    _get_artist_information,
    _get_lyrics_for_songs,
    _get_songs_for_artist,
)


def generate_dataset_command(artist: str):
    """Generate a dataset from an artist lyrics.

    Args:
      artist: The name of the artist to search lyrics for.
    """
    # TODO: Check if a dataset already exists and warn the user it will be overwritten
    typer.echo(f"Generating dataset for artist: {artist}")

    # First locate the artist id
    artist_info = _get_artist_information(artist)
    songs = _get_songs_for_artist(artist_info["artist_id"])
    typer.echo(f"Found {len(songs)} song(s) for {artist_info['artist_name']}.")
    _get_lyrics_for_songs(songs)

    dataset = Dataset.from_generator(
        _get_lyrics_for_songs, gen_kwargs={"songs": songs}, num_proc=16
    )
    assert isinstance(dataset, Dataset)

    new_col = [artist_info["artist_name"]] * len(dataset)
    dataset = dataset.add_column(name="artist_name", column=new_col)  # type: ignore

    # Remove all errors from the dataset
    dataset = dataset.filter(
        lambda example: example["lyrics"] is not None and example["error"] == "NONE"
    )

    dataset._info.description = f"Lyrics dataset for {artist_info["artist_name"]}"
    dataset._info.dataset_name = f"{artist_info["artist_name"]}_lyrics"
    dataset.save_to_disk(DATASET_OUTPUT_DIR)

    typer.echo(f"Dataset saved to {DATASET_OUTPUT_DIR}")

    info = dataset.info
    typer.echo(f"Dataset name: {info.dataset_name}")
    typer.echo(f"Dataset description: {info.description}")
    typer.echo(f"Dataset size: {len(dataset)}")
