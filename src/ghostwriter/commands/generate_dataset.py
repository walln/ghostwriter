"""Generate a dataset for a given artist."""

from datasets import Dataset
from ghostwriter.dataset.common import DATASET_OUTPUT_DIR
from ghostwriter.dataset.scrape_lyrics import (
    get_artist_information,
    get_lyrics_for_songs,
    get_songs_for_artist,
)
from ghostwriter.model.common import get_artist_name
from rich.console import Console
from rich.prompt import Confirm


def generate_dataset_command(artist: str):
    """Generate a dataset from an artist lyrics.

    Args:
      artist: The name of the artist to search lyrics for.
    """
    console = Console()
    console.print(
        "This command will generate a dataset from the lyrics of a given artist."
    )

    # Check if a dataset already exists
    try:
        artist_name = get_artist_name()
    except FileNotFoundError:
        artist_name = None

    if artist_name:
        overwrite = Confirm.ask(
            f"A dataset for {artist_name} already exists. Do you want to overwrite it?"
        )
        if not overwrite:
            console.print("Exiting.")
            return

    console.print(f"Generating dataset for artist: {artist}")

    # First locate the artist id
    artist_info = get_artist_information(artist)
    songs = get_songs_for_artist(artist_info["artist_id"])
    console.print(f"Found {len(songs)} song(s) for {artist_info['artist_name']}.")
    get_lyrics_for_songs(songs)

    dataset = Dataset.from_generator(
        get_lyrics_for_songs, gen_kwargs={"songs": songs}, num_proc=16
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

    console.print(f"Dataset saved to {DATASET_OUTPUT_DIR}")

    info = dataset.info
    console.print(f"Dataset name: {info.dataset_name}")
    console.print(f"Dataset description: {info.description}")
    console.print(f"Dataset size: {len(dataset)}")
