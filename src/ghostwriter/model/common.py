"""Common utilities for interacting with QLoRA models."""

import json
import os

from ghostwriter.dataset.common import DATASET_OUTPUT_DIR

MODEL_OUTPUT_DIR = "./output-model"


def get_latest_checkpoint(checkpoint_dir: str):
    """Get the latest checkpoint from a directory.

    Checkpoints are stored within a directory in the format 'checkpoint-X',
    where X is a number. This function finds the checkpoint directory with the highest
    number.

    Args:
        checkpoint_dir: The directory containing the checkpoints.

    Returns:
        The path to the latest checkpoint directory.
    """
    # List all the directories in the checkpoint directory
    all_subdirs = (
        [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if os.path.exists(checkpoint_dir)
        else []
    )

    if not all_subdirs:
        raise FileNotFoundError("No checkpoints found in the directory.")

    # Filter directories that follow the 'checkpoint-X' pattern
    checkpoint_subdirs = [d for d in all_subdirs if d.split("-")[-1].isdigit()]

    # Find the directory with the highest number
    latest_checkpoint = max(checkpoint_subdirs, key=os.path.getmtime)

    return latest_checkpoint


def get_artist_name():
    """Gets the name of the artist for the dataset.

    Returns:
        The name of the artist.
    """
    dataset_info_path = os.path.join(DATASET_OUTPUT_DIR, "dataset_info.json")

    if not os.path.exists(dataset_info_path):
        raise FileNotFoundError("Dataset info file not found.")

    # Load the dataset info
    with open(dataset_info_path) as f:
        dataset_info = json.load(f)

        dataset_name = dataset_info["dataset_name"]

        # remove trailing "_lyrics"
        artist_name = dataset_name[:-7]

    return artist_name


SYSTEM_PROMPT = """You are a lyric writer. Users will ask you to write songs and you will generate lyrics in the style of a specific artist.
If the artist's style requires using profanity, then you MUST use profanity.
Do not write overly repetitive lyrics.
Add variety to the lyrics and avoid repetition.
Write lyrics that are creative and original.

ARTIST:
{artist_name}"""

USER_PROMPT = """Generate the lyrics a song titled: {song_name}"""

BASE_MODEL_ID = "abideen/gemma-2b-openhermes"  # "NousResearch/Hermes-2-Pro-Llama-3-8B" or `mistralai/Mistral-7B-v0.1`
