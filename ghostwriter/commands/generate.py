"""CLI command to generate text using a trained model."""
import os
from typing import Literal, Optional

import click
import torch

from ghostwriter.data.dataset import create_or_load_dataset
from ghostwriter.models.gpt.configs import GPTGenerationConfig, get_config
from ghostwriter.models.gpt.generator import GPTGenerator
from ghostwriter.models.gpt.model import GPT
from ghostwriter.utils import set_seed_and_backends

set_seed_and_backends()

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"


@click.command(help="Load a trained model and generate text")
@click.option(
    "--artist",
    "-a",
    default="Drake",
    type=str,
    help="The name of the artist whose discography should be used as a dataset",
)
@click.option(
    "--model",
    "-m",
    default="gpt",
    type=click.Choice(["gpt"]),
    help="The model architecture to train",
)
@click.option(
    "--pretrained",
    "-p",
    default=True,
    help="Use pretrained weights and fine-tune?",
)
@click.option(
    "--config",
    "-c",
    default=None,
    help="Model configuration name to use",
)
@click.option(
    "--clip_block_size",
    "-cbs",
    default=None,
    type=int,
    help="Smaller transformer block size to clip to",
)
def generate(
    artist: str,
    model: Literal["gpt"],
    pretrained: bool,
    config: Optional[str],
    clip_block_size: Optional[int],
):
    """
    CLI Command to generate text using a trained model.

    Parameters
    ----------
    artist
        The artist to use as a dataset
    model
        The model architecture to use
    pretrained
        If pretrained weights should be used
    config
        The model config to use
    clip_block_size
        The size of the block to shrink to if needed

    """
    checkpoints_dir = os.path.join("checkpoints", model)
    if model == "gpt":
        path = os.path.join(
            checkpoints_dir,
            "gpt2" if config is None else config,
            artist,
            "pretrained" if pretrained else "scratch",
        )

        model_config = get_config(pretrained, config)
        if pretrained:
            gpt = GPT.from_pretrained(config if config is not None else "gpt2")
        else:
            gpt = GPT(model_config)

        if clip_block_size is not None:
            gpt.crop_block_size(clip_block_size)

        gpt.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        gpt.eval()

        _, _, tokenizer = create_or_load_dataset(artist, "GPT", pretrained)

        generator = GPTGenerator(gpt, tokenizer)
        generation_config = GPTGenerationConfig()

        try:
            print("---------------------------------------------------")
            while True:
                user_input = input(
                    "Prompt the model with a few words to start the song: "
                )
                result = generator.generate(user_input, generation_config)

                print("Generated: ", result)
                print("---------------------------------------------------")
        except KeyboardInterrupt:
            print("\n---------------------------------------------------")
            print("Shutting down")

    else:
        raise ValueError("No model exists with this name")
