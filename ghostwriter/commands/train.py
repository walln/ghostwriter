"""CLI command to train a model."""

import os
from typing import Literal, Optional

import click
import torch

from ghostwriter.data.dataset import create_or_load_dataset
from ghostwriter.models.gpt.configs import get_config
from ghostwriter.models.gpt.model import GPT
from ghostwriter.models.gpt.trainer import GPTTrainer
from ghostwriter.utils import set_seed_and_backends

set_seed_and_backends()

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"


@click.command(help="Train models from scratch or fine tune pretrained weights")
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
    "--batch_size", "-b", default=4, help="Batch size to use to train the model"
)
@click.option(
    "--grad-clip",
    "-gc",
    default=1.0,
    type=float,
    help="Value to use for gradient clipping",
)
@click.option(
    "--learning_rate",
    "-lr",
    default=6e-4,
    type=float,
    help="Initial learning rate to use",
)
@click.option(
    "--max_iters",
    "-i",
    default=1000,
    type=int,
    help="Maximimum number of iterations to train for",
)
@click.option(
    "--weight-decay", "-wd", default=1e-2, type=float, help="Weight decay rate"
)
@click.option(
    "--clip_block_size",
    "-cbs",
    type=int,
    default=None,
    help="Smaller transformer block size to clip to",
)
def train(
    artist: str,
    model: Literal["gpt"],
    pretrained: bool,
    config: Optional[str],
    batch_size: int,
    grad_clip: float,
    learning_rate: float,
    weight_decay: float,
    max_iters: int,
    clip_block_size: Optional[int],
):
    """
    CLI Command to train a model.

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
    batch_size
        The batch size to use during training
    grad_clip
        The gradient clipping value to use
    learning_rate
        The eta size to use for AdamW
    weight_decay
        The amount to decay weights during training
    max_iters
        The maximum number of iterations to train the model for
    clip_block_size
        The size of the block to shrink to if needed

    """
    beta1 = 0.9
    beta2 = 0.95

    print("Training a model...")

    if model == "gpt":
        model_config = get_config(pretrained, config)
        print("Config: ", model_config)

        train_data, val_data, _ = create_or_load_dataset(artist, "GPT", pretrained)

        if pretrained:
            gpt = GPT.from_pretrained(config if config is not None else "gpt2")
        else:
            gpt = GPT(model_config)

        out_dir = os.path.join(
            "checkpoints",
            model,
            config if config is not None else "gpt2",
            artist,
            "pretrained" if pretrained else "scratch",
        )

        if clip_block_size is not None:
            gpt.crop_block_size(clip_block_size)

        optimizer = gpt.configure_optimizers(
            weight_decay, learning_rate, (beta1, beta2)
        )

        trainer = GPTTrainer(
            gpt,
            max_iters=max_iters,
            batch_size=batch_size,
            grad_clip=grad_clip,
        )
        trainer.fit(optimizer, train_data, val_data)
        trainer.save_model(out_dir)

    else:
        raise ValueError("No model exists with that name")
