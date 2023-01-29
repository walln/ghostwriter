import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim.adamw import AdamW
from tqdm import trange

from ghostwriter.models.gpt.model import GPT


class GPTTrainer:
    """
    Trainer handles the training loop and saving checkpoints of the trained model class

    Parameters
    ----------
    model
        The model instance to train
    device
        The device to train on
    max_iters
        The maximum number of iterations to train the model
    batch_size
        The amount of parallel batches to use when training the model
    grad_clip
        The level for gradient clipping
    """

    def __init__(
        self,
        model: GPT,
        device: str = "auto",
        max_iters: int = 1000,
        batch_size: int = 4,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = None
        self.callbacks = defaultdict(list)
        self.max_iters = max_iters
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.iter_num = 0

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

    def fit(self, optimizer: AdamW, train_data, val_data):
        """
        Fits the model to attempt to predict the next token in a sequence

        Parameters
        ----------
        optimizer
            The optimizer to use to improve the model
        train_data
            The training dataset split
        val_data
            The validation dataset split
        """
        model = self.model
        self.optimizer = optimizer

        def get_batch(split):
            data = train_data if split == "train" else val_data
            indices = torch.randint(len(data) - model.block_size, (self.batch_size,))
            x = torch.stack(
                [
                    torch.from_numpy(
                        (data[index : index + model.block_size]).astype(np.int64)
                    )
                    for index in indices
                ]
            )
            y = torch.stack(
                [
                    torch.from_numpy(
                        (data[index + 1 : index + 1 + model.block_size]).astype(
                            np.int64
                        )
                    )
                    for index in indices
                ]
            )
            x, y = x.to(self.device), y.to(self.device)
            return x, y

        model.train()
        self.iter_num = 0

        with trange(self.max_iters, unit="iters") as pbar:
            for _ in pbar:
                x, y = get_batch("train")

                _, self.loss = model(x, y)

                self.optimizer.zero_grad()
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()

                pbar.set_postfix_str(f"Loss: {self.loss.item():.4f}")

                self.iter_num += 1

                if self.max_iters is not None and self.iter_num >= self.max_iters:
                    pbar.update(self.max_iters - pbar.n)
                    break

    def save_model(self, path: str):
        """
        Save the model's state dictionary to a file

        Parameters
        ----------
        path
            The path to the directory where the checkpoint should be saved
        """

        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
