from contextlib import nullcontext

import torch

from ghostwriter.models.gpt.configs import GPTGenerationConfig
from ghostwriter.models.gpt.model import GPT


class GPTGenerator:
    """
    Generator utility class to handle encoding and decoding inputs and outputs,
    as well as managing evaluation state, model devices and and autograd context

    Parameters
    ----------
    model
        The GPT model instance to use to generate new tokens
    tokenizer
        The tokenizer instance to encode and decode tokens
    device
        The device to use for the model
    """

    def __init__(self, model: GPT, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.device_type = "cuda" if "cuda" in self.device else "cpu"

        self.model.eval()
        self.model.to(self.device)

    def generate(self, input: str, generation_config: GPTGenerationConfig) -> str:
        """
        Encode an input string and use model.generate() to complete the input

        Parameters
        ----------
        input
            A string to use to start the generation
        generation_config
            Generation options to pass to the model

        Returns
        -------
        A string including the input and the predicted sequence of tokens
        """
        with torch.no_grad():
            ctx = (
                nullcontext()
                if self.device_type == "cpu"
                else torch.amp.autocast(
                    device_type=self.device_type, dtype=torch.bfloat16
                )
            )

            tokens = self.tokenizer.encode(input)
            x = torch.tensor(tokens, dtype=torch.long, device=self.device)[None, ...]

            with ctx:
                y = self.model.generate(x, generation_config)
                return self.tokenizer.decode(y[0].tolist())
