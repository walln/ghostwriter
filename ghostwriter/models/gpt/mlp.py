"""GPT multi-layer perceptron implementation as outlined by OpenAI."""
import torch.nn as nn
from torch import Tensor
import torch
import math

# Activations taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in intial BERT repo.
    For information: OpenAI GPT's GELU is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Parameters
    ----------
    use_gelu_python
        Whether to use the python or torch implementation of GELU

    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        """
        Python implementation of the GELU activation.

        Parameters
        ----------
        input
            The input tensor to apply the activation functio to

        Returns
        -------
        The tensor with the applied activation function.

        """
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for the GELU activation.

        Parameters
        ----------
        input
            The input to apply the activation function

        Returns
        -------
        The tensor with the applied activation function.

        """
        return self.act(input)


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo.
    This implementation is identical to OpenAI GPT.
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for the GELU activation in BERT.

        Parameters
        ----------
        input
            The input to apply the activation function

        Returns
        -------
        The tensor with the applied activation function.

        """
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


class GPT2MLP(nn.Module):
    """
    GPT2 Multi-layer perceptron as implemented by OpenAI.

    Parameters
    ----------
    embedding_dimension
        The size of the embedding to use
    dropout_ratio
        The rate to dropout weights during training

    """

    def __init__(self, embedding_dimension: int, dropout_ratio: float):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.dropout_ratio = dropout_ratio

        # GPT2 paper references hidden layer size of 4x embedding size
        self.c_fc = nn.Linear(self.embedding_dimension, 4 * self.embedding_dimension)
        self.c_proj = nn.Linear(4 * self.embedding_dimension, self.embedding_dimension)
        self.activation = NewGELUActivation()
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x: Tensor):
        """
        Forward pass for the MLP.

        Parameters
        ----------
        x
            The input tensor

        Returns
        -------
        The result of the forward pass through the MLP.

        """
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
