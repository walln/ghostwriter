"""GPT2 Tranformer decoder block as outlined by OpenAI."""

import torch.nn as nn
from torch import Tensor

from ghostwriter.models.gpt.attention import GPT2Attention
from ghostwriter.models.gpt.mlp import GPT2MLP


class GPT2TransformerBlock(nn.Module):
    """
    GPT decoder transformer block as outlined by OpenAI.

    Parameters
    ----------
    embedding_dimension
        The size of the embedding dimension
    num_heads
        The number of attention heads to use
    block_size
        The size of each transformer block
    dropout_ratio
        The ratio of weights to dropout during training

    """

    def __init__(
        self,
        embedding_dimension: int,
        num_heads: int,
        block_size: int,
        dropout_ratio: float,
    ):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout_ratio = dropout_ratio

        self.ln_1 = nn.LayerNorm(self.embedding_dimension)
        self.ln_2 = nn.LayerNorm(self.embedding_dimension)
        self.attn = GPT2Attention(
            self.embedding_dimension,
            self.num_heads,
            self.block_size,
            self.dropout_ratio,
        )
        self.mlp = GPT2MLP(self.embedding_dimension, self.dropout_ratio)

    def forward(self, x: Tensor):
        """
        Forward pass for the transformer decoder block.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
        The result of the forward pass through the decoder block.

        """
        pre_norm = self.ln_1(x)
        x = x + self.attn(pre_norm)
        post_norm = self.ln_2(x)
        x = x + self.mlp(post_norm)
        return x
