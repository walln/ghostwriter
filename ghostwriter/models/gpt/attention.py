"""Causal Self Attention for Decoder Only Tranformer blocks as outlined in GPT2."""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPT2Attention(nn.Module):
    """
    GPT2 implementation of decoder only causal self attention.

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

        self.n_embd = embedding_dimension
        self.n_head = num_heads
        self.block_size = block_size
        self.dropout_ratio = dropout_ratio

        assert (
            self.n_embd % self.n_head == 0
        ), "Embedding must evenly divide into multiple heads"

        # Batchwise projection of k, q, v
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

        self.attn_dropout = nn.Dropout(self.dropout_ratio)
        self.resid_dropout = nn.Dropout(self.dropout_ratio)

        # Mask applied to keep attention to only already "seen" tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(
                1, 1, self.block_size, self.block_size
            ),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass to compute attention for the input tensor.

        Parameters
        ----------
        x
            The input tensor to compute attention for

        Returns
        -------
        An output projection for the input tensor.

        """
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding size

        # Compute queries, keys, and values for every head
        # head -> batch dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (Batch Size, Num Heads, Sequence Length, Head Size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (Batch Size, Num Heads, Sequence Length, Head Size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (Batch Size, Num Heads, Sequence Length, Head Size)

        # Self attention (causal) requires the positional attention multiplied by
        # queries @ keys (requires shape augmentation)
        # (Batch Size, Num Heads, Sequence Length, Head Size) x
        # (Batch Size, Num Heads, Head Size, Sequence Length)
        # -> (Batch Size, Num Heads, Sequence Length, Sequence Length)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        #  (Batch Size, Num Heads, Sequence Length, Sequence Length) X
        # (Batch Size, Num Heads, Sequence Length, Head Size)
        # -> (Batch Size, Num Heads, Sequence Length, Head Size)
        y = att @ v

        # Take head outputs and reshape
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
