"""
GPT2 Language Model using references from:
https://github.com/openai/gpt-2/
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2
"""


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import LongTensor, Tensor

from ghostwriter.models.gpt.block import GPT2TransformerBlock
from ghostwriter.models.gpt.configs import (
    GPTConfig,
    GPTGenerationConfig,
    PRETRAINED_CONFIGS,
)

from typing import Literal, Optional, Tuple

from logging import Logger
import logging

logger = Logger(__name__, logging.DEBUG)


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.embedding_dimension = config.n_embd
        self.num_heads = config.n_head
        self.block_size = config.block_size
        self.dropout_ratio = config.dropout
        self.num_layers = config.n_layer

        # Layer names need to match original implementation to enable loading existing weights
        # Transformer body is composed of token and positional embeddings
        # dropout and a sequence of transformer blocks (decoder only in this case)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.vocab_size, self.embedding_dimension),
                wpe=nn.Embedding(self.block_size, self.embedding_dimension),
                drop=nn.Dropout(self.dropout_ratio),
                h=nn.ModuleList(
                    [
                        GPT2TransformerBlock(
                            self.embedding_dimension,
                            self.num_heads,
                            self.block_size,
                            self.dropout_ratio,
                        )
                        for _ in range(self.num_layers)
                    ]
                ),
                ln_f=nn.LayerNorm(self.embedding_dimension),
            )
        )
        # Finally a output layer
        self.lm_head = nn.Linear(self.embedding_dimension, self.vocab_size, bias=False)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, sequence: LongTensor, ground_truth: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the model - takes a sequence and outputs the logits of the next token

        Parameters
        ----------
        sequence
            The input sequence used to predict the next token
        ground_truth
            The true next token used to train the model to predict the next token

        Returns
        -------
        Returns a Tuple[Tensor, Tensor] containing the logits and optionally the loss
        """
        _, sequence_length = sequence.size()

        assert (
            sequence_length <= self.block_size
        ), f"Cannot forward sequence of length {sequence_length}, block size is only {self.block_size}"

        positions = torch.arange(
            0, sequence_length, dtype=torch.long, device=sequence.device
        ).unsqueeze(0)

        # Get the token and positional embeddings for the input sequence
        # Token embeddings (Batch Size, Sequence Length, Embedding Dimension)
        token_embeddings = self.transformer.wte(sequence)
        # Positional Embeddings (1, Sequence Length, Embedding Dimension)
        positional_embeddings = self.transformer.wpe(positions)

        # Forward through transformer blocks
        x = self.transformer.drop(token_embeddings + positional_embeddings)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        # If labels exist then calculate the loss
        loss = None
        if ground_truth is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), ground_truth.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self, sequence: LongTensor, generation_config: GPTGenerationConfig
    ) -> LongTensor:
        """
        Complete a sequence using a conditioning of indicies by continually feeding the results
        from the previous prediction into the model for the next prediction.

        Parameters
        ----------
        sequence
            The encoded input sequence
        generation_config
            Generation options

        Returns
        -------
        A tensor of size (Sequence Length) with the token indexes for the generated text

        """
        for _ in range(generation_config.max_new_tokens):

            # Ensure the sequence is not longer than the size of the transformer block
            sequence_length = sequence.size(1)
            sequence_cond = (
                sequence
                if sequence_length <= self.block_size
                else sequence[:, -self.block_size :]
            )

            # Forward pass to get logits for next prediction in the sequence
            logits, _ = self(sequence_cond)

            # At the final step of we take the logits and scale by temperature
            # to alter the variance in predicted tokens
            logits = logits[:, -1, :] / generation_config.temperature

            # Top k limits low probability options by cropping the logits
            if generation_config.top_k is not None:
                v, _ = torch.topk(logits, min(generation_config.top_k, logits.size(-1)))
                # Mask all values outside the top-k
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Logits -> normalzied probabilites
            # Then sample to get the next prediction
            # and add to the sequence as the next token
            probs = F.softmax(logits, dim=-1)
            sequence_next = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat((sequence, sequence_next), dim=1)

        return sequence

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        Only certain layers need to experience weight decay so some configuration is needed with parameter
        groups for AdamW.

        Parameters
        ----------
        weight_decay
            The rate to decay the weights
        learning_rate
            The step size to modify the weights as the optimizer updates the backwards pass
        betas
            Coefficients used for computing running averages of gradient and its square

        Returns
        -------
        An instance of the AdamW optimizer configured correctly for weight decay
        """

        layers_with_decay = set()
        layers_without_decay = set()

        # Iterate through all modules and child parameters and check their type
        # Linear layers experience weight decay
        # Layer norms and embeddings should never experience weight decay
        for module_name, module in self.named_modules():
            for parameter_name, _ in module.named_parameters():
                full_parameter_name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )

                if parameter_name.endswith("bias"):
                    # Never decay bias terms
                    layers_without_decay.add(full_parameter_name)

                if parameter_name.endswith("weight"):
                    if isinstance(module, nn.Linear):
                        layers_with_decay.add(full_parameter_name)
                    elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
                        layers_without_decay.add(full_parameter_name)

        parameters = {pn: p for pn, p in self.named_parameters()}
        optimizer_groups = [
            {
                "params": [
                    parameters[parameter_name]
                    for parameter_name in sorted(list(layers_with_decay))
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    parameters[parameter_name]
                    for parameter_name in sorted(list(layers_without_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate, betas=betas)
        return optimizer

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        dropout: Optional[float] = None,
    ):
        """
        Loads a pretrained checkpoint from the transformers library using the original
        OpenAI gpt2 weights. Useful for finetuning models without training from scratch.

        Parameters
        ----------
        checkpoint
            The checkpoint to load, the current supported checkpoints are:
                - gpt2
                - gpt2-medium
                - gpt2-large
                - gpt2-xl
        dropout
            The dropout ratio to use in the pretrained model the default is 0.1
            but this may need to be different with fine tuning.

        Returns
        -------
        A "GPT" object (itself).

        """
        assert checkpoint in {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        }, "Invalid checkpoint name provided"

        from transformers import GPT2LMHeadModel

        logger.debug(f"Loading {checkpoint} checkpoint from HuggingFace")
        config = PRETRAINED_CONFIGS[checkpoint]

        if dropout is not None:
            # Override the default dropout for the pretrained checkpoint
            config.dropout = dropout

        print(config)

        model = GPT(config)
        state_dict = model.state_dict()

        # Load the weights from transformers
        huggingface_model = GPT2LMHeadModel.from_pretrained(checkpoint)
        pretrained_state_dict = huggingface_model.state_dict()

        # Need to copy all of the parameters from the huggingface model to the local model
        # this is why the variable names must match (see constructor)
        # Have to do some augmentation to handle some choices that do not really make sense to use here
        # For example:
        # The OpenAI implementation uses one dimensional convolutions instead of linear layers
        # for fully connected layers. Will have to adjust this.
        # Also, do not care about the maksed_bias on the attention layers
        layer_names = [layer_name for layer_name in pretrained_state_dict]
        layers_to_augment = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        for layer_name in layer_names:
            if layer_name.endswith("attn.masked_bias"):
                # Skip masking
                continue

            if any(layer_name.endswith(w) for w in layers_to_augment):
                # The aforementioned 1D convolutions need to be transposed to act as a linear layer
                assert (
                    pretrained_state_dict[layer_name].shape[::-1]
                    == state_dict[layer_name].shape
                ), "Shape mismatch when transposing pretrained convolutions"

                with torch.no_grad():
                    state_dict[layer_name].copy_(pretrained_state_dict[layer_name].t())
            else:
                assert (
                    pretrained_state_dict[layer_name].shape
                    == state_dict[layer_name].shape
                )
                with torch.no_grad():
                    state_dict[layer_name].copy_(pretrained_state_dict[layer_name])

        return model

    def crop_block_size(self, block_size: int):
        """
        Taken from karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
        this enables loading a larger model checkpoint's weights and then shrinking
        the size of the transformer blocks. This is useful for fine tuning on small
        datasets and simpler objectives.

        Parameters
        ----------
        block_size
            The new block size to shrink to
        """
        assert block_size <= self.block_size, "Cannot crop to a larger block size"
        self.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
