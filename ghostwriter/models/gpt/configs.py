from dataclasses import dataclass
from typing import Optional

PRETRAINED_DEFAULT_BLOCK_SIZE = 1024
PRETRAINED_DEFAULT_VOCAB_SIZE = 50257


@dataclass
class GPTConfig:
    block_size: int = PRETRAINED_DEFAULT_BLOCK_SIZE
    vocab_size: int = PRETRAINED_DEFAULT_VOCAB_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


@dataclass
class GPTGenerationConfig:
    max_new_tokens: int = 100
    temperature: int = 0.8
    top_k: int = 200


DEFAULT_GPT = GPTConfig()

GPT_CONFIGS = {"default": DEFAULT_GPT}

# All pretrained checkpoints require a 1024 transformer block size
# Use crop_block_size on the model to shrink later.
PRETRAINED_CONFIGS = dict(
    {
        # 124 Million Parameter Model - Base Model
        "gpt2": GPTConfig(
            PRETRAINED_DEFAULT_BLOCK_SIZE,
            PRETRAINED_DEFAULT_VOCAB_SIZE,
            12,
            12,
            768,
            0.1,
        ),
        # 250 Million Parameter Model
        "gpt2-medium": GPTConfig(
            PRETRAINED_DEFAULT_BLOCK_SIZE,
            PRETRAINED_DEFAULT_VOCAB_SIZE,
            24,
            16,
            1024,
            0.1,
        ),
        # 774 Million Parameter Model
        "gpt2-large": GPTConfig(
            PRETRAINED_DEFAULT_BLOCK_SIZE,
            PRETRAINED_DEFAULT_VOCAB_SIZE,
            36,
            20,
            1280,
            0.1,
        ),
        # 1.558 Billion Parameter Model
        "gpt2-xl": GPTConfig(
            PRETRAINED_DEFAULT_BLOCK_SIZE,
            PRETRAINED_DEFAULT_VOCAB_SIZE,
            48,
            25,
            1600,
            0.1,
        ),
    }
)


def get_gpt_config_from_arg(config_name: str) -> GPTConfig:
    if config_name not in GPT_CONFIGS:
        raise ValueError("Invalid config for GPT model")
    return GPT_CONFIGS[config_name]


def get_config(pretrained: bool = True, config_name: Optional[str] = None):
    if pretrained:
        if config_name is None:
            return PRETRAINED_CONFIGS["gpt2"]

        if config_name is not None and config_name not in PRETRAINED_CONFIGS.keys():
            raise ValueError(
                f"No pretrained config exists for this name.\n Options are: {PRETRAINED_CONFIGS.keys()}"
            )
        else:
            return PRETRAINED_CONFIGS[config_name]

    else:
        if config_name is None:
            return PRETRAINED_CONFIGS["gpt2"]

        if config_name is not None and config_name not in GPT_CONFIGS.keys():
            raise ValueError(
                f"No config exists for this name.\n Options are: {GPT_CONFIGS.keys()}"
            )
        else:
            return GPT_CONFIGS[config_name]
