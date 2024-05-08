"""QLoRA training script."""

import torch
from datasets import Dataset
from ghostwriter.dataset.common import DATASET_OUTPUT_DIR
from ghostwriter.model.common import (
    BASE_MODEL_ID,
    MODEL_OUTPUT_DIR,
    SYSTEM_PROMPT,
    USER_PROMPT,
    get_latest_checkpoint,
)
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from rich.console import Console
from rich.prompt import Confirm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers import logging as transformers_logging
from trl import SFTTrainer, setup_chat_format
from trl.trainer.utils import RichProgressCallback

transformers_logging.set_verbosity(transformers_logging.CRITICAL)
transformers_logging.disable_progress_bar()

max_seq_length = 1024

training_arguments = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,  # directory to save and repository id
    num_train_epochs=3,  # number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",  # use fused adamw optimizer
    logging_steps=10,  # log every 10 steps
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    # tf32=True,  # use tf32 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    # push_to_hub=True,  # push model to hub
    # report_to="tensorboard",  # report metrics to tensorboard
)

# TODO: I think lower r is still fine the tuning is working well for even really small datasets
peft_config = LoraConfig(
    lora_alpha=8,
    # lora_dropout=0.05,
    lora_dropout=0,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# TODO: Use HQQ quantization


def train_qlora_model_command():
    """Train a QLoRA model."""
    console = Console()

    # Load the dataset
    dataset = Dataset.load_from_disk(DATASET_OUTPUT_DIR)
    dataset_info = dataset.info

    console.rule("Dataset Information")
    console.print(f"Dataset name: {dataset_info.dataset_name}")
    console.print(f"Dataset description: {dataset_info.description}")
    console.print(f"Dataset size: {len(dataset)}")

    # Check if a model already exists
    try:
        checkpoint = get_latest_checkpoint(MODEL_OUTPUT_DIR)
    except FileNotFoundError:
        checkpoint = None

    if checkpoint:
        overwrite = Confirm.ask(
            f"A model already exists in {MODEL_OUTPUT_DIR}. Do you want to overwrite it?"
        )
        if not overwrite:
            console.print("Exiting.")
            return

    console.rule("Setting up training environment")

    with console.status("[bold green]Creating conversation examples"):
        train_dataset = dataset.map(
            create_conversation, remove_columns=dataset.column_names, batched=False
        )
    console.print("Created conversation examples.")
    console.print(f"Number of examples: {len(train_dataset)}")

    with console.status("[bold cyan]Loading model"):
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map="auto",
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            use_cache=False,
        )
    console.print("Base model loaded")

    with console.status("[bold pink]Setting up chat format"):
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        tokenizer.padding_side = "right"  # to prevent warnings
    console.print("Loaded tokenizer.")

    with console.status("[bold blue]Setting up chat format and peft training"):
        model, tokenizer = setup_chat_format(model, tokenizer)  # type: ignore
        model = prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        quantized_model = get_peft_model(model, peft_config)
    console.print("PEFT model created and ready for training.")

    trainer = SFTTrainer(
        model=quantized_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_arguments,
        peft_config=peft_config,
        # packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
        max_seq_length=max_seq_length,
        callbacks=[RichProgressCallback()],
        # neftune_noise_alpha=5,
    )

    console.print("Initialized SFTTrainer.")
    console.print(f"Model will be saved to {MODEL_OUTPUT_DIR}")

    console.rule("Training QLoRA Model")
    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()  # type: ignore Broken types in TRL

    console.print("[bold green]Training complete.")
    console.print(f"[bold purple]Model saved to {MODEL_OUTPUT_DIR}")


def create_conversation(sample):
    """Create a conversation from a sample in the dataset.

    The conversation consists of a system prompt, a user prompt, and the assistant's response in
    the OpenAI format.

    Args:
        sample: The sample from the dataset.

    Returns:
    A dictionary formatted for the SFTTrainer.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(artist_name=sample["artist_name"]),
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(song_name=sample["song_name"]),
            },
            {"role": "assistant", "content": sample["lyrics"]},
        ]
    }
