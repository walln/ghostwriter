"""Command to generate lyrics using the PEFT model."""

import torch
from ghostwriter.model.common import (
    MODEL_OUTPUT_DIR,
    SYSTEM_PROMPT,
    get_artist_name,
    get_latest_checkpoint,
)
from peft.auto import AutoPeftModelForCausalLM
from peft.peft_model import PeftModelForCausalLM
from rich.console import Console
from rich.prompt import Confirm
from transformers import AutoTokenizer, pipeline
from transformers import logging as transformers_logging

transformers_logging.set_verbosity(transformers_logging.CRITICAL)
transformers_logging.disable_progress_bar()


def generate_lyrics_command():
    """Generate lyrics using the PEFT model."""
    console = Console()

    peft_model_id = get_latest_checkpoint(MODEL_OUTPUT_DIR)
    console.print(f"Using PEFT model: {peft_model_id}")

    artist_name = get_artist_name()
    console.print(f"Generating lyrics for artist: {artist_name}")

    user_message = Confirm.get_input(
        console,
        "What instructions should the model follow when generating lyrics?",
        password=False,
    )
    user_message = user_message.strip() if user_message else "Write a song about love."

    user_message = "Write a song about the problems in the music industry."

    with console.status("[bold green]Loading the tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    with console.status("[bold cyan]Loading the model"):
        model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    console.print("Model loaded.")

    assert isinstance(
        model, PeftModelForCausalLM
    ), f"Model is not a PEFT model. found {type(model)}"

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(artist_name=artist_name),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert isinstance(prompt, str), f"Prompt is not a string. found {type(prompt)}"

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)  # type: ignore PeftModel extends ModelForCausalLM
    assert pipe.tokenizer is not None

    prompt = pipe.tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(artist_name=artist_name),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    with console.status("[bold purple]Generating lyrics"):
        outputs = pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=1,
            # top_k=25,
            top_p=0.95,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )

    assert isinstance(outputs, list)
    console.rule("Generated Lyrics")
    console.print(outputs[0]["generated_text"][len(prompt) :].strip())
