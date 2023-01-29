"""Load an artist's discography as a dataset to train a transfromer."""


from typing import Literal
import tiktoken
import os
import numpy as np

import logging
from logging import Logger

import pickle

logger = Logger(__name__, logging.DEBUG)


class GPTCharacterLevelTokenizer:
    """
    Character level tokenizer with the tiktoken API for encoding and decoding strings.
    Used for fully training the model from scratch because I cannot be bothered
    to implement byte-pair encoding from scratch.

    Parameters
    ----------
    chars
        A sorted list of the set of characters in the training corpus

    """

    def __init__(self, chars: list):
        self.chars = chars
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, input: str):
        """Encode a string to a list of token ids."""
        return [self.stoi[char] for char in input]

    def decode(self, tokens: list[int]):
        """Decode a list of token ids to a string."""
        return "".join([self.itos[token] for token in tokens])


def get_lyrics(data_dir: str) -> str:
    """
    Get an artist's entire discography as a single string.
    Caches the intermediate processing steps in a folder
    /data/{artist}/...

    Parameters
    ----------
    data_dir
        The path to the directory where the data is cached and downloaded

    """
    if not os.path.exists(os.path.join(data_dir, "lyrics.txt")):

        if not os.path.exists(os.path.join(data_dir, "songs.json")):
            print(os.path.join(data_dir, "songs.json"))
            logger.info("Lyrics cache does not exist, will need to redownload lyrics")

            from lyricsgenius import Genius
            import json

            token = os.getenv("GENIUS_API_TOKEN")
            if token is None:
                raise ValueError("Missing GENIUS_API_TOKEN")

            # Download discography by scraping Genius and using API
            genius = Genius(token, retries=5, timeout=30)
            artist = genius.search_artist("Drake", max_songs=1000, get_full_info=False)
            artist.save_lyrics(
                os.path.join(data_dir, "songs.json"),
                overwrite=True,
                sanitize=False,
            )

        output_file_path = os.path.join(data_dir, "lyrics.txt")

        # Get only song lyrics from discogrphy and append to
        # text file as one long document
        with open(os.path.join(data_dir, "songs.json"), "r") as f:
            data = json.load(f)
            songs = data["songs"]
            with open(output_file_path, "w") as output_file:
                for song in songs:
                    lyrics = song["lyrics"]
                    if lyrics is not None:
                        output_file.write(lyrics)

        logger.info(f"Lyrics cached to file: {output_file_path}")

    with open(os.path.join(data_dir, "lyrics.txt"), "r") as f:
        data = f.read()

    return data


def create_or_load_dataset(
    artist: str, model: Literal["GPT"], use_pretrained_tokenizer: bool = True
):
    """
    Attempt to load the dataset for a given artist and download the data if needed.
    Uses model metadata to load and configure a tokenizer for the dataset

    Parameters
    ----------
    artist
        The name of the artist whose discography should be used as the dataset
    model
        The model architecture to use for the tokenizer
    use_pretrained_tokenizer
        If the tokenizer should use the pretrained vocabulary

    """
    data_dir = os.path.join("ghostwriter", "data", artist)

    if model == "GPT":
        if use_pretrained_tokenizer:
            tokenizer = tiktoken.get_encoding("gpt2")

            # Check to see if the dataset exists already
            if not os.path.exists(
                os.path.join(data_dir, "gpt2_pretrained_train.bin")
            ) or not os.path.exists(os.path.join(data_dir, "gpt2_pretrained_val.bin")):

                logger.info("No dataset cache exists, need to create the dataset")

                data = get_lyrics(data_dir)
                n = len(data)
                train_split = data[: int(n * 0.9)]
                val_split = data[int(n * 0.9) :]

                train_ids = tokenizer.encode_ordinary(train_split)
                val_ids = tokenizer.encode_ordinary(val_split)

                train_ids = np.array(train_ids, dtype=np.uint16)
                val_ids = np.array(val_ids, dtype=np.uint16)

                logger.info(f"Training split has {len(train_ids)} tokens")
                logger.info(f"Validation split has {len(val_ids)} tokens")

                train_ids.tofile(os.path.join(data_dir, "gpt2_pretrained_train.bin"))
                val_ids.tofile(os.path.join(data_dir, "gpt2_pretrained_val.bin"))

            train_data = np.memmap(
                os.path.join(data_dir, "gpt2_pretrained_train.bin"),
                dtype=np.uint16,
                mode="r",
            )
            val_data = np.memmap(
                os.path.join(data_dir, "gpt2_pretrained_val.bin"),
                dtype=np.uint16,
                mode="r",
            )

            return train_data, val_data, tokenizer

        else:
            if (
                not os.path.exists(os.path.join(data_dir, "gpt2_scratch_tokenizer.pkl"))
                or not os.path.exists(os.path.join(data_dir, "gpt2_scratch_train.bin"))
                or not os.path.exists(os.path.join(data_dir, "gpt2_scratch_val.bin"))
            ):
                data = get_lyrics(data_dir)
                chars = sorted(list(set(data)))

                tokenizer = GPTCharacterLevelTokenizer(chars)

                with open(
                    os.path.join(data_dir, "gpt2_scratch_tokenizer.pkl"), "wb"
                ) as f:
                    pickle.dump(tokenizer, f)

                n = len(data)
                train_data = data[: int(n * 0.9)]
                val_data = data[int(n * 0.9) :]

                train_ids = tokenizer.encode(train_data)
                val_ids = tokenizer.encode(val_data)

                train_ids = np.array(train_ids, dtype=np.uint16)
                val_ids = np.array(val_ids, dtype=np.uint16)
                train_ids.tofile(os.path.join(data_dir, "gpt2_scratch_train.bin"))
                val_ids.tofile(os.path.join(data_dir, "gpt2_scratch_val.bin"))

            with open(os.path.join(data_dir, "gpt2_scratch_tokenizer.pkl"), "rb") as f:
                tokenizer = pickle.load(f)

            train_data = np.memmap(
                os.path.join(data_dir, "gpt2_scratch_train.bin"),
                dtype=np.uint16,
                mode="r",
            )
            val_data = np.memmap(
                os.path.join(data_dir, "gpt2_scratch_val.bin"),
                dtype=np.uint16,
                mode="r",
            )

            return train_data, val_data, tokenizer

    pass
