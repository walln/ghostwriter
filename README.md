# Ghostwriter

Generating song lyrics with language modeling. Ghostwriter can scrape the internet for song lyrics from your favorite artists and then fine tune large lanauge models to generate lyrics in their style. Ghostwriter leverages QLoRA Supervised Fine Tuning to enable fine tuning large lanauge models on consumer hardware.

## Example

To train a model we need a dataset. Ghostwriter automatically handles creating a dataset in the correct format.

```sh
rye run cli generate-dataset "kendrick lamar"
```

This script will then search for artists matching the supplied name and prompt you to select the correct artist. Then it will scrape all of that artist's lyrics and create a dataset designed for chat prompting.

After generating the dataset you can train a model simply run:

```sh
rye run cli train-model
```

This command will use the generated dataset and being performing QLoRA SFT training to generate lyrics.

Once the model has been trained run the lyric generation script to write new lyrics in that arists style. Here is an excerpt from some lyrics in the style of Kendrick Lamar about corruption.

```txt
[Hook]
See we live in a world, not ruled by laws
But ruled by money, power and hate for more money and fame
I won't say it ain't tough
When you gotta hustle for every coin
You got to ride hard, jump through hoops just to get on up
It's life (Life) is it worth living?
```

These are pretty good and certainly resemble something Kendrick might write.

## Notice

This project is a toy used for research and educational purposes. Lyrical content is copyright to the respective authors and should not be used for commercial purposes.
