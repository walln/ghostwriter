import os
import tiktoken
import numpy as np

def sanitize_lyrics(lyrics: str) -> str:
    """
    Lots of popovers and other strange text appears due to the structure of the scraped data
    Just remove the most common issues
    """
    lyrics = lyrics.replace("You might also like", "")
    lyrics = lyrics.replace("See Drake LiveGet tickets as low as $99", "")
    lyrics = lyrics.replace("Embed", "")
    return lyrics

if not os.path.exists(os.path.join(os.path.dirname(__file__),'lyrics.txt')):

    from lyricsgenius import Genius
    import json

    token = os.getenv("GENIUS_API_TOKEN")
    if token is None:
        raise ValueError("Missing GENIUS_API_TOKEN")

    genius = Genius(token, retries=5, timeout=30)
    artist = genius.search_artist("Drake", max_songs=1000, get_full_info=False)
    print(artist.songs)
    artist.save_lyrics(os.path.join(os.path.dirname(__file__), "songs.json"), overwrite=True, sanitize=False)

    with open(os.path.join(os.path.dirname(__file__), "songs.json"), "r") as f:
        data = json.load(f)
        print(data["songs"][0]["lyrics"])

        songs = data["songs"]
        with open(os.path.join(os.path.dirname(__file__), "lyrics.txt"), "w") as output_file:
            for song in songs:
                lyrics = song["lyrics"]
                if lyrics is not None:
                    lyrics = sanitize_lyrics(lyrics=lyrics)
                    output_file.write(lyrics)

with open(os.path.join(os.path.dirname(__file__), "lyrics.txt"), 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 346,791 tokens
# val has 41,396 tokens