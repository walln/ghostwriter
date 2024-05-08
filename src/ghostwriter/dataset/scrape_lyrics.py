"""Aggregate lyrics from an artist's discography."""

import logging
import os

import lyricsgenius
import requests
from rich.console import Console
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()

auth_token = os.getenv("GENIUS_API_TOKEN")
assert auth_token is not None, "Missing GENIUS_API_TOKEN"
genius = lyricsgenius.Genius(auth_token)


def create_row(
    song: dict, *, lyrics: str | None = None, error: str | None = None
) -> dict:
    """Create a row for the dataset.

    Args:
        song: The song information.
        lyrics: The lyrics for the song.
        error: The error message if the lyrics could not be found.

    Returns:
        A dictionary containing the song information.
    """
    return {
        "song_id": song["song_id"],
        "song_name": song["song_name"],
        "lyrics": lyrics if lyrics else "NONE",
        "lyrics_path": song["lyrics_path"],
        "error": error if error else "NONE",
    }


# TODO: docstrings and public functions
def get_lyrics_for_songs(songs: list[dict]):
    """For each song in the list, get the lyrics from the Genius API.

    This is a generator function that yields the song information for each
    song. This is because the song list can be large and because scraping the
    lyrics for each song is IO-bound, it is more efficient to yield the results
    so that the dataset builder can parallelize this process across multiple
    songs at once.

    Args:
        songs: A list of songs to get the lyrics for.

    Returns:
        A list of dictionaries containing the song information.
    """
    for song in songs:
        if song["lyrics_state"] != "complete":
            continue
        else:
            try:
                res = genius.lyrics(song_url="https://genius.com" + song["lyrics_path"])
                yield (
                    create_row(song, error="lyrics not found")
                    if not res
                    else create_row(song, lyrics=res[res.find("\n") + 1 :])
                )
            except Exception as e:
                yield create_row(song, error=str(e))


def get_songs_for_artist(artist_id: str) -> list[dict]:
    """Get the songs for an artist from the Genius API.

    Searches for all songs where the supplied artist is the primary
    artist on the song. The song information is returned as a list of
    dictionaries. Each dictionary contains the song ID, song name,
    lyrics state, and lyrics path.

    Args:
        artist_id: The ID of the artist.

    Returns:
    A list of songs for the artist.
    """
    url = f"https://genius.com/api/artists/{artist_id}/songs"

    table = Table(title="Songs")
    table.add_column("Number", style="cyan", no_wrap=True)
    table.add_column("Song Name", style="magenta")
    table.add_column("Song ID", style="green")

    loading_status = Status("Loading songs...")

    all_songs = []

    page = 1

    loading_status.start()
    while True:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {auth_token}"},
            params={
                "per_page": 50,
                "page": page,
                "sort": "popularity",
            },
        )

        if response.status_code == 200:
            body = response.json()
            songs = body["response"]["songs"]

            if len(songs) == 0:
                loading_status.stop()
                console.print(table)
                return all_songs

            for song in songs:
                all_songs.append(
                    {
                        "song_id": song["id"],
                        "song_name": song["title"],
                        "lyrics_state": song["lyrics_state"],
                        "lyrics_path": song["path"],
                    }
                )
                table.add_row(
                    str(len(all_songs)),
                    str(song["title"]),
                    str(song["id"]),
                )
                loading_status.update(
                    f"Loading songs... {len(all_songs)} song(s) found."
                )
            page += 1

        else:
            raise ValueError("Failed to get lyrics")


def get_artist_information(artist_name: str) -> dict:
    """Get the artist information from the Genius API.

    Searches for the artist with the given name and returns the artist ID and name.
    This function is interactive and will prompt the user to select the correct artist as it
    iteratively searches for the artist corresponding to the given name.

    Args:
        artist_name: The name of the artist to search for.

    Returns:
        A dictionary containing the artist ID and name.
    """
    table = Table(title="Artist")
    table.add_column("Number", style="cyan", no_wrap=True)
    table.add_column("Artist Name", style="magenta")
    table.add_column("Artist ID", style="green")

    page = 1
    artists = []

    while True:
        response = requests.get(
            "https://genius.com/api/search/artist",
            params={"q": artist_name, "page": page},
        )

        if response.status_code == 200:
            body = response.json()
            sections = body["response"]["sections"]

            for section in sections:
                if section["type"] != "artist":
                    continue

                hits = section["hits"]

                if len(hits) == 0:
                    raise ValueError("No artists found")

                for hit in hits:
                    if hit["type"] == "artist":
                        table.add_row(
                            str(len(artists)),
                            str(hit["result"]["name"]),
                            str(hit["result"]["id"]),
                        )
                        artists.append(
                            {
                                "artist_id": hit["result"]["id"],
                                "artist_name": hit["result"]["name"],
                            }
                        )
            # Check if the artist was found
            console.print(table)
            response = Prompt.ask(
                "Press Enter to continue searching or select an artist number to choose an artist"
            )

            if response == "":
                page += 1
                continue
            elif response.isdigit():
                response = int(response)
                if response <= len(artists):
                    console.print("Selected artist:", artists[response]["artist_name"])
                    return {
                        "artist_id": artists[response]["artist_id"],
                        "artist_name": artists[response]["artist_name"],
                    }
                else:
                    console.print("Invalid artist number")
                    print("Invalid artist number")
