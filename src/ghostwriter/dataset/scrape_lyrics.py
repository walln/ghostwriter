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


def _get_lyrics_for_songs(songs: list[dict]):
    for song in songs:
        if song["lyrics_state"] != "complete":
            continue
        else:
            try:
                genius = lyricsgenius.Genius(auth_token)
                song_url = "https://genius.com" + song["lyrics_path"]
                res = genius.lyrics(song_url=song_url)

                if not res:
                    yield {
                        "error": "lyrics not found",
                        "song_id": song["song_id"],
                        "song_name": song["song_name"],
                        "lyrics_path": song["lyrics_path"],
                        "lyrics": "NONE",
                    }

                else:
                    # Drop everything before the first line break
                    res = res[res.find("\n") + 1 :]

                    yield {
                        "song_id": song["song_id"],
                        "song_name": song["song_name"],
                        "lyrics": res,
                        "lyrics_path": song["lyrics_path"],
                        "error": "NONE",
                    }
            except Exception as e:
                yield {
                    "error": str(e),
                    "song_id": song["song_id"],
                    "song_name": song["song_name"],
                    "lyrics_path": song["lyrics_path"],
                    "lyrics": "NONE",
                }


def _get_songs_for_artist(artist_id: str) -> list:
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


def _get_artist_information(artist_name: str) -> dict:
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
