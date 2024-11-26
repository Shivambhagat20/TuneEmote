import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth

emotion_to_genre = {
    "Angry": "rock",
    "Disgust": "chill",
    "Fear": "ambient",
    "Happy": "pop",
    "Sad": "acoustic",
    "Surprise": "party",
    "Neutral": "focus"
}


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.environ.get("spotify_client"),
    client_secret=os.environ.get("spotify_secret"),
    redirect_uri="http://localhost:5000/callback",
    scope="user-read-playback-state user-modify-playback-state"
    
))

def suggest_music(emotion):
    genre = emotion_to_genre.get(emotion, "pop")
    recommendations = sp.recommendations(seed_genres=[genre], limit=10)
    return [
        {"name": track["name"], "artist": track["artists"][0]["name"], "uri": track["uri"]}
        for track in recommendations["tracks"]
    ]

def play_song(uri):
    sp.start_playback(uris=[uri])
