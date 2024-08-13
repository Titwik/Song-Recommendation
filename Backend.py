import pandas as pd
import json
import numpy as np
import os
from flask import Flask, session, redirect, url_for, request
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler

# set the client id, client secret, redirect_uri and scope for project
client_id = "CLIENT_ID"
client_secret = "CLIENT_SECRET"
redirect_uri = "http://localhost:5000/callback"
scope = "user-library-read"

# intialize the spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

# Load the JSON file
with open('/home/titwik/Projects/Spotify Project/song_dataset/data/mpd.slice.0-999.json', 'r') as f:
    data = json.load(f)

# Extract playlists
playlists = data['playlists']

# initialise a list to store track uri's
track_uri = []

# loop through the tracks in each playlist 
for playlist in playlists:
    for track in playlist['tracks']:
        track_uri.append(track['track_uri'])

# Display the DataFrame
def display_track_name(track_uri):
    try:
        # Fetch the track's information
        track_info = sp.track(track_uri)
        # Extract and print the track name
        track_name = track_info['name']
        print(f"Track Name: {track_name}")
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

for i in range(50):
    track = track_uri[i]
    display_track_name(track)




